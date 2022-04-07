from typing import Type, Callable, Tuple

import torch
import torch.nn as nn

from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv
from timm.models.layers import drop_path


def _gelu_ignore_parameters(
        *args,
        **kwargs
) -> nn.Module:
    """ Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Returns:
        activation (nn.Module): GELU activation function.
    """
    activation = nn.GELU()
    return activation


class MBConv(nn.Module):
    """ MBConv block as described in: https://arxiv.org/pdf/2204.01697.pdf

        Without downsampling:
        x ← x + Proj(SE(DWConv(Conv(Norm(x)))))

        With downsampling:
        x ← Proj(Pool2D(x)) + Proj(SE(DWConv ↓(Conv(Norm(x))))).

        Conv is a 1 X 1 convolution followed by a Batch Normalization layer and a GELU activation.
        SE is the Squeeze-Excitation layer.
        Proj is the shrink 1 X 1 convolution.

        Note: This implementation differs slightly from the original MobileNet implementation!

    Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            downscale (bool): If true downscale by a factor of two is performed. Default: False
            act_layer (Type[nn.Module]): Type of activation layer to be utilized. Default: nn.GELU
            norm_layer (Type[nn.Module]): Type of normalization layer to be utilized. Default: nn.BatchNorm
            drop_path_rate (float): Dropout rate to be applied during training. Default 0.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            drop_path_rate: float = 0.,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MBConv, self).__init__()
        # Save parameter
        self.drop_path_rate: float = drop_path_rate
        # Check parameters for downscaling
        if not downscale:
            assert in_channels == out_channels, "If downscaling is utilized input and output channels must be equal."
        # Ignore inplace parameter if GELU is used
        if act_layer == nn.GELU:
            act_layer = _gelu_ignore_parameters
        # Make main path
        self.main_path = nn.Sequential(
            norm_layer(in_channels),
            DepthwiseSeparableConv(in_chs=in_channels, out_chs=out_channels, stride=2 if downscale else 1,
                                   act_layer=act_layer, norm_layer=norm_layer, drop_path_rate=drop_path_rate),
            SqueezeExcite(in_chs=out_channels, rd_ratio=0.25),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1))
        )
        # Make skip path
        self.skip_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        ) if downscale else nn.Identity()

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass.

        Args:
            x (torch.Tensor): Input tensor of the shape [B, C_in, H, W].

        Returns:
            x (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)] (downscaling is optional).
        """
        shortcut = self.skip_path(x)
        x = self.main_path(x)
        if self.drop_path_rate > 0.:
            x = drop_path(x, self.drop_path_rate, self.training)
        x += shortcut
        return x


def window_partition(
        input: torch.Tensor,
        window_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Window partition function.

    Args:
        input (torch.Tensor): Input tensor of the shape [B, C, H, W].
        window_size (Tuple[int, int]): Window size to be applied.

    Returns:
        windows (torch.Tensor): Unfolded input tensor of the shape [B * windows, window_size[0], window_size[1], C].
    """
    # Get size of input
    B, C, H, W = input.shape
    # Unfold input
    windows = input.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    # Permute and reshape to [B * windows, window_size[0], window_size[1], channels]
    windows = windows.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(
        windows: torch.Tensor,
        original_size: Tuple[int, int],
        window_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Reverses the window partition.

    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0], window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int]): Window size to be applied.

    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]]
    """
    # Get height and width
    H, W = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return output


def grid_partition(
        input: torch.Tensor,
        grid_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Grid partition function.

    Args:
        input (torch.Tensor): Input tensor of the shape [B, C, H, W].
        grid_size (Tuple[int, int]): Grid size to be applied.

    Returns:
        grid (torch.Tensor): Unfolded input tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
    """
    # Get size of input
    B, C, H, W = input.shape
    # Unfold input
    windows = input.view(B, C, H // grid_size[0], grid_size[0], W // grid_size[1], grid_size[1])
    # Permute and reshape [B * (H // grid_size[0]) * (W // grid_size[1]), grid_size[0], window_size[1], C]
    windows = windows.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return windows


def grid_reverse(
        grid: torch.Tensor,
        original_size: Tuple[int, int],
        grid_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Reverses the grid partition.

    Args:
        Grid (torch.Tensor): Grid tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        grid_size (Tuple[int, int]): Grid size to be applied.

    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]]
    """
    # Get height, width, and channels
    (H, W), C = original_size, grid.shape[-1]
    # Compute original batch size
    B = int(grid.shape[0] / (H * W / grid_size[0] / grid_size[1]))
    # Fold grid tensor
    output = grid.view(B, grid_size[0], grid_size[1], H // grid_size[0], W // grid_size[1], C)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, H, W)
    return output


class SelfAttention(nn.Module):

    def __init__(
            self,
            in_channels: int,
            partition_function: Callable,
            reverse_function: Callable,
            num_heads: int = 32,
    ) -> None:
        # Call super constructor
        super(SelfAttention, self).__init__()


class MaxViTBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            num_heads: int = 32,
            act_layer=nn.GELU,
            norm_layer=nn.BatchNorm2d,
            drop_path_rate: float = 0.
    ) -> None:
        # Call super constructor
        super(MaxViTBlock, self).__init__()
        # Init MBConv block
        self.mb_conv = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            downscale=downscale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path_rate=drop_path_rate
        )
        # Init Block and Grid Attention
        self.block_attention = nn.Identity()
        self.grid_attention = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            x (torch.Tensor): Input tensor of the shape [B, C_in, H, W]

        Returns:
            x (torch.Tensor): Output tensor of the shape [B, C_out, H // 2, W // 2]
        """
        x = self.grid_attention(self.block_attention(self.mb_conv(x)))
        return x


class MaxViT(nn.Module):
    pass


if __name__ == '__main__':
    input = torch.rand(7, 3, 14, 14)
    windows = window_partition(input=input)
    windows = window_reverse(windows=windows, window_size=(7, 7), original_size=input.shape[2:])
    print(torch.all(input == windows))
    grid = grid_partition(input=input)
    grid = grid_reverse(grid=grid, grid_size=(7, 7), original_size=input.shape[2:])
    print(torch.all(input == grid))
