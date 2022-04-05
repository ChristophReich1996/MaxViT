from typing import Type

import torch
import torch.nn as nn

from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv
from timm.models.layers import drop_path


def _gelu_ignore_parameters(*args, **kwargs) -> nn.Module:
    return nn.GELU()


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            x (torch.Tensor): Input tensor of the shape [B, C_in, H, W]

        Returns:
            x (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)] (downscaling is optional)
        """
        shortcut = self.skip_path(x)
        x = self.main_path(x)
        if self.drop_path_rate > 0.:
            x = drop_path(x, self.drop_path_rate, self.training)
        x += shortcut
        return x


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            x (torch.Tensor): Input tensor of the shape [B, C_in, H, W]

        Returns:
            x (torch.Tensor): Output tensor of the shape [B, C_out, H // 2, W // 2]
        """
        return x


class MaxViT(nn.Module):
    pass


if __name__ == '__main__':
    block = MBConv(in_channels=32, out_channels=64, downscale=True)
    output = block(torch.rand(2, 32, 16, 16))
    print(output.shape)
