# MaxViT: Multi-Axis Vision Transformer

**Work in progress!**

Unofficial **PyTorch** reimplementation of the
paper [MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/pdf/2204.01697.pdf)
by Zhengzhong Tu et al. (Google Research).

## Installation

You can simply install the MaxViT implementation as a Python package by using `pip`.

```shell script
pip install git+https://github.com/ChristophReich1996/MaxViT
```

Alternatively, you can clone the repository and use the implementation in [maxvit](maxvit) directly in your project.

## Usage

```python



```

In case you want to use a custom configuration you can use the `MaxViT` class. The constructor method takes 
the following parameters.

| Parameter | Description | Type |
| ------------- | ------------- | ------------- |
| in_channels | Number of input channels | int |

[This file](example.py) includes a full example how to use this implementation.

This repository also provides an [image classification training script](image_classification/main.py) for [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [Places365](https://www.cs.toronto.edu/~kriz/cifar.html).

## Disclaimer

This is a very experimental implementation only based on the [MaxViT paper](https://arxiv.org/pdf/2204.01697.pdf).
Since an official implementation of the MaxViT is not yet published, it is not possible to say to which extent this implementation might differ from the original one. If you have any issues with this implementation please raise an issue.

## Reference

```bibtex
@article{Liu2021,
    title={{MaxViT: Multi-Axis Vision Transformer}},
    author={Tu, Zhengzhong and Talebi, Hossein and Zhang, Han and Yang, Feng and Milanfar, Peyman and Bovik, Alan 
            and Li, Yinxiao}
    journal={arXiv preprint arXiv:2204.01697},
    year={2022}
}
```
