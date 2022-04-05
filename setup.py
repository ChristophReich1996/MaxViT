from setuptools import setup

setup(
    name="maxvit",
    version="0.1",
    url="https://github.com/ChristophReich1996/MaxViT",
    license="MIT License",
    author="Christoph Reich",
    author_email="ChristophReich@gmx.net",
    description="PyTorch MaxViT",
    packages=["maxvit"],
    install_requires=["torch>=1.7.0", "timm>=0.4.12"],
)
