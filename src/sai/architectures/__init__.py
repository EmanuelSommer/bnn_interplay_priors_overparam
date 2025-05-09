"""Module for Registering new models."""

from src.sai.architectures.tabular import FCN
from src.sai.architectures.image import LeNet, LeNetti, TinyResNet

__all__ = [
    "LeNet",
    "LeNetti",
    "TinyResNet",
    "FCN",
]