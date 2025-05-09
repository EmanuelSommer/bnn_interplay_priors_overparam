"""Basic ConvNN building blocks and models."""

from typing import Callable

import jax.numpy as jnp
from flax import linen as nn

import src.sai.config.models.image as cfg


class LeNet(nn.Module):
    """Implementation of LeNet."""

    config: cfg.LeNetConfig

    def setup(self):
        """Initialize the fully connected neural network."""
        self.core = LeNetCore(
            activation=self.config.activation.flax_activation,
            out_dim=self.config.out_dim,
            use_bias=self.config.use_bias,
        )

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass."""
        return self.core(x)


class LeNetCore(nn.Module):
    """Core Implementation of LeNet."""

    activation: Callable
    out_dim: int
    use_bias: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward pass.

        Args:
            x (jnp.ndarray): The input data of
            shape (batch_size, channels, height, width).
        """
        x = nn.Conv(
            features=6, kernel_size=(5, 5), strides=(1, 1), padding=2, name="conv1"
        )(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = nn.Conv(
            features=16, kernel_size=(5, 5), strides=(1, 1), padding=0, name="conv2"
        )(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=120, use_bias=self.use_bias, name="fc1")(x)
        x = self.activation(x)
        x = nn.Dense(features=84, use_bias=self.use_bias, name="fc2")(x)
        x = self.activation(x)
        x = nn.Dense(features=self.out_dim, use_bias=self.use_bias, name="fc3")(x)
        return x


class LeNetti(nn.Module):
    """A super simple LeNet version."""

    config: cfg.LeNettiConfig

    def setup(self):
        """Initialize the fully connected neural network."""
        self.core = LeNettiCore(
            activation=self.config.activation.flax_activation,
            out_dim=self.config.out_dim,
            use_bias=self.config.use_bias,
        )

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass."""
        return self.core(x)


class LeNettiCore(nn.Module):
    """Core Implementation of LeNetti."""

    activation: Callable[..., jnp.ndarray]
    out_dim: int
    use_bias: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward pass.

        Args:
            x (jnp.ndarray): The input data of
            shape (batch_size, channels, height, width).
        """
        x = nn.Conv(
            features=1, kernel_size=(3, 3), strides=(1, 1), padding=2, name="conv1"
        )(x)
        x = self.activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=8, use_bias=self.use_bias, name="fc1")(x)
        x = self.activation(x)
        x = nn.Dense(features=8, use_bias=self.use_bias, name="fc2")(x)
        x = self.activation(x)
        x = nn.Dense(features=8, use_bias=self.use_bias, name="fc3")(x)
        x = self.activation(x)
        x = nn.Dense(features=self.out_dim, use_bias=self.use_bias, name="fc4")(x)
        return x


class BasicBlock(nn.Module):
    """Basic ResNet block with two convolutional layers."""

    features: int
    stride: int = 1
    use_downsample: bool = False
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass for the BasicBlock.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            train: Whether the model is in training mode.

        Returns:
            Output tensor of the same shape as input.
        """
        identity = x

        # First convolution
        x = nn.Conv(
            self.features, (3, 3), strides=self.stride, padding="SAME", use_bias=False
        )(x)
        x = self.activation(x)

        # Second convolution
        x = nn.Conv(self.features, (3, 3), strides=1, padding="SAME", use_bias=False)(x)

        # Downsample if required
        if self.use_downsample:
            identity = nn.Conv(
                self.features, (1, 1), strides=self.stride, use_bias=False
            )(identity)

        x += identity
        x = self.activation(x)
        return x


class ResNetCore(nn.Module):
    """Core implementation of ResNet."""

    block_fn: Callable
    layers: list[int]
    num_classes: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass through ResNet.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            train: Whether the model is in training mode.

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        # TODO 7x7 + stride 2 for ImageNet
        x = nn.Conv(64, (3, 3), strides=1, padding="SAME", use_bias=False)(x)
        x = self.activation(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        # Residual layers
        x = self._make_layer(64, self.layers[0], stride=1, train=train)(x)
        x = self._make_layer(128, self.layers[1], stride=2, train=train)(x)
        x = self._make_layer(256, self.layers[2], stride=2, train=train)(x)
        x = self._make_layer(512, self.layers[3], stride=2, train=train)(x)

        x = jnp.mean(x, axis=(1, 2))  # pool over width and height
        x = nn.Dense(self.num_classes)(x)
        return x

    def _make_layer(
        self, features: int, blocks: int, stride: int, train: bool
    ) -> nn.Sequential:
        """Creates a layer of residual blocks.

        Args:
            features: Number of output features for the blocks.
            blocks: Number of blocks in the layer.
            stride: Stride for the first block in the layer.

        Returns:
            A sequential module containing the residual blocks.
        """

        def layer_fn(x: jnp.ndarray) -> jnp.ndarray:
            # First block with stride
            x = BasicBlock(features, stride, use_downsample=(stride != 1))(
                x, train=train
            )
            # Remaining blocks
            for _ in range(1, blocks):
                x = BasicBlock(features)(x, train=train)
            return x

        return layer_fn


class TinyResNetCore(nn.Module):
    """Core implementation of a tiny ResNet."""

    block_fn: Callable
    layers: list[int]
    num_classes: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass through ResNet.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            train: Whether the model is in training mode.

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        x = nn.Conv(64, (3, 3), strides=1, padding="SAME", use_bias=False)(x)
        # x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        x = BasicBlock(64)(x, train=train)

        x = jnp.mean(x, axis=(1, 2))  # pool over width and height
        x = nn.Dense(self.num_classes)(x)
        return x


class TinyResNet(nn.Module):
    """A tiny ResNet model."""

    config: cfg.TinyResNetConfig

    def setup(self):
        """Set up the ResNetTiny model."""
        self.core = TinyResNetCore(
            block_fn=BasicBlock,
            layers=self.config.layers,
            num_classes=self.config.out_dim,
            activation=self.config.activation.flax_activation,
        )

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass through ResNetTiny.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            train: Whether the model is in training mode.

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        return self.core(x, train)


class ResNet(nn.Module):
    """ResNet implementation.

    Inspired by https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """

    config: cfg.ResNetConfig

    def setup(self):
        """Set up the ResNet18 model."""
        self.core = ResNetCore(
            block_fn=BasicBlock,
            layers=self.config.layers,
            num_classes=self.config.out_dim,
            activation=self.config.activation.flax_activation,
        )

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass through ResNet18.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            train: Whether the model is in training mode.

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        return self.core(x, train)
