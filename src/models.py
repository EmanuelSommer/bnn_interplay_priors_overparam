"""Flax model definitions."""
from flax import linen as nn
from jax import numpy as jnp

class MLPModelUCI(nn.Module):
    depth: int = 3
    width: int = 16
    activation: str = "relu"
    use_bias: bool = True

    def setup(self) -> None:
        if self.activation == "identity":
            self.activation_fn = lambda x: x
        else:
            self.activation_fn = getattr(nn, self.activation)
        return super().setup()

    @nn.compact
    def __call__(self, x,):
        for _ in range(self.depth):
            x = nn.Dense(self.width, use_bias=self.use_bias)(x)
            x = self.activation_fn(x)
        x = nn.Dense(1)(x)
        return x