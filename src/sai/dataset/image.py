"""DataLoader Implementations for Image Data."""

import jax
import jax.numpy as jnp

from src.sai.config.data import DataConfig, DatasetType, Source
from src.sai.dataset.base import BaseLoader
from src.sai.types import PRNGKey


class ImageLoader(BaseLoader):
    """DataLoader for Image data."""

    def __init__(self, config: DataConfig, rng_key: PRNGKey, n_chains: int):
        """__init__ method for the ImageLoader class."""
        assert config.data_type == DatasetType.IMAGE
        super().__init__(config=config, rng_key=rng_key, n_chains=n_chains)

    def load_data(self):
        """Load the dataset from different sources."""
        if self.config.source == Source.TORCHVISION:
            data_x, data_y, test_size = _get_torchvision_data(self._name, self._dir)
            non_test_x, non_test_y = self.shuffle_arrays(
                data_x[:-test_size], data_y[:-test_size]
            )
            data_x = jnp.concatenate(
                [
                    non_test_x,
                    data_x[-test_size:],
                ],
                axis=0,
            )
            data_y = jnp.concatenate(
                [
                    non_test_y,
                    data_y[-test_size:],
                ],
                axis=0,
            )

            # append loading_permutations for not permuted test data
            self.loading_permutation = jnp.concatenate(
                [self.loading_permutation, jnp.arange(data_x.shape[0])[-test_size:]],
                axis=0,
            )

            # Limit the number of datapoints
            data_x = data_x[: self.config.datapoint_limit]
            data_y = data_y[: self.config.datapoint_limit]

            # normalize
            if self.config.normalize:
                data_x = data_x / 255.0

            if self.config.flatten:
                data_x = jax.vmap(jnp.ravel)(data_x)

            return data_x, data_y

        raise NotImplementedError(
            f"Source {self.config.source} is not supported for image data."
        )


def _get_torchvision_data(name: str, dir: str) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Get torchvision datasets."""
    from torchvision import datasets, transforms

    match name:
        case "mnist":
            d_train = datasets.MNIST(
                dir, train=True, download=True, transform=transforms.ToTensor()
            )
            d_test = datasets.MNIST(
                dir, train=False, download=True, transform=transforms.ToTensor()
            )
        case "fashion_mnist":
            d_train = datasets.FashionMNIST(
                dir, train=True, download=True, transform=transforms.ToTensor()
            )
            d_test = datasets.FashionMNIST(
                dir, train=False, download=True, transform=transforms.ToTensor()
            )
            # retrieve original test dataset with `test_split: 0.14285`
        case "cifar10":
            d_train = datasets.CIFAR10(
                dir,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        # CIFAR10 mean and std from
                        # https://github.com/kuangliu/pytorch-cifar/issues/19
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                        ),
                    ]
                ),
            )
            d_test = datasets.CIFAR10(
                dir,
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        # CIFAR10 mean and std from
                        # https://github.com/kuangliu/pytorch-cifar/issues/19
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                        ),
                    ]
                ),
            )
            # retrieve original test dataset with `test_split: 0.16666`
        case "cifar100":
            d_train = datasets.CIFAR100(
                dir, train=True, download=True, transform=transforms.ToTensor()
            )
            d_test = datasets.CIFAR100(
                dir, train=False, download=True, transform=transforms.ToTensor()
            )
        case _:
            raise NotImplementedError(f"Dataset {name} is not supported.")

    test_size = d_test.data.shape[0]
    data_x = jnp.concatenate(
        [jnp.array(d_train.data), jnp.array(d_test.data)],
        axis=0,
    )
    if len(data_x.shape) == 3:
        data_x = data_x[:, None, ...]
        data_x = data_x.transpose((0, 2, 3, 1))

    data_y = jnp.concatenate(
        [jnp.array(d_train.targets), jnp.array(d_test.targets)],
        axis=0,
    )
    return data_x, data_y, test_size
