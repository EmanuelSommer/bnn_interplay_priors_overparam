"""What to load from training part."""

from typing import Callable, Optional, TypeAlias

from blackjax import (  # maybe wrap them in kernels.py for cleaner sampling
    hmc,
    mclmc,
    nuts,
)
from src.sai.kernels.sghmc import (
    SGHMC,
    AdaSGHMCWarmup,
)

__all__ = ["nuts", "hmc", "mclmc", "adasghmc"]

KernelRegistry: TypeAlias = dict[str, Optional[Callable]]

KERNELS: KernelRegistry = {
    "nuts": nuts,
    "hmc": hmc,
    "mclmc": mclmc,
    "adasghmc": SGHMC,

}

WARMUP_KERNELS: KernelRegistry = {
    "adasghmc": AdaSGHMCWarmup,
}
