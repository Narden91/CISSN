"""Device selection with an explicit, auditable CPU fallback.

The default PyPI torch wheel is CPU-only on Windows, so a machine with a
working GPU can silently run every experiment on the CPU. Selection therefore
warns loudly when it falls back, and ``require_gpu`` turns the fallback into a
hard error for grid runs that must not start without a GPU.
"""

import logging
import os
import warnings

import torch

logger = logging.getLogger(__name__)


def select_device(require_gpu: bool = False) -> torch.device:
    """Return the CUDA device when available, otherwise fall back to the CPU.

    Args:
        require_gpu: Raise instead of falling back to the CPU. Also enabled by
            setting the ``CISSN_REQUIRE_GPU=1`` environment variable, which lets
            a whole grid be guarded without editing every command.

    Raises:
        RuntimeError: If no GPU is usable and a GPU was required.
    """
    require_gpu = require_gpu or os.environ.get("CISSN_REQUIRE_GPU", "") == "1"

    # device_count() rather than is_available(): with CUDA_VISIBLE_DEVICES=""
    # availability can still report True while no device is actually selectable.
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda")
        logger.info(
            "Using GPU: %s (torch %s, capability %s)",
            torch.cuda.get_device_name(0),
            torch.__version__,
            ".".join(str(v) for v in torch.cuda.get_device_capability(0)),
        )
        return device

    # No CUDA: distinguish "no GPU in this machine" from the much more common
    # "GPU is present but the installed torch build cannot use it".
    if "+cpu" in torch.__version__ or torch.version.cuda is None:
        reason = (
            f"torch {torch.__version__} is a CPU-only build. Reinstall a CUDA "
            "build, e.g. `uv pip install torch --index-url "
            "https://download.pytorch.org/whl/cu128`."
        )
    else:
        reason = (
            f"torch {torch.__version__} was built against CUDA "
            f"{torch.version.cuda} but no GPU is visible to this process "
            f"(device_count={torch.cuda.device_count()}); check drivers and "
            "CUDA_VISIBLE_DEVICES."
        )

    if require_gpu:
        raise RuntimeError(f"GPU required but unavailable: {reason}")

    warnings.warn(
        f"No GPU available - falling back to CPU, which is far slower. {reason}",
        RuntimeWarning,
        stacklevel=2,
    )
    logger.warning("Falling back to CPU. %s", reason)
    return torch.device("cpu")
