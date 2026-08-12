"""
CISSN: Conformally Calibrated Interpretable State-Space Networks
"""
import os

# Reserve one logical core for the main process unless the user specifies a
# joblib limit. This avoids joblib's unavailable Windows physical-core probe.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max((os.cpu_count() or 1) - 1, 1)))

from cissn.constants import STRUCTURED_STATE_DIM

__version__ = "0.1.0"
__all__ = ["STRUCTURED_STATE_DIM"]
