"""Dataset-agnostic Parkinson EEG feature pipeline.

The package deliberately keeps the canonical input contract separate from the
older, study-specific runners.  A dataset adapter produces one compact
recordings table; analyses then consume only that table and cleaned FIF
epochs.
"""

from .schema import CanonicalRecording, GlobalConfig, load_global_config

__all__ = ["CanonicalRecording", "GlobalConfig", "load_global_config"]
