"""Readable orchestration and shared infrastructure for the EEG project.

Scientific calculations remain in their domain packages (``psd_analysis``,
``ordinal_analysis``, and so on).  This package provides the single public
interface, validated pipeline configuration, dependency graph, and provenance
checks that connect those calculations.
"""

from __future__ import annotations

__version__ = "1.0.0"

