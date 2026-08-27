"""Repository-wide plotting policy and figure-output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


SCALAR_COLORMAP = "viridis"
GROUP_ORDER = ("PD", "Control")
GROUP_COLORS = {"PD": "#D55E00", "Control": "#0072B2"}


def save_figure(fig: Any, path: str | Path, dpi: int) -> None:
    """Save a tightly bounded non-interactive figure and release its memory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

