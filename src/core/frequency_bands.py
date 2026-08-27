"""Repository-wide canonical EEG frequency bands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


CANONICAL_FREQUENCY_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}
CANONICAL_BAND_NAMES: tuple[str, ...] = tuple(CANONICAL_FREQUENCY_BANDS)
CANONICAL_BOUT_BAND_NAMES: tuple[str, ...] = CANONICAL_BAND_NAMES[1:]
CANONICAL_BAND_LABELS: dict[str, str] = {
    name: name.title() for name in CANONICAL_BAND_NAMES
}


def validate_frequency_bands(
    bands: Mapping[str, Sequence[Any]],
    *,
    context: str = "bands",
    expected_names: Sequence[str] = CANONICAL_BAND_NAMES,
) -> None:
    """Require canonical names, order, and limits for a requested band subset."""
    expected_names = tuple(str(name) for name in expected_names)
    observed_names = tuple(str(name) for name in bands)
    if observed_names != expected_names:
        raise ValueError(
            f"{context} must be ordered as {list(expected_names)}"
        )
    observed = {
        str(name): tuple(float(value) for value in limits)
        for name, limits in bands.items()
    }
    expected = {
        name: CANONICAL_FREQUENCY_BANDS[name] for name in expected_names
    }
    if observed != expected:
        raise ValueError(
            f"{context} must use the canonical limits "
            f"{expected}"
        )
