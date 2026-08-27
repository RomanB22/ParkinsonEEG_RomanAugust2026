"""Shared, explicit Montreal Cognitive Assessment status classification."""

from __future__ import annotations

import numpy as np
import pandas as pd


IMPAIRED_LABEL = "cognitive_impairment"
NORMAL_LABEL = "cognitively_normal"


def classify_moca(
    scores: pd.Series,
    *,
    impairment_below: float = 26.0,
    normal_minimum: float = 26.0,
    normal_maximum: float = 30.0,
) -> pd.Series:
    """Classify valid MoCA scores without changing continuous analyses."""
    values = pd.to_numeric(scores, errors="raise").astype(float)
    if impairment_below != normal_minimum:
        raise ValueError("The impairment cutoff and normal-range minimum must agree")
    finite = values[np.isfinite(values)]
    if ((finite < 0.0) | (finite > normal_maximum)).any():
        raise ValueError(f"MOCA scores must be between 0 and {normal_maximum:g}")
    status = pd.Series(pd.NA, index=values.index, dtype="string")
    status.loc[values.lt(impairment_below)] = IMPAIRED_LABEL
    status.loc[values.between(normal_minimum, normal_maximum, inclusive="both")] = (
        NORMAL_LABEL
    )
    return status
