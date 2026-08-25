"""Deterministic demographic matching for the PD/Control sensitivity cohort."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _standardized_mean_difference(first: pd.Series, second: pd.Series) -> float:
    first_values = first.to_numpy(dtype=float)
    second_values = second.to_numpy(dtype=float)
    pooled_sd = np.sqrt(
        (np.var(first_values, ddof=1) + np.var(second_values, ddof=1)) / 2.0
    )
    if np.isclose(pooled_sd, 0.0):
        return 0.0
    return float((np.mean(first_values) - np.mean(second_values)) / pooled_sd)


def demographic_balance(table: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Summarize age and sex balance using transparent descriptive quantities."""
    pd_rows = table.loc[table["group"].eq("PD")]
    control_rows = table.loc[table["group"].eq("Control")]
    return pd.DataFrame.from_records(
        [
            {
                "cohort": cohort,
                "variable": "age_years",
                "pd_mean": float(pd_rows["age_years"].mean()),
                "control_mean": float(control_rows["age_years"].mean()),
                "pd_proportion": np.nan,
                "control_proportion": np.nan,
                "standardized_mean_difference_pd_minus_control": (
                    _standardized_mean_difference(
                        pd_rows["age_years"], control_rows["age_years"]
                    )
                ),
            },
            {
                "cohort": cohort,
                "variable": "sex_male",
                "pd_mean": np.nan,
                "control_mean": np.nan,
                "pd_proportion": float(pd_rows["sex_male"].mean()),
                "control_proportion": float(control_rows["sex_male"].mean()),
                "standardized_mean_difference_pd_minus_control": (
                    _standardized_mean_difference(
                        pd_rows["sex_male"], control_rows["sex_male"]
                    )
                ),
            },
        ]
    )


def match_control_pd_pairs(
    table: pd.DataFrame,
    *,
    maximum_age_difference_years: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Optimally pair every Control to a unique PD subject by exact sex and age.

    The Hungarian assignment minimizes the cohort-wide sum of absolute age
    differences independently within each sex stratum. Pair IDs are retained so
    nested validation can keep both members of every matched pair in one fold.
    """
    required = {"subject_id", "group", "age_years", "sex_male", "target_pd"}
    missing = sorted(required - set(table))
    if missing:
        raise ValueError(f"Matching table is missing columns: {missing}")
    if set(table["group"]) != {"PD", "Control"}:
        raise ValueError("Matching requires both PD and Control subjects")
    if float(maximum_age_difference_years) <= 0:
        raise ValueError("maximum_age_difference_years must be positive")

    pairs = []
    for sex_male in sorted(table["sex_male"].unique()):
        controls = (
            table.loc[
                table["group"].eq("Control") & table["sex_male"].eq(sex_male),
                ["subject_id", "age_years"],
            ]
            .sort_values("subject_id")
            .reset_index(drop=True)
        )
        pd_subjects = (
            table.loc[
                table["group"].eq("PD") & table["sex_male"].eq(sex_male),
                ["subject_id", "age_years"],
            ]
            .sort_values("subject_id")
            .reset_index(drop=True)
        )
        if len(pd_subjects) < len(controls):
            raise ValueError(
                f"Not enough PD subjects to match sex_male={sex_male}: "
                f"PD={len(pd_subjects)}, Control={len(controls)}"
            )
        distances = np.abs(
            controls["age_years"].to_numpy(dtype=float)[:, None]
            - pd_subjects["age_years"].to_numpy(dtype=float)[None, :]
        )
        control_indices, pd_indices = linear_sum_assignment(distances)
        for control_index, pd_index in zip(control_indices, pd_indices):
            difference = float(distances[control_index, pd_index])
            pairs.append(
                {
                    "control_subject_id": str(
                        controls.iloc[control_index]["subject_id"]
                    ),
                    "pd_subject_id": str(pd_subjects.iloc[pd_index]["subject_id"]),
                    "sex_male": int(sex_male),
                    "control_age_years": float(
                        controls.iloc[control_index]["age_years"]
                    ),
                    "pd_age_years": float(pd_subjects.iloc[pd_index]["age_years"]),
                    "absolute_age_difference_years": difference,
                }
            )

    pair_table = pd.DataFrame.from_records(pairs).sort_values(
        ["sex_male", "control_subject_id"]
    ).reset_index(drop=True)
    pair_table.insert(
        0,
        "match_pair_id",
        [f"pair-{index:03d}" for index in range(1, len(pair_table) + 1)],
    )
    outside_caliper = pair_table.loc[
        pair_table["absolute_age_difference_years"]
        > float(maximum_age_difference_years)
    ]
    if not outside_caliper.empty:
        raise ValueError(
            f"{len(outside_caliper)} optimal pairs exceed the "
            f"{maximum_age_difference_years:g}-year age caliper"
        )

    pair_lookup = pd.concat(
        [
            pair_table[["match_pair_id", "control_subject_id"]].rename(
                columns={"control_subject_id": "subject_id"}
            ),
            pair_table[["match_pair_id", "pd_subject_id"]].rename(
                columns={"pd_subject_id": "subject_id"}
            ),
        ],
        ignore_index=True,
    )
    matched = table.merge(pair_lookup, on="subject_id", how="inner", validate="one_to_one")
    matched["cv_group"] = matched["match_pair_id"]
    matched = matched.sort_values(["match_pair_id", "target_pd"]).reset_index(drop=True)
    if len(matched) != 2 * len(pair_table):
        raise RuntimeError("Matched cohort does not contain exactly two subjects per pair")
    if not (matched.groupby("match_pair_id")["target_pd"].nunique() == 2).all():
        raise RuntimeError("Every matched pair must contain one PD and one Control")

    balance = pd.concat(
        [
            demographic_balance(table, "full"),
            demographic_balance(matched, "matched"),
        ],
        ignore_index=True,
    )
    return matched, pair_table, balance


def remove_demographic_predictors(
    models: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Return matched-cohort models without age, sex, or the demographic baseline."""
    result: dict[str, dict[str, object]] = {}
    for model_name, specification in models.items():
        # After age/sex removal ordinal_core and ordinal_adjusted are identical;
        # retain the prespecified adjusted model name and avoid duplicate testing.
        if model_name in {"demographics", "ordinal_core"}:
            continue
        copied = dict(specification)
        copied["features"] = [
            feature
            for feature in specification["features"]  # type: ignore[index]
            if feature not in {"age_years", "sex_male"}
        ]
        copied["label"] = str(specification["label"]).replace(
            " + demographics", ""
        )
        copied["role"] = f"matched_{specification['role']}"
        result[model_name] = copied
    return result
