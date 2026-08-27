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


def apply_precomputed_control_pd_pairs(
    table: pd.DataFrame,
    pair_table: pd.DataFrame,
    balance_table: pd.DataFrame,
    *,
    maximum_age_difference_years: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate and apply canonical pairs that were created upstream once."""
    required_table = {
        "subject_id",
        "group",
        "age_years",
        "sex_male",
        "target_pd",
    }
    missing = sorted(required_table - set(table))
    if missing:
        raise ValueError(f"Pre-matched feature table is missing columns: {missing}")
    required_pairs = {
        "match_pair_id",
        "control_subject_id",
        "pd_subject_id",
        "sex_male",
        "control_age_years",
        "pd_age_years",
        "absolute_age_difference_years",
    }
    missing = sorted(required_pairs - set(pair_table))
    if missing:
        raise ValueError(f"Precomputed pair table is missing columns: {missing}")
    required_balance = {
        "cohort",
        "variable",
        "standardized_mean_difference_pd_minus_control",
    }
    missing = sorted(required_balance - set(balance_table))
    if missing:
        raise ValueError(f"Precomputed balance table is missing columns: {missing}")
    if pair_table["match_pair_id"].duplicated().any():
        raise ValueError("Precomputed match pair IDs must be unique")
    subject_columns = ["control_subject_id", "pd_subject_id"]
    paired_subjects = pair_table[subject_columns].astype(str).to_numpy().ravel()
    if len(set(paired_subjects)) != len(paired_subjects):
        raise ValueError("A precomputed matched subject appears in multiple pairs")
    table_subjects = set(table["subject_id"].astype(str))
    if set(paired_subjects) != table_subjects:
        raise ValueError(
            "Precomputed pairs and the matched feature table do not contain "
            "the same subjects"
        )
    if (
        pair_table["absolute_age_difference_years"].to_numpy(dtype=float)
        > float(maximum_age_difference_years)
    ).any():
        raise ValueError("A precomputed pair exceeds the configured age caliper")

    lookup = pd.concat(
        [
            pair_table[["match_pair_id", "control_subject_id"]]
            .rename(columns={"control_subject_id": "subject_id"})
            .assign(expected_group="Control", expected_target=0),
            pair_table[["match_pair_id", "pd_subject_id"]]
            .rename(columns={"pd_subject_id": "subject_id"})
            .assign(expected_group="PD", expected_target=1),
        ],
        ignore_index=True,
    )
    lookup["subject_id"] = lookup["subject_id"].astype(str)
    matched = table.copy()
    matched["subject_id"] = matched["subject_id"].astype(str)
    if "match_pair_id" in matched:
        recorded = matched.set_index("subject_id")["match_pair_id"].astype(str)
        expected = lookup.set_index("subject_id")["match_pair_id"].astype(str)
        if not recorded.sort_index().equals(expected.sort_index()):
            raise ValueError(
                "Participant metadata match_pair_id values disagree with the "
                "canonical pair table"
            )
        matched = matched.drop(columns=["match_pair_id"])
    matched = matched.merge(lookup, on="subject_id", how="left", validate="one_to_one")
    if not matched["group"].eq(matched["expected_group"]).all():
        raise ValueError("Precomputed pair group labels disagree with feature metadata")
    if not matched["target_pd"].eq(matched["expected_target"]).all():
        raise ValueError("Precomputed pair targets disagree with feature metadata")
    matched = matched.drop(columns=["expected_group", "expected_target"])
    matched["cv_group"] = matched["match_pair_id"]
    matched = matched.sort_values(["match_pair_id", "target_pd"]).reset_index(drop=True)

    pair_sex = matched.groupby("match_pair_id")["sex_male"].nunique()
    if not pair_sex.eq(1).all():
        raise ValueError("Precomputed pairs do not match sex exactly")
    observed = matched.pivot(
        index="match_pair_id", columns="group", values="age_years"
    )
    observed_gap = (observed["PD"] - observed["Control"]).abs().sort_index()
    recorded_gap = pair_table.set_index("match_pair_id")[
        "absolute_age_difference_years"
    ].astype(float).sort_index()
    if not np.allclose(observed_gap, recorded_gap, rtol=0.0, atol=1e-12):
        raise ValueError("Precomputed pair age differences disagree with feature metadata")
    expected_balance_rows = {
        ("full", "age_years"),
        ("full", "sex_male"),
        ("matched", "age_years"),
        ("matched", "sex_male"),
    }
    found_balance_rows = set(
        balance_table[["cohort", "variable"]].itertuples(index=False, name=None)
    )
    if not expected_balance_rows.issubset(found_balance_rows):
        raise ValueError("Precomputed balance table lacks full/matched age/sex rows")
    return matched, pair_table.copy(), balance_table.copy()


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
