#!/usr/bin/env python
"""Compare demographic and clinical composition across configured datasets.

The unit for composition statistics is one participant. Session/condition
recordings are retained in a separate count table, while repeated participant
metadata are averaged for numeric clinical variables.
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu

from global_pipeline.converter import convert_config
from global_pipeline.schema import load_global_config


def _bh(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    if valid.any():
        order = np.argsort(values[valid])
        adjusted = values[valid][order] * valid.sum() / np.arange(1, valid.sum() + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        result[np.flatnonzero(valid)[order]] = np.minimum(adjusted, 1.0)
    return result


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _read_canonical(config, dataset_ids: list[str] | None) -> pd.DataFrame:
    path = config.output_root / "canonical" / "recordings.csv.gz"
    if path.exists():
        table = pd.read_csv(path)
    else:
        table = convert_config(config, dataset_ids=dataset_ids)
    if dataset_ids:
        table = table.loc[table["dataset_id"].isin(dataset_ids)].copy()
    return table


def _metadata_extras(config, dataset_ids: list[str]) -> pd.DataFrame:
    """Return numeric participant metadata not represented in the canonical table."""
    pieces: list[pd.DataFrame] = []
    excluded = {
        "participant_id", "subject_id", "recording_id", "id", "group",
        "diagnosis", "condition", "status", "age", "age_years", "sex",
        "gender", "moca", "mmse", "updrs", "updrs_total",
        "type", "eeg",
    }
    for dataset in config.enabled_datasets:
        if dataset.dataset_id not in dataset_ids or dataset.metadata is None:
            continue
        if not dataset.metadata.exists():
            continue
        separator = "\t" if dataset.metadata.suffix.lower() in {".tsv", ".txt"} else ","
        raw = pd.read_csv(dataset.metadata, sep=separator, dtype=str)
        id_column = next(
            (column for column in ("participant_id", "subject_id", "recording_id", "ID", "id") if column in raw),
            None,
        )
        if id_column is None:
            continue
        frame = pd.DataFrame({"dataset_id": dataset.dataset_id, "participant_id": raw[id_column].astype(str)})
        frame["participant_id"] = frame["participant_id"].map(
            lambda value: value if value.startswith("sub-") else f"sub-{value}"
        )
        for column in raw.columns:
            if column.lower() in excluded:
                continue
            numeric = pd.to_numeric(raw[column], errors="coerce")
            if numeric.notna().sum() >= 3:
                frame[f"metadata__{column}"] = numeric
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _participant_table(recordings: pd.DataFrame, extras: pd.DataFrame) -> pd.DataFrame:
    numeric = ["age_years", "updrs", "moca", "mmse"]
    rows: list[dict[str, object]] = []
    for (dataset_id, participant_id), frame in recordings.groupby(
        ["dataset_id", "participant_id"], dropna=False
    ):
        groups = sorted(str(value) for value in frame["group"].dropna().unique())
        population = "Control" if any(value == "Control" for value in groups) else "PD"
        row: dict[str, object] = {
            "dataset_id": dataset_id,
            "participant_id": participant_id,
            "population": population,
            "groups": ";".join(groups),
            "n_recordings": int(frame["recording_id"].nunique()),
        }
        for column in numeric:
            row[column] = pd.to_numeric(frame[column], errors="coerce").mean()
        for column in ("sex",):
            values = frame[column].dropna().astype(str).str.strip()
            row[column] = values.iloc[0].upper() if not values.empty else np.nan
        rows.append(row)
    participants = pd.DataFrame(rows)
    if extras.empty:
        return participants
    extras = extras.groupby(["dataset_id", "participant_id"], dropna=False).first().reset_index()
    return participants.merge(extras, on=["dataset_id", "participant_id"], how="left")


def _summary_tables(participants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_columns = [
        column for column in participants.columns
        if column in {"age_years", "updrs", "moca", "mmse"} or column.startswith("metadata__")
    ]
    continuous_rows: list[dict[str, object]] = []
    for (dataset_id, population), frame in participants.groupby(["dataset_id", "population"], dropna=False):
        for variable in numeric_columns:
            values = pd.to_numeric(frame[variable], errors="coerce").dropna()
            if values.empty:
                continue
            continuous_rows.append({
                "dataset_id": dataset_id, "population": population, "variable": variable,
                "n": len(values), "missing": int(frame[variable].isna().sum()),
                "mean": values.mean(), "sd": values.std(ddof=1), "median": values.median(),
                "q1": values.quantile(.25), "q3": values.quantile(.75),
                "min": values.min(), "max": values.max(),
            })
    categorical_rows: list[dict[str, object]] = []
    for variable in ("population", "sex"):
        for dataset_id, frame in participants.groupby("dataset_id", dropna=False):
            values = frame[variable].fillna("Missing").astype(str)
            counts = values.value_counts()
            for level, count in counts.items():
                categorical_rows.append({
                    "dataset_id": dataset_id, "variable": variable, "level": level,
                    "n": int(count), "percent": 100.0 * count / len(values),
                })
    return pd.DataFrame(continuous_rows), pd.DataFrame(categorical_rows)


def _comparability_tests(participants: pd.DataFrame, alpha: float = .05) -> pd.DataFrame:
    variables = [
        column for column in participants.columns
        if column in {"age_years", "updrs", "moca", "mmse"} or column.startswith("metadata__")
    ]
    rows: list[dict[str, object]] = []
    datasets = sorted(participants["dataset_id"].unique())
    for variable in ("population", "sex"):
        contingency = pd.crosstab(
            participants["dataset_id"], participants[variable].fillna("Missing")
        ).reindex(datasets, fill_value=0)
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            statistic, p_value, _, _ = chi2_contingency(contingency)
            rows.append({
                "scope": "all", "variable": variable, "dataset_a": "ALL",
                "dataset_b": "ALL", "test": "Chi-square", "statistic": statistic,
                "p_value": p_value,
            })
    strata = ["all"] + sorted(participants["population"].dropna().unique())
    for stratum in strata:
        subset = participants if stratum == "all" else participants.loc[participants["population"].eq(stratum)]
        for variable in variables:
            samples = [
                pd.to_numeric(subset.loc[subset["dataset_id"].eq(dataset), variable], errors="coerce").dropna().to_numpy()
                for dataset in datasets
            ]
            usable = [sample for sample in samples if len(sample) >= 2]
            if len(usable) >= 2:
                statistic, p_value = kruskal(*usable)
                rows.append({"scope": stratum, "variable": variable, "dataset_a": "ALL", "dataset_b": "ALL", "test": "Kruskal-Wallis", "statistic": statistic, "p_value": p_value})
            for left, right in itertools.combinations(datasets, 2):
                a = pd.to_numeric(subset.loc[subset["dataset_id"].eq(left), variable], errors="coerce").dropna().to_numpy()
                b = pd.to_numeric(subset.loc[subset["dataset_id"].eq(right), variable], errors="coerce").dropna().to_numpy()
                if len(a) >= 2 and len(b) >= 2:
                    statistic, p_value = mannwhitneyu(a, b, alternative="two-sided")
                    rows.append({"scope": stratum, "variable": variable, "dataset_a": left, "dataset_b": right, "test": "Mann-Whitney U", "statistic": statistic, "p_value": p_value})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_fdr_bh"] = np.nan
        for _, indices in result.groupby(["scope", "test"], sort=False).groups.items():
            result.loc[indices, "q_fdr_bh"] = _bh(result.loc[indices, "p_value"].to_numpy(float))
        result["significant_fdr"] = result["q_fdr_bh"].lt(alpha)
    return result


def _plot_composition(participants: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    datasets = sorted(participants["dataset_id"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    population = pd.crosstab(participants["dataset_id"], participants["population"]).reindex(datasets, fill_value=0)
    population.plot.bar(stacked=True, ax=axes[0], title="Participants by population")
    sex = pd.crosstab(participants["dataset_id"], participants["sex"].fillna("Missing")).reindex(datasets, fill_value=0)
    sex.plot.bar(stacked=True, ax=axes[1], title="Participants by sex")
    axes[0].set_ylabel("Participants"); axes[1].set_ylabel("Participants")
    axes[2].axis("off")
    axes[0].legend(title="Population", frameon=False); axes[1].legend(title="Sex", frameon=False)
    fig.suptitle("Dataset composition")
    fig.savefig(output / "composition_population_sex.png", dpi=160)
    plt.close(fig)

    variables = [column for column in participants if column in {"age_years", "updrs", "moca", "mmse"} or column.startswith("metadata__")]
    if not variables:
        return
    ncols = min(3, len(variables)); nrows = int(np.ceil(len(variables) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False, constrained_layout=True)
    for index, variable in enumerate(variables):
        axis = axes.flat[index]
        values = participants[["dataset_id", "population", variable]].copy()
        values[variable] = pd.to_numeric(values[variable], errors="coerce")
        values = values.dropna()
        for position, dataset in enumerate(datasets):
            for offset, population_name in enumerate(("Control", "PD")):
                data = values.loc[values["dataset_id"].eq(dataset) & values["population"].eq(population_name), variable].to_numpy()
                if len(data):
                    axis.scatter(np.full(len(data), position + (offset - .5) * .22), data, s=12, alpha=.55)
        axis.set_title(variable.removeprefix("metadata__").replace("_", " "))
        axis.set_xticks(range(len(datasets)), datasets, rotation=35, ha="right")
        axis.grid(axis="y", alpha=.2)
    for axis in axes.flat[len(variables):]: axis.axis("off")
    fig.suptitle("Clinical and demographic distributions by dataset and population")
    fig.savefig(output / "composition_continuous_variables.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/global_pipeline.json")
    parser.add_argument("--datasets", nargs="+", help="Dataset ids; defaults to all enabled datasets")
    parser.add_argument("--output", help="Output directory; defaults to outputs/global/composition")
    args = parser.parse_args()
    config = load_global_config(args.config)
    dataset_ids = args.datasets or [dataset.dataset_id for dataset in config.enabled_datasets]
    output = Path(args.output) if args.output else config.output_root / "composition"
    recordings = _read_canonical(config, dataset_ids)
    extras = _metadata_extras(config, dataset_ids)
    participants = _participant_table(recordings, extras)
    continuous, categorical = _summary_tables(participants)
    tests = _comparability_tests(participants)
    output.mkdir(parents=True, exist_ok=True)
    participants.to_csv(output / "participant_composition.csv.gz", index=False, compression="gzip")
    continuous.to_csv(output / "continuous_summary.csv.gz", index=False, compression="gzip")
    categorical.to_csv(output / "categorical_summary.csv.gz", index=False, compression="gzip")
    tests.to_csv(output / "comparability_tests.csv.gz", index=False, compression="gzip")
    pooling = tests.loc[
        (tests["test"].eq("Kruskal-Wallis") | tests["test"].eq("Chi-square"))
        & tests["scope"].eq("all")
    ].copy()
    if not pooling.empty:
        pooling["pooling_flag"] = np.where(
            pooling["significant_fdr"],
            "adjust_for_dataset_or_stratify",
            "no_detectable_overall_difference",
        )
        pooling.to_csv(output / "pooling_assessment.csv.gz", index=False, compression="gzip")
    _plot_composition(participants, output / "figures")
    significant = tests.loc[tests["significant_fdr"]] if not tests.empty else tests
    report = output / "pooling_assessment.txt"
    report.write_text(
        "Dataset composition assessment\n\n"
        f"Datasets: {', '.join(dataset_ids)}\n"
        f"Participants: {len(participants)}\n"
        f"FDR-significant composition tests: {len(significant)}\n"
        "See pooling_assessment.csv.gz for overall variable-level flags.\n\n"
        "Interpretation: significant age/clinical differences or unequal population/sex composition "
        "do not automatically prohibit pooling, but pooled EEG models should include dataset as a "
        "covariate (and account for age/sex). If group-specific distributions do not overlap, prefer "
        "dataset-stratified or meta-analytic results rather than an unadjusted pooled comparison.\n",
        encoding="utf-8",
    )
    print(f"Wrote composition assessment to {output}")


if __name__ == "__main__":
    main()
