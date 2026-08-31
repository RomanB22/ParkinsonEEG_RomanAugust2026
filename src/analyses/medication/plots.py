"""Focused condition and MMSE figures for ds002778."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_ORDER = ("HC", "PD_OFF", "PD_ON")
MMSE_MODEL_ORDER = ("HC", "PD_OFF", "PD_ON", "PD_ON_minus_PD_OFF")
MMSE_MODEL_LABELS = {
    "HC": "Healthy control",
    "PD_OFF": "PD OFF",
    "PD_ON": "PD ON",
    "PD_ON_minus_PD_OFF": "PD ON − OFF",
}
WITHIN_BOUT_THETA_METRICS = (
    ("entropy", "Permutation entropy (H)"),
    ("complexity", "Statistical complexity (C)"),
    ("fisher_information", "Fisher information (F)"),
)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def plot_condition_features(
    features: pd.DataFrame,
    recordings: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    """Plot prespecified subject-level features with paired PD trajectories."""
    settings = config["plots"]
    requested = [str(value) for value in settings["feature_ids"]]
    primary = features.loc[features["duration_variant"].eq("all_retained")]
    attached = primary.merge(
        recordings[["recording_id", "participant_id", "condition", "mmse"]],
        on="recording_id",
        validate="many_to_one",
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = settings["condition_colors"]
    rng = np.random.default_rng(int(config["statistics"]["random_seed"]))
    paths: list[Path] = []
    for feature_id in requested:
        table = attached.loc[attached["feature_id"].eq(feature_id)].dropna(
            subset=["value"]
        )
        if table.empty:
            continue
        figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        pd_values = table.loc[table["condition"].isin(["PD_OFF", "PD_ON"])]
        paired = pd_values.pivot(
            index="participant_id", columns="condition", values="value"
        ).dropna()
        for _, row in paired.iterrows():
            axis.plot([1, 2], [row["PD_OFF"], row["PD_ON"]], color="#999999", alpha=0.35, linewidth=0.8, zorder=1)
        for index, condition in enumerate(CONDITION_ORDER):
            values = table.loc[table["condition"].eq(condition), "value"].to_numpy(float)
            jitter = rng.uniform(-0.09, 0.09, size=len(values))
            axis.scatter(
                index + jitter,
                values,
                color=colors[condition],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
                s=38,
                zorder=3,
            )
            if len(values):
                axis.plot(
                    [index - 0.18, index + 0.18],
                    [np.median(values)] * 2,
                    color="black",
                    linewidth=2.2,
                    zorder=4,
                )
        axis.set_xticks(range(3), ["Healthy control", "PD OFF", "PD ON"])
        axis.set_ylabel(str(table.iloc[0]["metric"]).replace("_", " "))
        axis.set_title(feature_id.replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
        path = output_dir / f"condition_{_safe_name(feature_id)}.png"
        figure.savefig(path, dpi=int(settings["dpi"]))
        plt.close(figure)
        paths.append(path)
    return paths


def plot_mmse_features(
    features: pd.DataFrame,
    recordings: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    """Plot MMSE against PD OFF, PD ON, and paired medication change."""
    settings = config["plots"]
    requested = [str(value) for value in settings["feature_ids"]]
    primary = features.loc[features["duration_variant"].eq("all_retained")]
    attached = primary.merge(
        recordings[["recording_id", "participant_id", "condition", "mmse"]],
        on="recording_id",
        validate="many_to_one",
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    colors = settings["condition_colors"]
    for feature_id in requested:
        table = attached.loc[attached["feature_id"].eq(feature_id)].dropna(
            subset=["value", "mmse"]
        )
        pd_table = table.loc[table["condition"].isin(["PD_OFF", "PD_ON"])]
        if pd_table.empty:
            continue
        pivot = pd_table.pivot(
            index="participant_id", columns="condition", values="value"
        ).dropna()
        mmse = (
            pd_table[["participant_id", "mmse"]]
            .drop_duplicates("participant_id")
            .set_index("participant_id")
        )
        delta = pivot.join(mmse).dropna()
        delta["PD_DELTA"] = delta["PD_ON"] - delta["PD_OFF"]
        panels = (
            ("PD_OFF", "PD OFF", colors["PD_OFF"]),
            ("PD_ON", "PD ON", colors["PD_ON"]),
            ("PD_DELTA", "PD ON − OFF", "#6A3D9A"),
        )
        figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharex=True, constrained_layout=True)
        for axis, (column, label, color) in zip(axes, panels):
            x = delta["mmse"].to_numpy(float)
            y = delta[column].to_numpy(float)
            axis.scatter(x, y, color=color, edgecolor="white", linewidth=0.5, s=42)
            if len(np.unique(x)) > 1:
                coefficients = np.polyfit(x, y, 1)
                grid = np.linspace(x.min(), x.max(), 100)
                axis.plot(grid, np.polyval(coefficients, grid), color=color, linewidth=1.5)
            axis.set_title(label)
            axis.set_xlabel("MMSE")
            axis.grid(alpha=0.2)
        axes[0].set_ylabel(str(table.iloc[0]["metric"]).replace("_", " "))
        figure.suptitle(feature_id.replace("_", " "))
        path = output_dir / f"mmse_{_safe_name(feature_id)}.png"
        figure.savefig(path, dpi=int(settings["dpi"]))
        plt.close(figure)
        paths.append(path)
    return paths


def plot_within_bout_theta_mmse(
    features: pd.DataFrame,
    recordings: pd.DataFrame,
    mmse_statistics: pd.DataFrame,
    path: str | Path,
    config: dict[str, Any],
) -> Path | None:
    """Plot the prespecified theta within-bout H/C/F associations with MMSE.

    Raw participant observations are shown for interpretability. Panel
    annotations report the primary age/sex-adjusted partial Spearman inference;
    PD medication sessions are kept separate and their paired change is shown
    in the fourth column.
    """
    selected_features = features.loc[
        features["duration_variant"].eq("all_retained")
        & features["family"].eq("within_bout_ordinal")
        & features["band"].eq("theta")
        & features["metric"].isin(
            tuple(metric for metric, _ in WITHIN_BOUT_THETA_METRICS)
        )
    ].merge(
        recordings[
            [
                "recording_id",
                "participant_id",
                "condition",
                "mmse",
            ]
        ],
        on="recording_id",
        validate="many_to_one",
    )
    selected_statistics = mmse_statistics.loc[
        mmse_statistics["duration_variant"].eq("all_retained")
        & mmse_statistics["sensitivity_cohort"].eq("all_participants")
        & mmse_statistics["family"].eq("within_bout_ordinal")
        & mmse_statistics["band"].eq("theta")
    ]
    if selected_features.empty or selected_statistics.empty:
        return None

    colors = {
        "HC": config["plots"]["condition_colors"]["HC"],
        "PD_OFF": config["plots"]["condition_colors"]["PD_OFF"],
        "PD_ON": config["plots"]["condition_colors"]["PD_ON"],
        "PD_ON_minus_PD_OFF": "#6A3D9A",
    }
    rng = np.random.default_rng(int(config["statistics"]["random_seed"]))
    figure, axes = plt.subplots(
        len(WITHIN_BOUT_THETA_METRICS),
        len(MMSE_MODEL_ORDER),
        figsize=(14.2, 10.2),
        sharex=True,
        squeeze=False,
    )
    for row, (metric, metric_label) in enumerate(WITHIN_BOUT_THETA_METRICS):
        metric_table = selected_features.loc[
            selected_features["metric"].eq(metric)
        ]
        feature_ids = metric_table["feature_id"].drop_duplicates()
        if len(feature_ids) != 1:
            raise ValueError(
                f"Expected one theta within-bout feature for {metric}; found {len(feature_ids)}"
            )
        feature_id = str(feature_ids.iloc[0])
        pd_table = metric_table.loc[
            metric_table["condition"].isin(["PD_OFF", "PD_ON"])
        ]
        pd_values = pd_table.pivot(
            index="participant_id", columns="condition", values="value"
        )
        pd_mmse = (
            pd_table[["participant_id", "mmse"]]
            .drop_duplicates("participant_id")
            .set_index("participant_id")
        )
        paired_change = pd_values.join(pd_mmse, how="inner").dropna(
            subset=["PD_OFF", "PD_ON", "mmse"]
        )
        paired_change["value"] = paired_change["PD_ON"] - paired_change["PD_OFF"]

        model_tables = {
            condition: metric_table.loc[
                metric_table["condition"].eq(condition), ["mmse", "value"]
            ].dropna()
            for condition in ("HC", "PD_OFF", "PD_ON")
        }
        model_tables["PD_ON_minus_PD_OFF"] = paired_change[["mmse", "value"]]

        for column, model in enumerate(MMSE_MODEL_ORDER):
            axis = axes[row, column]
            observations = model_tables[model]
            x = observations["mmse"].to_numpy(float)
            y = observations["value"].to_numpy(float)
            jittered_x = x + rng.uniform(-0.055, 0.055, len(x))
            axis.scatter(
                jittered_x,
                y,
                color=colors[model],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.88,
                s=42,
                zorder=3,
            )
            if len(np.unique(x)) > 1:
                coefficients = np.polyfit(x, y, 1)
                grid = np.linspace(float(x.min()), float(x.max()), 100)
                axis.plot(
                    grid,
                    np.polyval(coefficients, grid),
                    color=colors[model],
                    linestyle="--",
                    linewidth=1.4,
                    zorder=2,
                )
            statistic = selected_statistics.loc[
                selected_statistics["feature_id"].eq(feature_id)
                & selected_statistics["mmse_model"].eq(model)
            ]
            if len(statistic) != 1:
                raise ValueError(
                    f"Expected one primary MMSE result for {feature_id}/{model}; "
                    f"found {len(statistic)}"
                )
            result = statistic.iloc[0]
            marker = " *" if bool(result["fdr_reject"]) else ""
            axis.set_title(
                f"{MMSE_MODEL_LABELS[model]} (n={int(result['n_participants'])})\n"
                f"partial ρ={float(result['statistic']):.2f}, "
                f"p={float(result['primary_p_value']):.3f}, "
                f"q={float(result['p_fdr_bh']):.3f}{marker}",
                fontsize=9.5,
            )
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_ylabel(metric_label)
            if row == len(WITHIN_BOUT_THETA_METRICS) - 1:
                axis.set_xlabel("MMSE score")

    mmse_values = selected_features["mmse"].dropna().to_numpy(float)
    if len(mmse_values):
        ticks = np.arange(int(mmse_values.min()), int(mmse_values.max()) + 1)
        for axis in axes.flat:
            axis.set_xticks(ticks)
    figure.suptitle(
        "Theta (4–8 Hz) ordinal metrics within detected bouts versus MMSE\n"
        "Points and dashed trends show raw values; ρ, p, and BH-FDR q use age/sex-adjusted ranks",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=int(config["plots"]["dpi"]), bbox_inches="tight")
    plt.close(figure)
    return path
