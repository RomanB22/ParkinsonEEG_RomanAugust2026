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
