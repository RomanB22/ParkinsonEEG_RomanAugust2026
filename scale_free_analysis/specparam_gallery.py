"""Generate one flat, browsable all-electrode specparam figure per subject."""

from __future__ import annotations

import html
import logging
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CURVE_NAMES = (
    "observed_psd_uv2_hz",
    "modeled_psd_uv2_hz",
    "aperiodic_psd_uv2_hz",
    "periodic_psd_uv2_hz",
)
OPTIONAL_CURVE_NAMES = (
    "fixed_aperiodic_psd_uv2_hz",
    "knee_aperiodic_psd_uv2_hz",
)

def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not cleaned:
        raise ValueError(f"Cannot create a safe filename for {value!r}")
    return cleaned


def _plot_subject_overview(
    subject_rows: list[dict[str, Any]],
    electrodes: list[str],
    electrode_indices: dict[str, int],
    frequencies: np.ndarray,
    arrays: dict[str, np.ndarray],
    output_path: Path,
    dpi: int,
) -> None:
    """Plot all electrode-level spectral fits in one subject overview."""
    rows = {str(row["electrode"]): row for row in subject_rows}
    ordered_electrodes = [electrode for electrode in electrodes if electrode in rows]
    if not ordered_electrodes:
        raise ValueError("Cannot render a subject overview without electrodes")

    n_columns = 6
    n_rows = int(np.ceil(len(ordered_electrodes) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(21, 2.75 * n_rows + 1.5),
        squeeze=False,
    )
    qc_passes = 0
    for axis, electrode in zip(axes.flat, ordered_electrodes):
        row = rows[electrode]
        index = electrode_indices[electrode]
        axis.semilogy(
            frequencies,
            arrays["observed_psd_uv2_hz"][index],
            color="0.15",
            linewidth=0.75,
        )
        axis.semilogy(
            frequencies,
            arrays["modeled_psd_uv2_hz"][index],
            color="#0072B2",
            linewidth=1.25,
        )
        axis.semilogy(
            frequencies,
            arrays["aperiodic_psd_uv2_hz"][index],
            color="#D55E00",
            linewidth=1.35,
        )
        if "fixed_aperiodic_psd_uv2_hz" in arrays:
            axis.semilogy(
                frequencies,
                arrays["fixed_aperiodic_psd_uv2_hz"][index],
                color="#666666",
                linestyle="--",
                linewidth=0.75,
            )
        if "knee_aperiodic_psd_uv2_hz" in arrays and np.isfinite(
            arrays["knee_aperiodic_psd_uv2_hz"][index]
        ).all():
            axis.semilogy(
                frequencies,
                arrays["knee_aperiodic_psd_uv2_hz"][index],
                color="#CC79A7",
                linestyle=":",
                linewidth=0.85,
            )
        axis.fill_between(
            frequencies,
            arrays["aperiodic_psd_uv2_hz"][index],
            arrays["modeled_psd_uv2_hz"][index],
            color="#009E73",
            alpha=0.12,
        )
        qc_value = row.get("specparam_fit_qc_pass")
        qc_known = isinstance(qc_value, (bool, np.bool_))
        qc_pass = bool(qc_value) if qc_known else False
        qc_passes += int(qc_known and qc_pass)
        title_color = "#007A3D" if qc_pass else ("#B22222" if qc_known else "0.2")
        axis.set_title(
            f"{electrode} [{row['specparam_aperiodic_mode']}] | "
            f"R²={float(row['specparam_r_squared']):.2f}  "
            f"E={float(row['aperiodic_exponent']):.2f}",
            fontsize=8,
            color=title_color,
            fontweight="bold" if qc_known and not qc_pass else "normal",
        )
        axis.set_xlim(float(frequencies[0]), float(frequencies[-1]))
        axis.grid(alpha=0.16, linewidth=0.5)
        axis.tick_params(labelsize=6)
    for axis in axes.flat[len(ordered_electrodes) :]:
        axis.set_axis_off()

    for axis in axes[-1, :]:
        if axis.axison:
            axis.set_xlabel("Frequency (Hz)", fontsize=7)
    for axis in axes[:, 0]:
        if axis.axison:
            axis.set_ylabel("PSD (µV²/Hz)", fontsize=7)

    first = rows[ordered_electrodes[0]]
    group = str(first.get("group", ""))
    subject_id = str(first["subject_id"])
    qc_text = (
        f"{qc_passes}/{len(ordered_electrodes)} fits pass QC"
        if all(isinstance(rows[name].get("specparam_fit_qc_pass"), (bool, np.bool_)) for name in ordered_electrodes)
        else "fit QC not available"
    )
    fig.suptitle(
        f"{subject_id} — {group}: all-electrode spectral fits\n{qc_text}; "
        "blue is selected full fit; orange is selected background; dashed fixed and dotted knee",
        fontsize=14,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color="0.15", linewidth=1.0, label="Observed PSD"),
            Line2D([0], [0], color="#0072B2", linewidth=1.6, label="Full model"),
            Line2D([0], [0], color="#D55E00", linewidth=1.4, label="Aperiodic component"),
            Line2D([0], [0], color="#666666", linestyle="--", label="Fixed background"),
            Line2D([0], [0], color="#CC79A7", linestyle=":", label="Knee background"),
            Patch(facecolor="#009E73", alpha=0.2, label="Periodic contribution"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=6,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.92, bottom=0.04, left=0.045, right=0.99, hspace=0.48, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _render_subject(
    spectra_path: str,
    subject_rows: list[dict[str, Any]],
    gallery_root: str,
    dpi: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    """Worker that renders exactly one all-electrode figure for one subject."""
    spectra_path_object = Path(spectra_path)
    if not spectra_path_object.exists():
        raise FileNotFoundError(f"Missing subject spectra: {spectra_path_object}")
    with np.load(spectra_path_object, allow_pickle=False) as spectra:
        required = {"electrodes", "frequencies_hz", *CURVE_NAMES}
        missing = sorted(required - set(spectra.files))
        if missing:
            raise ValueError(f"{spectra_path_object} is missing arrays: {missing}")
        electrodes = spectra["electrodes"].astype(str).tolist()
        if len(electrodes) != len(set(electrodes)):
            raise ValueError(f"{spectra_path_object} contains duplicate electrodes")
        electrode_indices = {electrode: index for index, electrode in enumerate(electrodes)}
        frequencies = spectra["frequencies_hz"].copy()
        arrays = {name: spectra[name].copy() for name in CURVE_NAMES}
        arrays.update(
            {
                name: spectra[name].copy()
                for name in OPTIONAL_CURVE_NAMES
                if name in spectra.files
            }
        )
    for name, values in arrays.items():
        if values.shape != (len(electrodes), len(frequencies)):
            raise ValueError(
                f"{spectra_path_object}/{name} has shape {values.shape}; "
                f"expected {(len(electrodes), len(frequencies))}"
            )

    first_row = subject_rows[0]
    subject_id = str(first_row["subject_id"])
    group = str(first_row["group"])
    filename = (
        f"{_safe_name(subject_id)}_{_safe_name(group)}_all_electrodes.png"
    )
    overview_path = Path(gallery_root) / filename
    rendered = overwrite or not overview_path.exists()
    if rendered:
        _plot_subject_overview(
            subject_rows,
            electrodes,
            electrode_indices,
            frequencies,
            arrays,
            overview_path,
            int(dpi),
        )
    qc_values = [
        row.get("specparam_fit_qc_pass")
        for row in subject_rows
        if isinstance(row.get("specparam_fit_qc_pass"), (bool, np.bool_))
    ]
    r_squared = np.asarray(
        [float(row["specparam_r_squared"]) for row in subject_rows], dtype=float
    )
    exponents = np.asarray(
        [float(row["aperiodic_exponent"]) for row in subject_rows], dtype=float
    )
    errors = np.asarray(
        [float(row["specparam_error_mae"]) for row in subject_rows], dtype=float
    )
    selected_modes = [str(row["specparam_aperiodic_mode"]) for row in subject_rows]
    return [
        {
            "subject_id": subject_id,
            "group": group,
            "figure_path": filename,
            "subject_figure_path": filename,
            "rendered_this_run": bool(rendered),
            "subject_overview_rendered_this_run": bool(rendered),
            "n_electrodes": int(len(subject_rows)),
            "n_qc_pass": int(sum(bool(value) for value in qc_values)),
            "qc_pass_fraction": (
                float(np.mean(qc_values)) if len(qc_values) else np.nan
            ),
            "median_aperiodic_exponent": float(np.median(exponents)),
            "median_specparam_r_squared": float(np.median(r_squared)),
            "median_specparam_error_mae": float(np.median(errors)),
            "fraction_knee_selected": float(
                np.mean(np.asarray(selected_modes) == "knee")
            ),
        }
    ]


def _write_html_indexes(index: pd.DataFrame, gallery_root: Path) -> None:
    style = """
body { font-family: sans-serif; margin: 2rem; color: #222; }
a { color: #0067a5; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 1rem; }
.card { border: 1px solid #ddd; border-radius: 6px; padding: .6rem; }
.card img { width: 100%; height: auto; display: block; }
.meta { font-size: .85rem; color: #444; margin-top: .35rem; }
.pass { color: #007A3D; font-weight: 700; }
.fail { color: #B22222; font-weight: 700; }
""".strip()
    root_sections = []
    for group, group_table in index.groupby("group", sort=False):
        cards = []
        for _, row in group_table.sort_values("subject_id").iterrows():
            figure = str(row["figure_path"])
            qc_fraction = row.get("qc_pass_fraction")
            qc_text = (
                f'{int(row["n_qc_pass"])}/{int(row["n_electrodes"])} fits pass QC'
                if np.isfinite(qc_fraction)
                else "fit QC not available"
            )
            cards.append(
                '<div class="card">'
                f'<a href="{html.escape(figure)}"><img loading="lazy" '
                f'src="{html.escape(figure)}" alt="{html.escape(str(row["subject_id"]))}"></a>'
                f'<div class="meta"><strong>{html.escape(str(row["subject_id"]))}</strong> — '
                f'{qc_text}; median exponent={row["median_aperiodic_exponent"]:.3f}, '
                f'median R²={row["median_specparam_r_squared"]:.3f}; '
                f'knee selected={100 * row["fraction_knee_selected"]:.1f}%</div></div>'
            )
        root_sections.append(
            f"<h2>{html.escape(str(group))}</h2>"
            f'<div class="grid">{"".join(cards)}</div>'
        )
    root_document = (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Specparam gallery</title>"
        f"<style>{style}</style></head><body>"
        "<h1>Subject specparam decompositions</h1>"
        f"<p>{len(index)} subjects; one all-electrode figure per subject. "
        "Every PNG is stored directly in this folder.</p>"
        f"{''.join(root_sections)}</body></html>"
    )
    (gallery_root / "index.html").write_text(root_document, encoding="utf-8")


def generate_specparam_gallery(
    spectra_dir: str | Path,
    aperiodic_metrics: pd.DataFrame,
    gallery_root: str | Path,
    *,
    dpi: int = 100,
    workers: int = 1,
    overwrite: bool = False,
    overwrite_subject_overviews: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Render one flat all-electrode figure per subject and write one HTML index."""
    spectra_dir = Path(spectra_dir)
    gallery_root = Path(gallery_root)
    gallery_root.mkdir(parents=True, exist_ok=True)
    required_columns = {
        "subject_id",
        "group",
        "electrode",
        "aperiodic_offset",
        "aperiodic_exponent",
        "specparam_r_squared",
        "specparam_error_mae",
        "specparam_aperiodic_mode",
    }
    missing = sorted(required_columns - set(aperiodic_metrics.columns))
    if missing:
        raise ValueError(f"Aperiodic metrics are missing columns: {missing}")
    if aperiodic_metrics.duplicated(["subject_id", "electrode"]).any():
        raise ValueError("Aperiodic metrics contain duplicate subject/electrode rows")
    if int(workers) < 1:
        raise ValueError("workers must be at least one")

    tasks = []
    for subject_id, selected in aperiodic_metrics.groupby("subject_id", sort=False):
        spectra_path = spectra_dir / f"{subject_id}_specparam_spectra.npz"
        tasks.append(
            (
                str(spectra_path),
                selected.to_dict(orient="records"),
                str(gallery_root),
                int(dpi),
                bool(overwrite or overwrite_subject_overviews),
            )
        )
    rows: list[dict[str, Any]] = []
    if int(workers) == 1:
        for task_index, task in enumerate(tasks, start=1):
            rows.extend(_render_subject(*task))
            if logger is not None:
                logger.info(
                    "Specparam gallery [%d/%d] | %s",
                    task_index,
                    len(tasks),
                    Path(task[0]).stem.removesuffix("_specparam_spectra"),
                )
    else:
        try:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures = {executor.submit(_render_subject, *task): task for task in tasks}
                for completed_index, future in enumerate(
                    as_completed(futures), start=1
                ):
                    task = futures[future]
                    rows.extend(future.result())
                    if logger is not None:
                        logger.info(
                            "Specparam gallery [%d/%d] | %s",
                            completed_index,
                            len(tasks),
                            Path(task[0]).stem.removesuffix(
                                "_specparam_spectra"
                            ),
                        )
        except PermissionError:
            if logger is not None:
                logger.warning(
                    "Process workers are unavailable; rendering gallery serially"
                )
            rows.clear()
            for task_index, task in enumerate(tasks, start=1):
                rows.extend(_render_subject(*task))
                if logger is not None:
                    logger.info(
                        "Specparam gallery [%d/%d] | %s",
                        task_index,
                        len(tasks),
                        Path(task[0]).stem.removesuffix("_specparam_spectra"),
                    )
    index = pd.DataFrame.from_records(rows).sort_values(
        ["group", "subject_id"]
    ).reset_index(drop=True)
    index.to_csv(
        gallery_root / "figure_index.csv", index=False, float_format="%.17g"
    )
    _write_html_indexes(index, gallery_root)
    # Remove only the legacy group directories that this generator created.
    # The new layout deliberately contains no per-group or per-subject folders.
    for group in aperiodic_metrics["group"].astype(str).unique():
        legacy_directory = gallery_root / _safe_name(group)
        if legacy_directory.is_dir():
            shutil.rmtree(legacy_directory)
    expected_figures = set(index["figure_path"].astype(str))
    for stale_figure in gallery_root.glob("sub-*_all_electrodes.png"):
        if stale_figure.name not in expected_figures:
            stale_figure.unlink()
    return index
