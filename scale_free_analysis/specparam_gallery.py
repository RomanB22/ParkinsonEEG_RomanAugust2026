"""Generate browsable per-subject/electrode specparam decomposition figures."""

from __future__ import annotations

import html
import logging
import re
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

from .plots import plot_spectral_example


CURVE_NAMES = (
    "observed_psd_uv2_hz",
    "modeled_psd_uv2_hz",
    "aperiodic_psd_uv2_hz",
    "periodic_psd_uv2_hz",
)

SUBJECT_OVERVIEW_FILENAME = "all_electrodes.png"


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
            linewidth=1.05,
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
            f"{electrode}  |  R²={float(row['specparam_r_squared']):.2f}  "
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
        "judge fit by blue vs black; orange is background only; red labels fail QC",
        fontsize=14,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color="0.15", linewidth=1.0, label="Observed PSD"),
            Line2D([0], [0], color="#0072B2", linewidth=1.6, label="Full model"),
            Line2D([0], [0], color="#D55E00", linewidth=1.4, label="Aperiodic component"),
            Patch(facecolor="#009E73", alpha=0.2, label="Periodic contribution"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
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
    overwrite_subject_overview: bool,
) -> list[dict[str, Any]]:
    """Worker that renders every electrode for one subject."""
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
    for name, values in arrays.items():
        if values.shape != (len(electrodes), len(frequencies)):
            raise ValueError(
                f"{spectra_path_object}/{name} has shape {values.shape}; "
                f"expected {(len(electrodes), len(frequencies))}"
            )

    first_row = subject_rows[0]
    subject_directory = (
        Path(gallery_root)
        / _safe_name(str(first_row["group"]))
        / str(first_row["subject_id"])
    )
    overview_path = subject_directory / SUBJECT_OVERVIEW_FILENAME
    overview_rendered = overwrite_subject_overview or not overview_path.exists()
    if overview_rendered:
        _plot_subject_overview(
            subject_rows,
            electrodes,
            electrode_indices,
            frequencies,
            arrays,
            overview_path,
            int(dpi),
        )

    output_rows = []
    used_filenames: set[str] = set()
    for metric_row in subject_rows:
        electrode = str(metric_row["electrode"])
        if electrode not in electrode_indices:
            raise ValueError(f"{spectra_path_object}: missing electrode {electrode}")
        electrode_filename = _safe_name(electrode) + ".png"
        if electrode_filename in used_filenames:
            raise ValueError(f"Electrode filename collision for {electrode}")
        used_filenames.add(electrode_filename)
        group = str(metric_row["group"])
        subject_id = str(metric_row["subject_id"])
        relative_path = Path(_safe_name(group)) / subject_id / electrode_filename
        output_path = Path(gallery_root) / relative_path
        rendered = overwrite or not output_path.exists()
        if rendered:
            index = electrode_indices[electrode]
            example = {
                "subject_id": subject_id,
                "group": group,
                "electrode": electrode,
                "frequencies_hz": frequencies,
                "aperiodic_exponent": float(metric_row["aperiodic_exponent"]),
                "specparam_r_squared": float(metric_row["specparam_r_squared"]),
                "specparam_error_mae": float(metric_row["specparam_error_mae"]),
                "specparam_fit_qc_pass": metric_row.get("specparam_fit_qc_pass"),
                "specparam_fit_qc_reasons": metric_row.get(
                    "specparam_fit_qc_reasons", "not_assessed"
                ),
                **{name: values[index] for name, values in arrays.items()},
            }
            plot_spectral_example(example, output_path, int(dpi))
        output_rows.append(
            {
                "subject_id": subject_id,
                "group": group,
                "electrode": electrode,
                "figure_path": relative_path.as_posix(),
                "subject_figure_path": (
                    Path(_safe_name(group)) / subject_id / SUBJECT_OVERVIEW_FILENAME
                ).as_posix(),
                "rendered_this_run": bool(rendered),
                "subject_overview_rendered_this_run": bool(overview_rendered),
                "aperiodic_offset": float(metric_row["aperiodic_offset"]),
                "aperiodic_exponent": float(metric_row["aperiodic_exponent"]),
                "specparam_r_squared": float(metric_row["specparam_r_squared"]),
                "specparam_error_mae": float(metric_row["specparam_error_mae"]),
                "specparam_fit_qc_pass": metric_row.get("specparam_fit_qc_pass"),
                "specparam_fit_qc_reasons": str(
                    metric_row.get("specparam_fit_qc_reasons", "not_assessed")
                ),
            }
        )
    return output_rows


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
        subject_links = []
        for subject_id in group_table["subject_id"].drop_duplicates():
            selected = group_table.loc[group_table["subject_id"].eq(subject_id)]
            overview = str(selected["subject_figure_path"].iloc[0])
            subject_links.append(
                f'<li><strong>{html.escape(subject_id)}</strong>: '
                f'<a href="{html.escape(overview)}">all-electrode figure</a> · '
                f'<a href="{html.escape(str(group))}/{html.escape(subject_id)}/index.html">'
                "individual fits and residuals</a></li>"
            )
        root_sections.append(
            f"<h2>{html.escape(str(group))}</h2><ul>{''.join(subject_links)}</ul>"
        )
    root_document = (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Specparam gallery</title>"
        f"<style>{style}</style></head><body>"
        "<h1>Subject specparam decompositions</h1>"
        f"<p>{index['subject_id'].nunique()} all-electrode subject figures; "
        f"{len(index)} detailed electrode figures.</p>"
        f"{''.join(root_sections)}</body></html>"
    )
    (gallery_root / "index.html").write_text(root_document, encoding="utf-8")

    for (group, subject_id), selected in index.groupby(
        ["group", "subject_id"], sort=False
    ):
        cards = []
        for _, row in selected.iterrows():
            filename = Path(row["figure_path"]).name
            qc_value = row.get("specparam_fit_qc_pass")
            if isinstance(qc_value, (bool, np.bool_)):
                qc_class = "pass" if bool(qc_value) else "fail"
                qc_text = "QC PASS" if bool(qc_value) else "QC FAIL"
                qc_html = f' — <span class="{qc_class}">{qc_text}</span>'
            else:
                qc_html = ""
            cards.append(
                '<div class="card">'
                f'<a href="{html.escape(filename)}"><img loading="lazy" '
                f'src="{html.escape(filename)}" alt="{html.escape(str(row["electrode"]))}"></a>'
                f'<div class="meta"><strong>{html.escape(str(row["electrode"]))}</strong> — '
                f'exponent={row["aperiodic_exponent"]:.3f}, R²={row["specparam_r_squared"]:.3f}, '
                f'MAE={row["specparam_error_mae"]:.3f}{qc_html}'
                "</div></div>"
            )
        subject_directory = gallery_root / _safe_name(str(group)) / str(subject_id)
        subject_document = (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            f"<title>{html.escape(str(subject_id))} specparam</title><style>{style}</style>"
            "</head><body>"
            '<p><a href="../../index.html">← All subjects</a></p>'
            f"<h1>{html.escape(str(subject_id))} — {html.escape(str(group))}</h1>"
            f'<p><a href="{SUBJECT_OVERVIEW_FILENAME}">Open the all-electrode figure at full size</a></p>'
            f'<a href="{SUBJECT_OVERVIEW_FILENAME}"><img src="{SUBJECT_OVERVIEW_FILENAME}" '
            'alt="All-electrode spectral fits" style="width:100%;height:auto"></a>'
            "<h2>Individual fits and signed residuals</h2>"
            f'<div class="grid">{"".join(cards)}</div></body></html>'
        )
        (subject_directory / "index.html").write_text(
            subject_document, encoding="utf-8"
        )


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
    """Render every subject/electrode fit and write browsable HTML indexes."""
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
                bool(overwrite),
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
        ["group", "subject_id", "electrode"]
    ).reset_index(drop=True)
    index.to_csv(
        gallery_root / "figure_index.csv", index=False, float_format="%.17g"
    )
    _write_html_indexes(index, gallery_root)
    return index
