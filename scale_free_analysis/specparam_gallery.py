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

from .plots import plot_spectral_example


CURVE_NAMES = (
    "observed_psd_uv2_hz",
    "modeled_psd_uv2_hz",
    "aperiodic_psd_uv2_hz",
    "periodic_psd_uv2_hz",
)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not cleaned:
        raise ValueError(f"Cannot create a safe filename for {value!r}")
    return cleaned


def _render_subject(
    spectra_path: str,
    subject_rows: list[dict[str, Any]],
    gallery_root: str,
    dpi: int,
    overwrite: bool,
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
                "rendered_this_run": bool(rendered),
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
            subject_links.append(
                f'<li><a href="{html.escape(str(group))}/{html.escape(subject_id)}/index.html">'
                f"{html.escape(subject_id)}</a></li>"
            )
        root_sections.append(
            f"<h2>{html.escape(str(group))}</h2><ul>{''.join(subject_links)}</ul>"
        )
    root_document = (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Specparam gallery</title>"
        f"<style>{style}</style></head><body>"
        "<h1>Subject/electrode specparam decompositions</h1>"
        f"<p>{index['subject_id'].nunique()} subjects, {len(index)} electrode figures.</p>"
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
