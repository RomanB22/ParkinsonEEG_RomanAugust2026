"""Top-to-bottom subject preprocessing that mirrors Prompt.md."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np

from .artifacts import annotate_large_artifacts, create_and_reject_epochs
from .channels import detect_bad_channels
from .config import (
    is_ica_review_confirmed,
    preprocessing_signature,
    subject_manual_ica,
    write_ica_review_proposal,
)
from .dataset import (
    load_subject,
    participant_metadata,
    recording_id_from_path,
    session_id_from_path,
    subject_id_from_path,
)
from .ica import (
    apply_ica_exclusions,
    fit_ica,
    make_ica_copy,
    proposed_ica_exclusions,
    score_ica_components,
)
from . import qc


@dataclass
class SubjectResult:
    subject_id: str
    qc_row: dict[str, Any] | None
    cleaned_raw_path: Path | None
    epochs_path: Path | None
    ica_path: Path
    review_only: bool


def _list_text(values) -> str:
    return ";".join(str(value) for value in values)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _subject_logger(
    subject_id: str, log_dir: Path, *, console_logging: bool = True
) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"preprocessing.{subject_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_dir / f"{subject_id}.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if console_logging:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def filter_eeg(raw, config: dict[str, Any], logger: logging.Logger):
    """Return a new, notched final-band signal without changing sample rate."""
    filtered = raw.copy()
    filtered.filter(
        l_freq=float(config["l_freq"]),
        h_freq=float(config["h_freq"]),
        method=str(config["method"]),
        phase=str(config["phase"]),
        picks="eeg",
        skip_by_annotation=("edge", "bad_acq_skip"),
        verbose="ERROR",
    )
    notch_applied = False
    notch_reason = str(config.get("reason", "not requested"))
    if bool(config.get("notch_enabled", False)):
        notch_frequency = float(config["notch_freq_hz"])
        if notch_frequency >= float(config["h_freq"]):
            notch_reason = (
                f"Requested {notch_frequency:g} Hz notch skipped because the final "
                f"low-pass is {float(config['h_freq']):g} Hz."
            )
        else:
            filtered.notch_filter(
                freqs=[notch_frequency],
                notch_widths=float(config["notch_width_hz"]),
                picks="eeg",
                phase="zero",
                verbose="ERROR",
            )
            notch_applied = True
            notch_reason = f"Applied configured {notch_frequency:g} Hz notch."
    logger.info("Applied %.1f–%.1f Hz zero-phase FIR filter", config["l_freq"], config["h_freq"])
    logger.info("Notch applied: %s (%s)", notch_applied, notch_reason)
    return filtered, notch_applied, notch_reason


def resample_eeg(raw, config: dict[str, Any], logger: logging.Logger):
    """Anti-alias and resample a filtered copy to the configured final rate."""
    target_sfreq = float(config["target_sfreq"])
    resampled = raw.copy()
    if not np.isclose(resampled.info["sfreq"], target_sfreq):
        resampled.resample(
            target_sfreq,
            npad=config.get("npad", "auto"),
            method=str(config.get("method", "fft")),
            verbose="ERROR",
        )
    logger.info(
        "Resampled filtered EEG from %.1f Hz to %.1f Hz",
        raw.info["sfreq"],
        resampled.info["sfreq"],
    )
    return resampled


def _valid_interpolation_channels(raw, channels: list[str]) -> tuple[list[str], list[str]]:
    montage = raw.get_montage()
    positions = montage.get_positions()["ch_pos"] if montage else {}
    valid, invalid = [], []
    for channel in channels:
        position = positions.get(channel)
        if position is not None and np.all(np.isfinite(position)) and np.linalg.norm(position) > 0:
            valid.append(channel)
        else:
            invalid.append(channel)
    return valid, invalid


def interpolate_bad_channels(raw, bad_channels: list[str], logger: logging.Logger):
    interpolated = raw.copy()
    valid, invalid = _valid_interpolation_channels(interpolated, bad_channels)
    interpolated.info["bads"] = list(valid)
    if valid:
        interpolated.interpolate_bads(reset_bads=True, mode="accurate", verbose="ERROR")
    interpolated.info["bads"] = list(invalid)
    logger.info("Interpolated recorded bad channels: %s", valid or "none")
    if invalid:
        logger.warning("Could not interpolate channels without valid positions: %s", invalid)
    return interpolated, valid, invalid


def rereference(raw, logger: logging.Logger, *, stage: str = "final"):
    rereferenced = raw.copy()
    rereferenced.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
    logger.info("Applied %s common-average reference", stage)
    return rereferenced


def _best_ica_interval(before, after, channels: list[str], default_start: float, duration_sec: float) -> float:
    difference = before.get_data(picks=channels) - after.get_data(picks=channels)
    sfreq = float(before.info["sfreq"])
    window = max(1, int(round(duration_sec * sfreq)))
    if difference.shape[1] <= window:
        return 0.0
    starts = np.arange(0, difference.shape[1] - window + 1, window)
    rms = [np.sqrt(np.mean(difference[:, start : start + window] ** 2)) for start in starts]
    if not rms or max(rms) <= np.finfo(float).eps:
        return default_start
    return float(starts[int(np.argmax(rms))] / sfreq)


def _write_decisions(qc_dir: Path, decisions: dict[str, Any]) -> None:
    path = qc_dir / "decisions.json"
    path.write_text(json.dumps(_json_ready(decisions), indent=2), encoding="utf-8")


def _participant_group(participant: dict[str, Any], participant_id: str) -> str:
    """Return a canonical diagnosis label across the two supported datasets."""
    if "GROUP" in participant and str(participant["GROUP"]).lower() != "nan":
        return str(participant["GROUP"])
    label = participant_id.removeprefix("sub-").lower()
    if label.startswith("pd"):
        return "PD"
    if label.startswith("hc"):
        return "Control"
    return "unknown"


def process_subject(
    set_path: str | Path,
    config: dict[str, Any],
    expected_channels: list[str],
    *,
    review_only: bool = False,
    require_review: bool = True,
    no_downsampling: bool = False,
    overwrite: bool = False,
    config_path: str | Path | None = None,
    skip_manual_ica_review: bool = False,
    reuse_existing_ica: bool = False,
    console_logging: bool = True,
) -> SubjectResult:
    """Preprocess one participant and save every decision and QC stage."""
    set_path = Path(set_path)
    participant_id = subject_id_from_path(set_path)
    session_id = session_id_from_path(set_path)
    # ``subject_id`` remains the output-analysis key for backward
    # compatibility. In a sessioned dataset it is deliberately a recording ID.
    subject_id = recording_id_from_path(set_path)
    dataset_dir = Path(config["project"]["dataset_dir"])
    output_dir = Path(config["project"]["output_dir"])
    task = str(config["project"]["task"])
    qc_dir = output_dir / "qc" / subject_id
    metadata_dir = output_dir / "metadata" / "subjects" / subject_id
    ica_dir = output_dir / "ica"
    cleaned_dir = output_dir / "cleaned_raw"
    epochs_dir = output_dir / "epochs"
    for directory in (qc_dir, metadata_dir, ica_dir, cleaned_dir, epochs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if overwrite:
        # Only remove previously generated numbered stage files for this exact
        # participant. This prevents stale "not applicable" markers or old ICA
        # pages from surviving a deliberate --overwrite rerun.
        for pattern in ("[0-9][0-9]_*.png", "[0-9][0-9]_*.txt"):
            for previous_stage_file in qc_dir.glob(pattern):
                previous_stage_file.unlink()

    logger = _subject_logger(
        subject_id,
        output_dir / "logs",
        console_logging=console_logging,
    )
    participant = participant_metadata(dataset_dir, participant_id)
    group = _participant_group(participant, participant_id)
    raw, provenance = load_subject(set_path, config["channels"]["auxiliary_names"])
    raw_original = raw.copy()
    missing_channels = sorted(set(expected_channels) - set(raw.ch_names))
    logger.info("Loading %s | participant=%s | group=%s | %.1f Hz | %d EEG channels | %.2f s", subject_id, participant_id, group, raw.info["sfreq"], len(raw.ch_names), raw.times[-1])
    logger.info("Original reference: %s | missing recorded channels: %s", provenance["original_reference"], missing_channels or "none")

    qcc = config["qc"]
    dpi = int(qcc["dpi"])
    channels = qc.select_channels(raw, qcc["preferred_channels"])
    start = float(qcc["trace_start_sec"])
    duration = float(qcc["trace_duration_sec"])
    qc.plot_signal(raw, channels, start, duration, "RAW EEG — unprocessed", qc_dir / "01_raw_signal.png", dpi)
    qc.plot_psd(raw, channels, float(qcc["psd_fmin_hz"]), float(qcc["psd_fmax_hz"]), "Raw EEG PSD", qc_dir / "02_raw_psd.png", dpi)

    filtered, notch_applied, notch_reason = filter_eeg(raw, config["filter"], logger)
    filtered = resample_eeg(filtered, config["resampling"], logger)
    expected_sfreq = float(config["resampling"]["target_sfreq"])
    if not np.isclose(filtered.info["sfreq"], expected_sfreq):
        raise RuntimeError(f"Expected final sampling frequency {expected_sfreq:g} Hz")
    qc.plot_signal(filtered, channels, start, duration, "FILTERED EEG — 1–100 Hz + 60 Hz notch, 250 Hz sampling", qc_dir / "03_filtered_signal.png", dpi)
    qc.plot_psd(filtered, channels, float(qcc["psd_fmin_hz"]), float(qcc["final_psd_fmax_hz"]), "Filtered EEG PSD — 1–100 Hz + 60 Hz notch", qc_dir / "04_filtered_psd.png", dpi)
    qc.plot_signal_comparison(raw, filtered, channels, start, duration, "raw 500 Hz", "1–100 Hz at 250 Hz", "Raw vs filtered/resampled EEG", qc_dir / "05_raw_vs_filtered.png", dpi)

    bad_result = detect_bad_channels(filtered, config["channels"])
    filtered.info["bads"] = list(bad_result.bad_channels)
    bad_result.metrics.to_csv(metadata_dir / "bad_channel_metrics.csv", index=False)
    logger.info("Confirmed bad recorded channels: %s", bad_result.bad_channels or "none")
    logger.info("Bad-channel reasons: %s", bad_result.reasons or "none")
    if not qc.plot_bad_channels(filtered, bad_result, start, duration, qc_dir / "06_bad_channels.png", dpi):
        qc.save_status(qc_dir, "06_bad_channels", "No bad recorded channels detected. Missing channels are listed separately and were not interpolated.")

    # ICLabel was trained on common-average-referenced data. Apply CAR after
    # bad-channel detection so marked channels are excluded from the reference.
    # A final CAR is calculated again after interpolation below.
    ica_referenced = rereference(filtered, logger, stage="pre-ICA")
    annotated, artifact_table = annotate_large_artifacts(ica_referenced, config["artifacts"])
    artifact_table.to_csv(metadata_dir / "temporal_artifacts.csv", index=False)
    logger.info("Added %d large-artifact annotation interval(s)", len(artifact_table))
    if not qc.plot_artifact_annotations(annotated, artifact_table, channels, qc_dir / "07_artifact_annotations.png", dpi):
        qc.save_status(qc_dir, "07_artifact_annotations", "No large temporal artifacts exceeded the conservative configured thresholds.")

    ica_path = ica_dir / f"{subject_id}_task-{task}_desc-preprocessing-ica.fif"
    if reuse_existing_ica:
        if not ica_path.is_file():
            raise FileNotFoundError(
                f"{subject_id}: reusable ICA was requested but is missing: {ica_path}"
            )
        ica = mne.preprocessing.read_ica(ica_path, verbose="ERROR")
        raw_for_ica = make_ica_copy(
            annotated,
            config["ica"],
            no_downsampling=no_downsampling,
        )
        eeg_picks = mne.pick_types(raw_for_ica.info, eeg=True, exclude="bads")
        expected_ica_channels = [raw_for_ica.ch_names[pick] for pick in eeg_picks]
        if list(ica.ch_names) != expected_ica_channels:
            raise RuntimeError(
                f"{subject_id}: saved ICA channel contract no longer matches the "
                "prepared EEG; rerun with explicit --overwrite to refit ICA"
            )
        logger.info(
            "Reused %d fitted ICA components from %s",
            ica.n_components_,
            ica_path,
        )
    else:
        ica, raw_for_ica = fit_ica(
            annotated,
            config["ica"],
            no_downsampling=no_downsampling,
        )
        ica.save(ica_path, overwrite=overwrite)
        logger.info(
            "Fitted %d ICA components at %.1f Hz (final signal remains %.1f Hz)",
            ica.n_components_,
            raw_for_ica.info["sfreq"],
            annotated.info["sfreq"],
        )
    # ICLabel receives the same 1-100 Hz, 250 Hz, CAR signal used to fit ICA.
    scores = score_ica_components(ica, raw_for_ica, config["ica"])
    proposed_components, proposed_reasons = proposed_ica_exclusions(scores)
    automatic_mode = bool(skip_manual_ica_review and not review_only)
    automatic_reasons = {
        component: reason.replace(
            "Visual confirmation required.",
            "Automatically accepted via --skip-manual-ica-review; not visually confirmed.",
        )
        for component, reason in proposed_reasons.items()
    }
    if automatic_mode:
        scores["automatic_removal"] = scores["proposed_exclusion"].astype(bool)
    scores.to_csv(metadata_dir / "ica_component_scores.csv", index=False)
    proposal_written = False
    if (review_only or automatic_mode) and config_path is not None:
        proposal_written = write_ica_review_proposal(
            config_path,
            subject_id,
            proposed_components,
            automatic_reasons if automatic_mode else proposed_reasons,
            automatic=automatic_mode,
        )
        if proposal_written:
            logger.info(
                "%s ICA proposal in %s: %s",
                "Recorded automatic" if automatic_mode else "Prefilled unconfirmed",
                config_path,
                proposed_components or "none",
            )
        else:
            logger.info(
                "Preserved confirmed ICA decision in %s; new machine candidates were %s",
                config_path,
                proposed_components or "none",
            )
    ranked_scores = scores if "iclabel_artifact_probability" in scores else None
    if ranked_scores is not None:
        qc.plot_ica_probabilities(scores, qc_dir / "08_ica_probabilities.png", dpi)
    else:
        qc.save_status(
            qc_dir,
            "08_ica_probabilities",
            "ICLabel was disabled; components are shown in original numerical order.",
        )
    qc.plot_ica_components(
        ica,
        annotated,
        qc_dir,
        dpi,
        int(qcc["ica_components_per_page"]),
        scores=ranked_scores,
    )
    qc.plot_ica_sources(
        ica,
        annotated,
        qc_dir,
        start,
        duration,
        dpi,
        int(qcc["ica_components_per_page"]),
        scores=ranked_scores,
    )
    qc.plot_ica_properties(ica, annotated, qc_dir, dpi, scores=ranked_scores)

    reviewed = is_ica_review_confirmed(config, subject_id)
    decisions: dict[str, Any] = {
        "subject_id": subject_id,
        "recording_id": subject_id,
        "participant_id": participant_id,
        "session_id": session_id,
        "group": group,
        "original_file": provenance["original_file"],
        "stored_channels": provenance["stored_channels"],
        "analysis_eeg_channels": provenance["analysis_eeg_channels"],
        "dropped_auxiliary_channels": provenance["dropped_auxiliary_channels"],
        "montage_source": provenance["montage_source"],
        "missing_channels": missing_channels,
        "bad_channel_candidates": bad_result.candidates,
        "confirmed_bad_channel_reasons": bad_result.reasons,
        "temporal_artifact_annotations": artifact_table.to_dict(orient="records"),
        "ica_reviewed": reviewed,
        "ica_selection_mode": (
            "automatic_iclabel" if automatic_mode else "manual_confirmed" if reviewed else "unreviewed"
        ),
        "ica_review_suggestions_only": scores.loc[scores["suggested_ocular_review"], "component"].astype(int).tolist(),
        "iclabel_ranked_components": (
            scores["component"].astype(int).tolist() if ranked_scores is not None else None
        ),
        "iclabel_proposed_exclusions": proposed_components,
        "iclabel_proposal_reasons": automatic_reasons if automatic_mode else proposed_reasons,
        "iclabel_proposal_written_to_config": proposal_written,
        "iclabel_model_input_note": (
            "Extended Infomax ICA and ICLabel use common-average-referenced "
            "1-100 Hz EEG at 250 Hz."
            if ranked_scores is not None
            else "ICLabel disabled by configuration."
        ),
        "preprocessing_signature": preprocessing_signature(config),
        "ica_fit_reused": bool(reuse_existing_ica),
        "ica_reference": "average",
        "automatic_ica_removal": automatic_mode,
        "notch_applied": notch_applied,
        "notch_reason": notch_reason,
        "temporary_ica_sampling_frequency": raw_for_ica.info["sfreq"],
        "final_sampling_frequency": annotated.info["sfreq"],
        "original_sampling_frequency": raw_original.info["sfreq"],
    }

    if review_only:
        _write_decisions(qc_dir, decisions)
        logger.info(
            "Review-only complete. Inspect ranked stages 08–10, edit the prefilled "
            "list if needed, then set manual_review_confirmed.%s to true.",
            subject_id,
        )
        return SubjectResult(subject_id, None, None, None, ica_path, True)
    if require_review and not reviewed and not automatic_mode:
        _write_decisions(qc_dir, decisions)
        raise RuntimeError(
            f"{subject_id}: ICA has not been visually confirmed. Inspect ranked stages "
            "08-10, edit the prefilled list if needed, and set "
            f"ica.manual_review_confirmed.{subject_id} to true."
        )

    if automatic_mode:
        components = proposed_components
        component_reasons = automatic_reasons
    elif reviewed:
        components, component_reasons = subject_manual_ica(config, subject_id)
    else:
        # --allow-unreviewed is a debugging escape hatch only. A machine-prefilled
        # proposal must never become an exclusion merely because that flag was used.
        components, component_reasons = [], {}
    decisions["ica_components_removed"] = components
    decisions["ica_component_removal_reasons"] = component_reasons
    logger.info(
        "ICA components removed (%s): %s",
        "automatic ICLabel proposal" if automatic_mode else "visual review",
        components or "none",
    )
    if not qc.plot_removed_ica_components(ica, annotated, components, component_reasons, qc_dir / "11_removed_ica_components.png", dpi):
        qc.save_status(
            qc_dir,
            "11_removed_ica_components",
            "No ICA components were selected for removal.",
        )

    after_ica = apply_ica_exclusions(annotated, ica, components)
    ica_start = _best_ica_interval(annotated, after_ica, channels, start, duration)
    qc.plot_signal_comparison(annotated, after_ica, channels, ica_start, duration, "before ICA", "after ICA", "BEFORE ICA vs AFTER ICA", qc_dir / "12_before_after_ica.png", dpi)
    qc.plot_ica_removed_signal(annotated, after_ica, channels, ica_start, duration, qc_dir / "13_ica_removed_signal.png", dpi)

    before_interpolation = after_ica.copy()
    interpolated, interpolated_channels, interpolation_failures = interpolate_bad_channels(after_ica, bad_result.bad_channels, logger)
    decisions["interpolated_channels"] = interpolated_channels
    decisions["bad_channels_not_interpolated"] = interpolation_failures
    if interpolated_channels:
        qc.plot_interpolation(before_interpolation, interpolated, interpolated_channels, start, duration, qc_dir, dpi)
    else:
        qc.save_status(qc_dir, "14_interpolation", "No recorded channels required interpolation.")

    before_reference = interpolated.copy()
    cleaned = rereference(interpolated, logger, stage="final post-interpolation")
    qc.plot_signal_comparison(before_reference, cleaned, channels, start, duration, "before reference", "average reference", "Re-referencing effect", qc_dir / "15_reference_comparison.png", dpi)
    qc.plot_signal(cleaned, channels, start, duration, "FINAL CLEANED EEG — 1–100 Hz + 60 Hz notch", qc_dir / "16_final_clean_signal.png", dpi)
    qc.plot_raw_vs_clean_intervals(raw_original, cleaned, channels, start, artifact_table, duration, qc_dir / "17_raw_vs_clean.png", dpi)
    qc.plot_psd_comparison(raw_original, cleaned, channels, float(qcc["psd_fmin_hz"]), float(qcc["final_psd_fmax_hz"]), "raw", "final cleaned", "Raw vs cleaned PSD — no independent normalization", qc_dir / "18_raw_vs_clean_psd.png", dpi)

    epoch_result = create_and_reject_epochs(cleaned, config["epochs"])
    epoch_result.rejection_table.to_csv(metadata_dir / "epoch_rejection.csv", index=False)
    if epoch_result.n_initial:
        qc.plot_epoch_rejection(epoch_result.all_epochs, epoch_result.rejection_table, channels, qc_dir / "19_epoch_rejection.png", dpi)
    else:
        qc.save_status(qc_dir, "19_epoch_rejection", "No complete fixed-length epochs were available.")
    if epoch_result.n_retained:
        qc.plot_epoch_psd(epoch_result.epochs, channels, float(qcc["psd_fmin_hz"]), float(qcc["final_psd_fmax_hz"]), qc_dir / "20_final_epoch_psd.png", dpi)
    else:
        qc.save_status(qc_dir, "20_final_epoch_psd", "No epochs remained after conservative rejection.")

    percent_retained = 100.0 * epoch_result.n_retained / max(1, epoch_result.n_initial)
    usable_duration = epoch_result.n_retained * float(config["epochs"]["duration_sec"])
    qc_row = {
        "subject_id": subject_id,
        "recording_id": subject_id,
        "participant_id": participant_id,
        "session_id": session_id,
        "group": group,
        "original_file": provenance["original_file"],
        "sampling_rate": float(cleaned.info["sfreq"]),
        "original_sampling_rate": float(raw_original.info["sfreq"]),
        "recording_duration_sec": float(cleaned.times[-1]),
        "original_channels": _list_text(provenance["stored_channels"]),
        "n_original_channels": len(provenance["stored_channels"]),
        "analysis_eeg_channels": _list_text(provenance["analysis_eeg_channels"]),
        "n_analysis_eeg_channels": len(provenance["analysis_eeg_channels"]),
        "dropped_auxiliary_channels": _list_text(provenance["dropped_auxiliary_channels"]),
        "montage_source": provenance["montage_source"],
        "missing_channels": _list_text(missing_channels),
        "bad_channels": _list_text(bad_result.bad_channels),
        "bad_channel_reasons": json.dumps(bad_result.reasons, sort_keys=True),
        "n_bad_channels": len(bad_result.bad_channels),
        "interpolated_channels": _list_text(interpolated_channels),
        "n_interpolated_channels": len(interpolated_channels),
        "bad_channels_not_interpolated": _list_text(interpolation_failures),
        "n_temporal_artifact_annotations": len(artifact_table),
        "annotated_bad_duration_sec": float(artifact_table["duration_sec"].sum()) if not artifact_table.empty else 0.0,
        "n_ica_components": int(ica.n_components_),
        "ica_fit_reused": bool(reuse_existing_ica),
        "ica_components_removed": _list_text(components),
        "ica_component_removal_reasons": json.dumps(component_reasons, sort_keys=True),
        "n_ica_components_removed": len(components),
        "ica_reviewed": reviewed,
        "ica_selection_mode": "automatic_iclabel" if automatic_mode else "manual_confirmed" if reviewed else "unreviewed_debug",
        "automatic_ica_removal": automatic_mode,
        "n_epochs_initial": epoch_result.n_initial,
        "n_epochs_rejected": epoch_result.n_rejected,
        "n_epochs_retained": epoch_result.n_retained,
        "percent_epochs_retained": percent_retained,
        "usable_duration_sec": usable_duration,
        "original_reference": provenance["original_reference"],
        "final_reference": "average",
        "filter_low_hz": float(config["filter"]["l_freq"]),
        "filter_high_hz": float(config["filter"]["h_freq"]),
        "notch_applied": notch_applied,
        "notch_reason": notch_reason,
        "temporary_ica_sampling_frequency": float(raw_for_ica.info["sfreq"]),
        "ica_reference": "average",
    }
    decisions["epoch_rejection"] = epoch_result.rejection_table.to_dict(orient="records")
    decisions["qc_summary"] = qc_row
    _write_decisions(qc_dir, decisions)
    qc.plot_summary(qc_row, qc_dir / "21_summary.png", dpi)

    cleaned_path = cleaned_dir / f"{subject_id}_task-{task}_desc-cleaned_raw.fif"
    epochs_path = epochs_dir / f"{subject_id}_task-{task}_desc-cleaned_epo.fif"
    cleaned.save(cleaned_path, overwrite=overwrite, verbose="ERROR")
    epoch_result.epochs.save(epochs_path, overwrite=overwrite, verbose="ERROR")
    logger.info("Saved final cleaned continuous EEG: %s", cleaned_path)
    logger.info("Epochs initial=%d rejected=%d retained=%d (%.1f%%; %.1f s)", epoch_result.n_initial, epoch_result.n_rejected, epoch_result.n_retained, percent_retained, usable_duration)
    return SubjectResult(subject_id, qc_row, cleaned_path, epochs_path, ica_path, False)
