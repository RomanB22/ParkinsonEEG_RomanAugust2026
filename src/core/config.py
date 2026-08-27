"""Configuration loading and validation.

The configuration file is JSON-formatted YAML. JSON is a strict subset of
YAML, so the file remains a valid ``.yaml`` document without adding PyYAML as a
project dependency.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "project",
        "filter",
        "resampling",
        "channels",
        "artifacts",
        "ica",
        "epochs",
        "qc",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")

    low = float(config["filter"]["l_freq"])
    high = float(config["filter"]["h_freq"])
    if (low, high) != (1.0, 100.0):
        raise ValueError(
            "The preprocessing contract requires final EEG to remain exactly 1-100 Hz; "
            f"received {low:g}-{high:g} Hz."
        )
    target_sfreq = float(config["resampling"]["target_sfreq"])
    if target_sfreq <= 2.0 * high:
        raise ValueError(
            "resampling.target_sfreq must be greater than twice filter.h_freq "
            f"({target_sfreq:g} <= {2.0 * high:g} Hz)"
        )
    if not bool(config["filter"].get("notch_enabled", False)):
        raise ValueError("The preprocessing contract requires the 60 Hz notch")
    if float(config["filter"].get("notch_freq_hz", 0.0)) != 60.0:
        raise ValueError("filter.notch_freq_hz must be 60 Hz")
    if (float(config["ica"]["fit_l_freq"]), float(config["ica"]["fit_h_freq"])) != (
        1.0,
        100.0,
    ):
        raise ValueError("ICLabel/ICA input must be filtered exactly 1-100 Hz")
    if float(config["epochs"]["duration_sec"]) <= 0:
        raise ValueError("epochs.duration_sec must be positive")
    if config["epochs"].get("baseline") is not None:
        raise ValueError("Prompt.md requires resting EEG epochs without baseline correction; epochs.baseline must be null")
    if bool(config["epochs"].get("autoreject_enabled", False)):
        raise ValueError(
            "AutoReject is not enabled in this conservative pipeline; "
            "epochs.autoreject_enabled must remain false"
        )
    if int(config["ica"]["random_state"]) < 0:
        raise ValueError("ica.random_state must be non-negative")
    for name in (
        "iclabel_artifact_probability_threshold",
        "iclabel_minimum_class_probability",
    ):
        value = float(config["ica"][name])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"ica.{name} must be between 0 and 1")
    exclusions = config["ica"].get("manual_exclude_components", {})
    confirmations = config["ica"].get("manual_review_confirmed", {})
    missing_decisions = [
        subject_id
        for subject_id, confirmed in confirmations.items()
        if bool(confirmed) and subject_id not in exclusions
    ]
    if missing_decisions:
        raise ValueError(
            "Confirmed ICA reviews lack exclusion lists: "
            f"{sorted(missing_decisions)}"
        )


def preprocessing_signature(config: dict[str, Any]) -> str:
    """Hash settings that determine cleaned EEG and ICA decomposition outputs."""
    ica = {
        key: value
        for key, value in config["ica"].items()
        if key
        not in {
            "manual_exclude_components",
            "manual_exclude_reasons",
            "manual_review_confirmed",
            "automatic_exclude_components",
            "automatic_exclude_reasons",
        }
    }
    relevant = {
        "filter": config["filter"],
        "resampling": config["resampling"],
        "channels": config["channels"],
        "artifacts": config["artifacts"],
        "ica": ica,
        "epochs": config["epochs"],
        "ica_reference": "average",
    }
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_ica_review_confirmed(config: dict[str, Any], subject_id: str) -> bool:
    """Return whether a person has visually confirmed this subject's ICA list."""
    return bool(config["ica"].get("manual_review_confirmed", {}).get(subject_id, False))


def subject_manual_ica(config: dict[str, Any], subject_id: str) -> tuple[list[int], dict[int, str]]:
    """Return reviewed ICA exclusions and their reasons for one participant."""
    exclusions = config["ica"].get("manual_exclude_components", {})
    reasons_config = config["ica"].get("manual_exclude_reasons", {})
    components = [int(value) for value in exclusions.get(subject_id, [])]
    reasons = {
        int(component): str(reason)
        for component, reason in reasons_config.get(subject_id, {}).items()
    }
    missing = [component for component in components if component not in reasons]
    if missing:
        raise ValueError(f"{subject_id}: missing rejection reasons for ICA components {missing}")
    return components, reasons


def write_ica_review_proposal(
    path: str | Path,
    subject_id: str,
    components: list[int],
    reasons: dict[int, str],
    *,
    automatic: bool = False,
) -> bool:
    """Atomically record an ICA proposal in the JSON-formatted YAML.

    Manual review proposals never overwrite confirmed decisions. Automatic-run
    selections are written to separate mappings so prior human decisions remain
    intact. The return value indicates whether a mapping was written.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    ica = config["ica"]
    confirmed = ica.setdefault("manual_review_confirmed", {})
    if automatic:
        # Preserve manual decisions while recording the exact list used by an
        # automatic run in its own auditable mapping.
        ica.setdefault("automatic_exclude_components", {})[subject_id] = [
            int(component) for component in components
        ]
        ica.setdefault("automatic_exclude_reasons", {})[subject_id] = {
            str(component): str(reason) for component, reason in reasons.items()
        }
    else:
        if bool(confirmed.get(subject_id, False)):
            return False
        ica.setdefault("manual_exclude_components", {})[subject_id] = [
            int(component) for component in components
        ]
        ica.setdefault("manual_exclude_reasons", {})[subject_id] = {
            str(component): str(reason) for component, reason in reasons.items()
        }
        confirmed[subject_id] = False

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return True
