#!/usr/bin/env python3
"""
Server MCP (FastMCP) per le query biomeccaniche.

Espone una ventina di tool metrici (rom, peak_flexion, angular_velocity,
relative_phase, symmetry_index, ...) che leggono i canali cinematici dalla
tabella c3d_analogs. Ogni tool restituisce sempre la stessa forma di risposta:
input_spec, selected_data, result, explain, warnings (garanzia di spiegabilita').
La DSN del database si configura con la variabile d'ambiente MCP_DB_DSN.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import psycopg
from scipy.signal import find_peaks

# Rimuove la cartella di questo file da sys.path PRIMA di importare `mcp`, cosi'
# `from mcp.server.fastmcp import FastMCP` risolve al pacchetto installato e non
# alla cartella locale src/mcp/ (che si chiama anch'essa "mcp"). Non rimuovere.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

from mcp.server.fastmcp import FastMCP  # noqa: E402


ROOT = SCRIPT_DIR
REGISTRY_PATH = ROOT / "label_registry.json"
QUERY_SPEC_PATH = ROOT / "query_1_5_spec.json"
DEFAULT_DSN = os.environ.get("MCP_DB_DSN", "dbname=biomech")


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File di configurazione mancante: {path}")
    return json.loads(path.read_text())


REGISTRY = load_json(REGISTRY_PATH)
QUERY_SPEC = load_json(QUERY_SPEC_PATH)


def resolve_label(joint, side, plane):
    joints = REGISTRY.get("joints", {})
    entry = joints.get(joint)
    if entry is None:
        return None
    if isinstance(entry, dict) and side:
        side_entry = entry.get(side)
        if isinstance(side_entry, dict):
            return side_entry.get(plane)
    if isinstance(entry, dict) and plane in entry:
        return entry.get(plane)
    return None


def fetch_signal(session_key, label, conn=None):
    if conn is None:
        with psycopg.connect(DEFAULT_DSN) as conn:
            return fetch_signal(session_key, label, conn=conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.fps, a.values
            FROM sessions s
            JOIN c3d_analogs a ON a.session_id = s.id
            WHERE s.session_key = %s AND a.label = %s
            """,
            (session_key, label),
        )
        row = cur.fetchone()
        if row is None:
            return None, None
        return float(row[0]), row[1]


def apply_percent_range(n_frames, frame_start, frame_end, percent_start, percent_end, warnings):
    if percent_start is None and percent_end is None:
        return frame_start, frame_end
    try:
        start_pct = 0.0 if percent_start is None else float(percent_start)
        end_pct = 100.0 if percent_end is None else float(percent_end)
    except (TypeError, ValueError):
        warnings.append("invalid_percent_range")
        return frame_start, frame_end
    if start_pct < 0 or end_pct > 100 or start_pct >= end_pct:
        warnings.append("invalid_percent_range")
        return frame_start, frame_end
    start = int(math.floor(n_frames * (start_pct / 100.0)))
    end = int(math.ceil(n_frames * (end_pct / 100.0)))
    return start, end


def prepare_series(values, frame_start, frame_end, percent_start=None, percent_end=None):
    warnings = []
    if values is None:
        return None, None, 0, None, ["signal_not_found"]

    safe = [v if v is not None else np.nan for v in values]
    arr = np.asarray(safe, dtype=float)
    n_frames = int(arr.shape[0])
    frame_start, frame_end = apply_percent_range(
        n_frames, frame_start, frame_end, percent_start, percent_end, warnings
    )
    start = 0 if frame_start is None else max(0, int(frame_start))
    end = n_frames if frame_end is None else min(n_frames, int(frame_end))
    if start >= end:
        return None, None, n_frames, (start, end), warnings + ["empty_range"]
    sliced = arr[start:end]
    mask = np.isfinite(sliced)
    if not mask.any():
        warnings.append("all_nan")
    return sliced, mask, n_frames, (start, end), warnings


def base_response(input_spec, label, fps, n_frames, frame_range, warnings):
    # Flag di onesta': solo gli angoli sagittali flexion_extension sono anatomicamente fondati.
    if label and "flexion_extension" not in label and "estimated_non_anatomical_axis" not in warnings:
        warnings.append("estimated_non_anatomical_axis")
    return {
        "input_spec": input_spec,
        "selected_data": {
            "label": label,
            "fps": fps,
            "n_frames": n_frames,
            "frame_range": frame_range,
            "units": REGISTRY.get("angle_units", "degrees"),
        },
        "warnings": warnings,
    }


def series_to_list(arr):
    return [float(v) if np.isfinite(v) else None for v in arr.tolist()]


def compute_metric(series, mask, metric):
    if series is None or mask is None or not mask.any():
        return None
    finite = series[mask]
    if metric == "rom":
        return float(np.max(finite) - np.min(finite))
    if metric == "peak_flexion":
        return float(np.max(finite))
    if metric == "peak_extension":
        return float(np.min(finite))
    if metric == "mean":
        return float(np.mean(finite))
    if metric == "mean_abs":
        return float(np.mean(np.abs(finite)))
    return None


def get_series(session_key, label, frame_start, frame_end, percent_start=None, percent_end=None, conn=None):
    fps, values = fetch_signal(session_key, label, conn=conn)
    series, mask, n_frames, frame_range, warnings = prepare_series(
        values, frame_start, frame_end, percent_start, percent_end
    )
    return fps, series, mask, n_frames, frame_range, warnings


def resample_series(series, n_samples):
    if series.size == 0:
        return None
    mask = np.isfinite(series)
    if int(mask.sum()) < 2:
        return None
    x = np.where(mask)[0]
    y = series[mask]
    target = np.linspace(0, len(series) - 1, n_samples)
    return np.interp(target, x, y)


def wrap_phase_deg(values):
    return (values + 180.0) % 360.0 - 180.0


def interp_nan(series):
    """Riempie i NaN interni per interpolazione lineare (per il rilevamento dei picchi). None se <2 valori finiti."""
    arr = np.asarray(series, dtype=float).copy()
    mask = np.isfinite(arr)
    if int(mask.sum()) < 2:
        return None
    idx = np.arange(arr.shape[0])
    arr[~mask] = np.interp(idx[~mask], idx[mask], arr[mask])
    return arr


def detect_cycles_from_signal(series, fps, min_period_s=0.3):
    """Cicli del passo = intervalli tra picchi successivi di un segnale ciclico di riferimento.
    Restituisce (cycles, warnings) dove cycles e' una lista di [frame_inizio, frame_fine]."""
    filled = interp_nan(series)
    if filled is None:
        return [], ["cycle_signal_insufficient"]
    distance = max(1, int(round(min_period_s * fps))) if fps else 1
    peaks, _ = find_peaks(filled, distance=distance)
    if len(peaks) < 2:
        return [], ["insufficient_peaks"]
    return [[int(peaks[i]), int(peaks[i + 1])] for i in range(len(peaks) - 1)], []


def normalize_for_phase(angle, velocity):
    """Normalizzazione alla Hamill prima della fase relativa continua:
    angolo -> [-1, 1] tramite min/max; velocita' -> [-1, 1] tramite max(|v|)."""
    a = np.asarray(angle, dtype=float)
    v = np.asarray(velocity, dtype=float)
    a_min = np.nanmin(a)
    a_max = np.nanmax(a)
    a_range = a_max - a_min
    a_norm = (2.0 * (a - a_min) / a_range - 1.0) if a_range > 1e-12 else np.zeros_like(a)
    v_peak = np.nanmax(np.abs(v))
    v_norm = (v / v_peak) if v_peak > 1e-12 else np.zeros_like(v)
    return a_norm, v_norm


mcp = FastMCP("Biomech MCP")


@mcp.tool()
def rom(session_key: str, joint: str, side: str | None = None, frame_start: int | None = None, frame_end: int | None = None):
    plane = "sagittal"
    label = resolve_label(joint, side, plane)
    input_spec = {"session_key": session_key, "joint": joint, "side": side, "plane": plane, "frame_start": frame_start, "frame_end": frame_end}
    if not label:
        return {**base_response(input_spec, None, None, 0, None, ["label_missing"]), "result": None, "explain": "No label resolved."}

    fps, values = fetch_signal(session_key, label)
    series, mask, n_frames, frame_range, warnings = prepare_series(values, frame_start, frame_end)
    response = base_response(input_spec, label, fps, n_frames, frame_range, warnings)
    if series is None or mask is None or not mask.any():
        response["result"] = None
        response["explain"] = "No finite data in range."
        return response

    finite = series[mask]
    min_val = float(np.min(finite))
    max_val = float(np.max(finite))
    response["result"] = {"min": min_val, "max": max_val, "rom": max_val - min_val}
    response["explain"] = "ROM = max(values) - min(values) on sagittal plane."
    return response


@mcp.tool()
def peak_flexion(session_key: str, joint: str, side: str | None = None, frame_start: int | None = None, frame_end: int | None = None):
    plane = "sagittal"
    label = resolve_label(joint, side, plane)
    input_spec = {"session_key": session_key, "joint": joint, "side": side, "plane": plane, "frame_start": frame_start, "frame_end": frame_end}
    if not label:
        return {**base_response(input_spec, None, None, 0, None, ["label_missing"]), "result": None, "explain": "No label resolved."}

    fps, values = fetch_signal(session_key, label)
    series, mask, n_frames, frame_range, warnings = prepare_series(values, frame_start, frame_end)
    response = base_response(input_spec, label, fps, n_frames, frame_range, warnings)
    if series is None or mask is None or not mask.any():
        response["result"] = None
        response["explain"] = "No finite data in range."
        return response

    peak = float(np.max(series[mask]))
    response["result"] = {"peak": peak}
    response["explain"] = "Peak flexion = max(values) on sagittal plane."
    return response


@mcp.tool()
def peak_extension(session_key: str, joint: str, side: str | None = None, frame_start: int | None = None, frame_end: int | None = None):
    plane = "sagittal"
    label = resolve_label(joint, side, plane)
    input_spec = {"session_key": session_key, "joint": joint, "side": side, "plane": plane, "frame_start": frame_start, "frame_end": frame_end}
    if not label:
        return {**base_response(input_spec, None, None, 0, None, ["label_missing"]), "result": None, "explain": "No label resolved."}

    fps, values = fetch_signal(session_key, label)
    series, mask, n_frames, frame_range, warnings = prepare_series(values, frame_start, frame_end)
    response = base_response(input_spec, label, fps, n_frames, frame_range, warnings)
    if series is None or mask is None or not mask.any():
        response["result"] = None
        response["explain"] = "No finite data in range."
        return response

    peak = float(np.min(series[mask]))
    response["result"] = {"peak": peak}
    response["explain"] = "Peak extension = min(values) on sagittal plane."
    return response


@mcp.tool()
def peak_timing(session_key: str, joint: str, side: str | None = None, peak: str = "flexion", frame_start: int | None = None, frame_end: int | None = None):
    plane = "sagittal"
    label = resolve_label(joint, side, plane)
    input_spec = {"session_key": session_key, "joint": joint, "side": side, "plane": plane, "peak": peak, "frame_start": frame_start, "frame_end": frame_end}
    if not label:
        return {**base_response(input_spec, None, None, 0, None, ["label_missing"]), "result": None, "explain": "No label resolved."}

    fps, values = fetch_signal(session_key, label)
    series, mask, n_frames, frame_range, warnings = prepare_series(values, frame_start, frame_end)
    response = base_response(input_spec, label, fps, n_frames, frame_range, warnings)
    if series is None or mask is None or not mask.any():
        response["result"] = None
        response["explain"] = "No finite data in range."
        return response

    finite = series.copy()
    finite[~mask] = -np.inf if peak == "flexion" else np.inf
    if peak == "flexion":
        idx = int(np.argmax(finite))
        peak_val = float(series[idx])
    else:
        idx = int(np.argmin(finite))
        peak_val = float(series[idx])

    frame_idx = (frame_range[0] if frame_range else 0) + idx
    time_s = float(frame_idx / fps) if fps else None
    response["result"] = {"peak": peak_val, "frame_index": frame_idx, "time_s": time_s}
    response["explain"] = "Peak timing = index of max/min converted to seconds using fps."
    return response


@mcp.tool()
def rom_frontal(session_key: str, joint: str, side: str | None = None, frame_start: int | None = None, frame_end: int | None = None):
    plane = "frontal"
    label = resolve_label(joint, side, plane)
    input_spec = {"session_key": session_key, "joint": joint, "side": side, "plane": plane, "frame_start": frame_start, "frame_end": frame_end}
    if not label:
        return {**base_response(input_spec, None, None, 0, None, ["label_missing"]), "result": None, "explain": "No label resolved."}

    fps, values = fetch_signal(session_key, label)
    series, mask, n_frames, frame_range, warnings = prepare_series(values, frame_start, frame_end)
    response = base_response(input_spec, label, fps, n_frames, frame_range, warnings)
    if series is None or mask is None or not mask.any():
        response["result"] = None
        response["explain"] = "No finite data in range."
        return response

    finite = series[mask]
    min_val = float(np.min(finite))
    max_val = float(np.max(finite))
    response["result"] = {"min": min_val, "max": max_val, "rom": max_val - min_val}
    response["explain"] = "ROM = max(values) - min(values) on frontal plane."
    return response


@mcp.tool()
def rom_transverse(session_key: str, joint: str, side: str | None = None, frame_start: int | None = None, frame_end: int | None = None):
    plane = "transverse"
    label = resolve_label(joint, side, plane)
    input_spec = {
        "session_key": session_key,
        "joint": joint,
        "side": side,
        "plane": plane,
        "frame_start": frame_start,
        "frame_end": frame_end,
    }
    if not label:
        return {
            **base_response(input_spec, None, None, 0, None, ["label_missing"]),
            "result": None,
            "explain": "No label resolved.",
        }

    fps, series, mask, n_frames, frame_range, warnings = get_series(session_key, label, frame_start, frame_end)
    response = base_response(input_spec, label, fps, n_frames, frame_range, warnings)
    if series is None or mask is None or not mask.any():
        response["result"] = None
        response["explain"] = "No finite data in range."
        return response

    finite = series[mask]
    min_val = float(np.min(finite))
    max_val = float(np.max(finite))
    response["result"] = {"min": min_val, "max": max_val, "rom": max_val - min_val}
    response["explain"] = "ROM = max(values) - min(values) on transverse plane."
    return response


@mcp.tool()
def angular_velocity(
    session_key: str,
    joint: str,
    side: str | None = None,
    plane: str = "sagittal",
    frame_start: int | None = None,
    frame_end: int | None = None,
    percent_start: float | None = None,
    percent_end: float | None = None,
    return_series: bool = False,
):
    label = resolve_label(joint, side, plane)
    input_spec = {
        "session_key": session_key,
        "joint": joint,
        "side": side,
        "plane": plane,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "percent_start": percent_start,
        "percent_end": percent_end,
    }
    if not label:
        return {
            **base_response(input_spec, None, None, 0, None, ["label_missing"]),
            "result": None,
            "explain": "No label resolved.",
        }

    fps, series, mask, n_frames, frame_range, warnings = get_series(
        session_key, label, frame_start, frame_end, percent_start, percent_end
    )
    response = base_response(input_spec, label, fps, n_frames, frame_range, warnings)
    if series is None or mask is None or not mask.any():
        response["result"] = None
        response["explain"] = "No finite data in range."
        return response
    if not fps or fps <= 0:
        warnings.append("fps_missing")
        response["warnings"] = warnings
        response["result"] = None
        response["explain"] = "Missing fps for velocity calculation."
        return response

    dt = 1.0 / fps
    velocity = np.gradient(series, dt)
    vel_mask = np.isfinite(velocity)
    if not vel_mask.any():
        warnings.append("all_nan")
        response["warnings"] = warnings
        response["result"] = None
        response["explain"] = "No finite velocity values."
        return response

    finite = velocity[vel_mask]
    units = REGISTRY.get("angle_units", "degrees")
    result = {
        "peak_positive": float(np.max(finite)),
        "peak_negative": float(np.min(finite)),
        "mean_abs": float(np.mean(np.abs(finite))),
        "units": f"{units}/s",
    }
    if return_series:
        result["series"] = series_to_list(velocity)
    response["result"] = result
    response["explain"] = "Angular velocity = d(angle)/dt."
    return response


@mcp.tool()
def angular_acceleration(
    session_key: str,
    joint: str,
    side: str | None = None,
    plane: str = "sagittal",
    frame_start: int | None = None,
    frame_end: int | None = None,
    percent_start: float | None = None,
    percent_end: float | None = None,
    return_series: bool = False,
):
    label = resolve_label(joint, side, plane)
    input_spec = {
        "session_key": session_key,
        "joint": joint,
        "side": side,
        "plane": plane,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "percent_start": percent_start,
        "percent_end": percent_end,
    }
    if not label:
        return {
            **base_response(input_spec, None, None, 0, None, ["label_missing"]),
            "result": None,
            "explain": "No label resolved.",
        }

    fps, series, mask, n_frames, frame_range, warnings = get_series(
        session_key, label, frame_start, frame_end, percent_start, percent_end
    )
    response = base_response(input_spec, label, fps, n_frames, frame_range, warnings)
    if series is None or mask is None or not mask.any():
        response["result"] = None
        response["explain"] = "No finite data in range."
        return response
    if not fps or fps <= 0:
        warnings.append("fps_missing")
        response["warnings"] = warnings
        response["result"] = None
        response["explain"] = "Missing fps for acceleration calculation."
        return response

    dt = 1.0 / fps
    velocity = np.gradient(series, dt)
    acceleration = np.gradient(velocity, dt)
    acc_mask = np.isfinite(acceleration)
    if not acc_mask.any():
        warnings.append("all_nan")
        response["warnings"] = warnings
        response["result"] = None
        response["explain"] = "No finite acceleration values."
        return response

    finite = acceleration[acc_mask]
    units = REGISTRY.get("angle_units", "degrees")
    result = {
        "peak_positive": float(np.max(finite)),
        "peak_negative": float(np.min(finite)),
        "mean_abs": float(np.mean(np.abs(finite))),
        "units": f"{units}/s^2",
    }
    if return_series:
        result["series"] = series_to_list(acceleration)
    response["result"] = result
    response["explain"] = "Angular acceleration = d²(angle)/dt²."
    return response


@mcp.tool()
def angular_impulse(
    session_key: str,
    joint: str,
    side: str | None = None,
    plane: str = "sagittal",
    frame_start: int | None = None,
    frame_end: int | None = None,
    percent_start: float | None = None,
    percent_end: float | None = None,
):
    label = resolve_label(joint, side, plane)
    input_spec = {
        "session_key": session_key,
        "joint": joint,
        "side": side,
        "plane": plane,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "percent_start": percent_start,
        "percent_end": percent_end,
    }
    if not label:
        return {
            **base_response(input_spec, None, None, 0, None, ["label_missing"]),
            "result": None,
            "explain": "No label resolved.",
        }

    fps, series, mask, n_frames, frame_range, warnings = get_series(
        session_key, label, frame_start, frame_end, percent_start, percent_end
    )
    response = base_response(input_spec, label, fps, n_frames, frame_range, warnings)
    if series is None or mask is None or not mask.any():
        response["result"] = None
        response["explain"] = "No finite data in range."
        return response
    if not fps or fps <= 0:
        warnings.append("fps_missing")
        response["warnings"] = warnings
        response["result"] = None
        response["explain"] = "Missing fps for impulse calculation."
        return response

    safe = series.copy()
    if mask is not None and (~mask).any():
        warnings.append("nan_filled_zero")
        safe[~mask] = 0.0
    dt = 1.0 / fps
    impulse = float(np.trapz(safe, dx=dt))
    units = REGISTRY.get("angle_units", "degrees")
    response["warnings"] = warnings
    response["result"] = {"impulse": impulse, "units": f"{units}*s"}
    response["explain"] = "Angular impulse = ∫ angle dt."
    return response


@mcp.tool()
def phase_profile(
    session_key: str,
    joint: str,
    side: str | None = None,
    plane: str = "sagittal",
    phase: str | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    percent_start: float | None = None,
    percent_end: float | None = None,
):
    label = resolve_label(joint, side, plane)
    input_spec = {
        "session_key": session_key,
        "joint": joint,
        "side": side,
        "plane": plane,
        "phase": phase,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "percent_start": percent_start,
        "percent_end": percent_end,
    }
    if not label:
        return {
            **base_response(input_spec, None, None, 0, None, ["label_missing"]),
            "result": None,
            "explain": "No label resolved.",
        }

    fps, series, mask, n_frames, frame_range, warnings = get_series(
        session_key, label, frame_start, frame_end, percent_start, percent_end
    )
    response = base_response(input_spec, label, fps, n_frames, frame_range, warnings)
    if series is None or mask is None or not mask.any():
        response["result"] = None
        response["explain"] = "No finite data in range."
        return response
    if phase and frame_start is None and frame_end is None and percent_start is None and percent_end is None:
        warnings.append("phase_range_missing")

    time_range_s = None
    if fps and frame_range:
        time_range_s = [float(frame_range[0] / fps), float(frame_range[1] / fps)]
    response["warnings"] = warnings
    response["result"] = {
        "phase": phase,
        "series": series_to_list(series),
        "frame_range": frame_range,
        "time_range_s": time_range_s,
    }
    response["explain"] = "Phase profile = angle series over selected range."
    return response


@mcp.tool()
def cycle_samples(
    session_key: str,
    joint: str,
    side: str | None = None,
    plane: str = "sagittal",
    percentiles: list[float] | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    percent_start: float | None = None,
    percent_end: float | None = None,
):
    label = resolve_label(joint, side, plane)
    input_spec = {
        "session_key": session_key,
        "joint": joint,
        "side": side,
        "plane": plane,
        "percentiles": percentiles,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "percent_start": percent_start,
        "percent_end": percent_end,
    }
    if not label:
        return {
            **base_response(input_spec, None, None, 0, None, ["label_missing"]),
            "result": None,
            "explain": "No label resolved.",
        }

    fps, series, mask, n_frames, frame_range, warnings = get_series(
        session_key, label, frame_start, frame_end, percent_start, percent_end
    )
    response = base_response(input_spec, label, fps, n_frames, frame_range, warnings)
    if series is None or mask is None or not mask.any():
        response["result"] = None
        response["explain"] = "No finite data in range."
        return response

    if percentiles is None:
        percentiles = [0.0, 25.0, 50.0, 75.0, 100.0]
    cleaned = []
    for p in percentiles:
        try:
            value = float(p)
        except (TypeError, ValueError):
            warnings.append("invalid_percentile")
            continue
        if value < 0 or value > 100:
            warnings.append("percentile_out_of_range")
            value = min(100.0, max(0.0, value))
        cleaned.append(value)
    if not cleaned:
        response["warnings"] = warnings
        response["result"] = None
        response["explain"] = "No valid percentiles provided."
        return response

    finite_idx = np.where(mask)[0]
    if finite_idx.size < 2:
        warnings.append("insufficient_finite_points")
        response["warnings"] = warnings
        response["result"] = None
        response["explain"] = "Not enough finite points for interpolation."
        return response

    positions = (np.asarray(cleaned, dtype=float) / 100.0) * (len(series) - 1)
    values = np.interp(positions, finite_idx, series[mask])
    units = REGISTRY.get("angle_units", "degrees")
    response["warnings"] = warnings
    response["result"] = {
        "percentiles": cleaned,
        "values": [float(v) for v in values.tolist()],
        "units": units,
    }
    response["explain"] = "Cycle samples = interpolated angles at percentiles."
    return response


@mcp.tool()
def relative_phase(
    session_key: str,
    joint_a: str,
    joint_b: str,
    side_a: str | None = None,
    side_b: str | None = None,
    plane: str = "sagittal",
    frame_start: int | None = None,
    frame_end: int | None = None,
    percent_start: float | None = None,
    percent_end: float | None = None,
    return_series: bool = False,
):
    label_a = resolve_label(joint_a, side_a, plane)
    label_b = resolve_label(joint_b, side_b, plane)
    input_spec = {
        "session_key": session_key,
        "joint_a": joint_a,
        "side_a": side_a,
        "joint_b": joint_b,
        "side_b": side_b,
        "plane": plane,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "percent_start": percent_start,
        "percent_end": percent_end,
    }
    if not label_a or not label_b:
        return {
            **base_response(input_spec, None, None, 0, None, ["label_missing"]),
            "result": None,
            "explain": "No label resolved.",
        }

    fps_a, series_a, mask_a, n_frames, frame_range, warnings = get_series(
        session_key, label_a, frame_start, frame_end, percent_start, percent_end
    )
    fps_b, series_b, mask_b, _, _, warnings_b = get_series(
        session_key, label_b, frame_start, frame_end, percent_start, percent_end
    )
    warnings.extend(warnings_b)
    response = base_response(input_spec, f"{label_a},{label_b}", fps_a, n_frames, frame_range, warnings)
    if series_a is None or series_b is None:
        response["result"] = None
        response["explain"] = "No data available for phase calculation."
        return response

    if not fps_a or fps_a <= 0:
        warnings.append("fps_missing")
        response["warnings"] = warnings
        response["result"] = None
        response["explain"] = "Missing fps for phase calculation."
        return response
    if fps_b and fps_b != fps_a:
        warnings.append("fps_mismatch")

    length = min(len(series_a), len(series_b))
    if len(series_a) != len(series_b):
        warnings.append("length_mismatch")
    series_a = series_a[:length]
    series_b = series_b[:length]
    dt = 1.0 / fps_a
    vel_a = np.gradient(series_a, dt)
    vel_b = np.gradient(series_b, dt)
    angle_a_n, vel_a_n = normalize_for_phase(series_a, vel_a)
    angle_b_n, vel_b_n = normalize_for_phase(series_b, vel_b)
    phase_a = np.degrees(np.arctan2(vel_a_n, angle_a_n))
    phase_b = np.degrees(np.arctan2(vel_b_n, angle_b_n))
    diff = wrap_phase_deg(phase_a - phase_b)
    mask = np.isfinite(diff)
    if not mask.any():
        warnings.append("all_nan")
        response["warnings"] = warnings
        response["result"] = None
        response["explain"] = "No finite phase values."
        return response

    finite = diff[mask]
    result = {
        "mean_phase_deg": float(np.mean(finite)),
        "std_phase_deg": float(np.std(finite)),
    }
    if return_series:
        result["series"] = series_to_list(diff)
    response["warnings"] = warnings
    response["result"] = result
    response["explain"] = "Continuous relative phase = phase(a) - phase(b), phase=atan2(norm_velocity, norm_angle) with amplitude normalization to [-1,1]."
    return response


@mcp.tool()
def angle_angle_curve(
    session_key: str,
    joint_x: str,
    joint_y: str,
    side_x: str | None = None,
    side_y: str | None = None,
    plane: str = "sagittal",
    frame_start: int | None = None,
    frame_end: int | None = None,
    percent_start: float | None = None,
    percent_end: float | None = None,
):
    label_x = resolve_label(joint_x, side_x, plane)
    label_y = resolve_label(joint_y, side_y, plane)
    input_spec = {
        "session_key": session_key,
        "joint_x": joint_x,
        "side_x": side_x,
        "joint_y": joint_y,
        "side_y": side_y,
        "plane": plane,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "percent_start": percent_start,
        "percent_end": percent_end,
    }
    if not label_x or not label_y:
        return {
            **base_response(input_spec, None, None, 0, None, ["label_missing"]),
            "result": None,
            "explain": "No label resolved.",
        }

    fps_x, series_x, mask_x, n_frames, frame_range, warnings = get_series(
        session_key, label_x, frame_start, frame_end, percent_start, percent_end
    )
    fps_y, series_y, mask_y, _, _, warnings_y = get_series(
        session_key, label_y, frame_start, frame_end, percent_start, percent_end
    )
    warnings.extend(warnings_y)
    response = base_response(input_spec, f"{label_x},{label_y}", fps_x, n_frames, frame_range, warnings)
    if series_x is None or series_y is None:
        response["result"] = None
        response["explain"] = "No data available for angle-angle curve."
        return response

    length = min(len(series_x), len(series_y))
    if len(series_x) != len(series_y):
        warnings.append("length_mismatch")
    series_x = series_x[:length]
    series_y = series_y[:length]
    mask = np.isfinite(series_x) & np.isfinite(series_y)
    response["warnings"] = warnings
    response["result"] = {
        "x_label": label_x,
        "y_label": label_y,
        "x_series": series_to_list(series_x),
        "y_series": series_to_list(series_y),
        "finite_points": int(mask.sum()),
    }
    response["explain"] = "Angle-angle curve = (angle_x, angle_y) pairs."
    return response


@mcp.tool()
def movement_variability(
    session_key: str,
    joint: str,
    side: str | None = None,
    plane: str = "sagittal",
    cycles: list[list[int]] | None = None,
    n_samples: int = 101,
):
    label = resolve_label(joint, side, plane)
    input_spec = {
        "session_key": session_key,
        "joint": joint,
        "side": side,
        "plane": plane,
        "cycles": cycles,
        "n_samples": n_samples,
    }
    if not label:
        return {
            **base_response(input_spec, None, None, 0, None, ["label_missing"]),
            "result": None,
            "explain": "No label resolved.",
        }

    fps, values = fetch_signal(session_key, label)
    warnings = []
    if values is None:
        return {
            **base_response(input_spec, label, fps, 0, None, ["signal_not_found"]),
            "result": None,
            "explain": "No signal found.",
        }
    safe = [v if v is not None else np.nan for v in values]
    arr = np.asarray(safe, dtype=float)
    n_frames = int(arr.shape[0])
    response = base_response(input_spec, label, fps, n_frames, (0, n_frames), warnings)

    if not cycles or len(cycles) < 2:
        ref_side = side if side in ("left", "right") else "left"
        ref_label = resolve_label("hip", ref_side, "sagittal")
        auto_cycles = []
        if ref_label:
            _, ref_values = fetch_signal(session_key, ref_label)
            if ref_values is not None:
                ref_arr = np.asarray([v if v is not None else np.nan for v in ref_values], dtype=float)
                auto_cycles, cyc_warn = detect_cycles_from_signal(ref_arr, fps)
                warnings.extend(cyc_warn)
        if len(auto_cycles) >= 2:
            cycles = auto_cycles
            warnings.append("cycles_autodetected")
        else:
            warnings.append("cycles_required")
            response["warnings"] = warnings
            response["result"] = None
            response["explain"] = "Provide at least two cycle ranges (auto-detection from hip flexion failed)."
            return response

    resampled = []
    for cycle in cycles:
        if not isinstance(cycle, (list, tuple)) or len(cycle) != 2:
            warnings.append("invalid_cycle_range")
            continue
        start = max(0, int(cycle[0]))
        end = min(n_frames, int(cycle[1]))
        if start >= end:
            warnings.append("invalid_cycle_range")
            continue
        segment = arr[start:end]
        sampled = resample_series(segment, n_samples)
        if sampled is None:
            warnings.append("cycle_insufficient_data")
            continue
        resampled.append(sampled)

    if len(resampled) < 2:
        warnings.append("insufficient_cycles")
        response["warnings"] = warnings
        response["result"] = None
        response["explain"] = "Not enough valid cycles for variability."
        return response

    stack = np.vstack(resampled)
    mean_profile = np.mean(stack, axis=0)
    std_profile = np.std(stack, axis=0)
    finite_std = std_profile[np.isfinite(std_profile)]
    result = {
        "n_cycles": len(resampled),
        "n_samples": int(n_samples),
        "mean_profile": series_to_list(mean_profile),
        "std_profile": series_to_list(std_profile),
        "mean_std": float(np.mean(finite_std)) if finite_std.size else None,
        "max_std": float(np.max(finite_std)) if finite_std.size else None,
    }
    response["warnings"] = warnings
    response["result"] = result
    response["explain"] = "Variability = std profile across resampled cycles."
    return response


@mcp.tool()
def symmetry_index(
    session_key: str,
    joint: str,
    plane: str = "sagittal",
    metric: str = "rom",
    frame_start: int | None = None,
    frame_end: int | None = None,
    percent_start: float | None = None,
    percent_end: float | None = None,
):
    label_left = resolve_label(joint, "left", plane)
    label_right = resolve_label(joint, "right", plane)
    input_spec = {
        "session_key": session_key,
        "joint": joint,
        "plane": plane,
        "metric": metric,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "percent_start": percent_start,
        "percent_end": percent_end,
    }
    if not label_left or not label_right:
        return {
            **base_response(input_spec, None, None, 0, None, ["label_missing"]),
            "result": None,
            "explain": "Missing left or right label.",
        }

    fps_l, series_l, mask_l, n_frames, frame_range, warnings = get_series(
        session_key, label_left, frame_start, frame_end, percent_start, percent_end
    )
    fps_r, series_r, mask_r, _, _, warnings_r = get_series(
        session_key, label_right, frame_start, frame_end, percent_start, percent_end
    )
    warnings.extend(warnings_r)
    response = base_response(input_spec, f"{label_left},{label_right}", fps_l, n_frames, frame_range, warnings)
    value_left = compute_metric(series_l, mask_l, metric)
    value_right = compute_metric(series_r, mask_r, metric)
    if value_left is None or value_right is None:
        warnings.append("metric_unavailable")
        response["warnings"] = warnings
        response["result"] = None
        response["explain"] = "Cannot compute symmetry index."
        return response

    denom = (abs(value_left) + abs(value_right)) / 2.0
    if denom == 0:
        warnings.append("zero_denominator")
        response["warnings"] = warnings
        response["result"] = None
        response["explain"] = "Cannot compute symmetry index with zero denominator."
        return response

    sym_index = 100.0 * (1.0 - abs(value_left - value_right) / denom)
    response["warnings"] = warnings
    response["result"] = {
        "left": float(value_left),
        "right": float(value_right),
        "symmetry_index": float(sym_index),
    }
    response["explain"] = "Symmetry index = 1 - |L-R| / mean(|L|,|R|)."
    return response


@mcp.tool()
def compare_sessions(
    session_keys: list[str],
    joint: str,
    side: str | None = None,
    plane: str = "sagittal",
    metric: str = "rom",
    frame_start: int | None = None,
    frame_end: int | None = None,
    percent_start: float | None = None,
    percent_end: float | None = None,
):
    label = resolve_label(joint, side, plane)
    input_spec = {
        "session_keys": session_keys,
        "joint": joint,
        "side": side,
        "plane": plane,
        "metric": metric,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "percent_start": percent_start,
        "percent_end": percent_end,
    }
    if not label:
        return {
            **base_response(input_spec, None, None, 0, None, ["label_missing"]),
            "result": None,
            "explain": "No label resolved.",
        }
    if not session_keys:
        return {
            **base_response(input_spec, label, None, 0, None, ["sessions_required"]),
            "result": None,
            "explain": "Provide at least one session key.",
        }

    warnings: list[str] = []
    results = []
    reference = None
    with psycopg.connect(DEFAULT_DSN) as conn:
        for key in session_keys:
            fps, series, mask, n_frames, frame_range, warn = get_series(
                key, label, frame_start, frame_end, percent_start, percent_end, conn=conn
            )
            warnings.extend(warn)
            value = compute_metric(series, mask, metric)
            if value is not None and reference is None:
                reference = value
            results.append(
                {
                    "session_key": key,
                    "value": float(value) if value is not None else None,
                    "fps": fps,
                    "n_frames": n_frames,
                    "frame_range": frame_range,
                }
            )

    response = base_response(input_spec, label, None, 0, None, warnings)
    if reference is None:
        response["result"] = None
        response["explain"] = "No valid metrics computed."
        return response

    for item in results:
        if item["value"] is None:
            item["delta"] = None
        else:
            item["delta"] = float(item["value"] - reference)
    response["result"] = {"reference": float(reference), "sessions": results}
    response["explain"] = "Compare sessions using the selected metric."
    return response


def fetch_session_keys_by_metadata(field, value, limit=None):
    query = "SELECT session_key FROM sessions WHERE metadata_json ->> %s = %s ORDER BY created_at"
    params = [field, value]
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)
    with psycopg.connect(DEFAULT_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return [row[0] for row in cur.fetchall()]


@mcp.tool()
def compare_sports(
    sport_a: str,
    sport_b: str,
    joint: str,
    side: str | None = None,
    plane: str = "sagittal",
    metric: str = "rom",
    sport_field: str = "sport",
    max_sessions: int | None = None,
):
    label = resolve_label(joint, side, plane)
    input_spec = {
        "sport_a": sport_a,
        "sport_b": sport_b,
        "joint": joint,
        "side": side,
        "plane": plane,
        "metric": metric,
        "sport_field": sport_field,
        "max_sessions": max_sessions,
    }
    if not label:
        return {
            **base_response(input_spec, None, None, 0, None, ["label_missing"]),
            "result": None,
            "explain": "No label resolved.",
        }

    session_keys_a = fetch_session_keys_by_metadata(sport_field, sport_a, max_sessions)
    session_keys_b = fetch_session_keys_by_metadata(sport_field, sport_b, max_sessions)
    warnings: list[str] = []
    if not session_keys_a:
        warnings.append("sport_a_empty")
    if not session_keys_b:
        warnings.append("sport_b_empty")

    def collect_values(keys, conn):
        values = []
        for key in keys:
            _, series, mask, _, _, warn = get_series(
                key, label, None, None, None, None, conn=conn
            )
            warnings.extend(warn)
            metric_value = compute_metric(series, mask, metric)
            if metric_value is not None:
                values.append(float(metric_value))
        return values

    with psycopg.connect(DEFAULT_DSN) as conn:
        values_a = collect_values(session_keys_a, conn)
        values_b = collect_values(session_keys_b, conn)

    def summarize(values):
        if not values:
            return {"n": 0, "mean": None, "std": None}
        arr = np.asarray(values, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    summary_a = summarize(values_a)
    summary_b = summarize(values_b)
    delta_mean = None
    if summary_a["mean"] is not None and summary_b["mean"] is not None:
        delta_mean = float(summary_a["mean"] - summary_b["mean"])

    response = base_response(input_spec, label, None, 0, None, warnings)
    response["result"] = {
        "sport_a": summary_a,
        "sport_b": summary_b,
        "delta_mean": delta_mean,
    }
    response["explain"] = "Compare sports using the selected metric."
    return response


@mcp.tool()
def chain_stability(
    session_key: str,
    joints: list[str],
    side: str | None = None,
    sides: list[str] | None = None,
    plane: str = "sagittal",
    frame_start: int | None = None,
    frame_end: int | None = None,
    percent_start: float | None = None,
    percent_end: float | None = None,
    tolerance: float | None = None,
):
    input_spec = {
        "session_key": session_key,
        "joints": joints,
        "side": side,
        "sides": sides,
        "plane": plane,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "percent_start": percent_start,
        "percent_end": percent_end,
        "tolerance": tolerance,
    }
    warnings: list[str] = []
    if not joints:
        return {
            **base_response(input_spec, None, None, 0, None, ["joints_required"]),
            "result": None,
            "explain": "Provide joints for chain stability.",
        }
    if sides is not None and len(sides) != len(joints):
        warnings.append("sides_length_mismatch")
        sides = None

    series_list = []
    labels = []
    fps = None
    n_frames = 0
    frame_range = None
    for idx, joint in enumerate(joints):
        joint_side = side if sides is None else sides[idx]
        label = resolve_label(joint, joint_side, plane)
        if not label:
            warnings.append("label_missing")
            continue
        labels.append(label)
        fps_val, series, mask, n_frames_val, frame_range_val, warn = get_series(
            session_key, label, frame_start, frame_end, percent_start, percent_end
        )
        warnings.extend(warn)
        if series is None or mask is None or not mask.any():
            warnings.append("empty_series")
            continue
        series_list.append(series)
        fps = fps_val if fps is None else fps
        n_frames = n_frames_val
        frame_range = frame_range_val

    response = base_response(input_spec, ",".join(labels) if labels else None, fps, n_frames, frame_range, warnings)
    if len(series_list) < 2:
        response["result"] = None
        response["explain"] = "Not enough joint series for chain stability."
        return response

    length = min(len(series) for series in series_list)
    stack = np.vstack([series[:length] for series in series_list])
    mask = np.all(np.isfinite(stack), axis=0)
    if not mask.any():
        warnings.append("all_nan")
        response["warnings"] = warnings
        response["result"] = None
        response["explain"] = "No finite chain values."
        return response

    chain_series = np.sum(stack, axis=0)
    finite = chain_series[mask]
    range_val = float(np.max(finite) - np.min(finite))
    std_val = float(np.std(finite))
    stable = None
    if tolerance is not None:
        stable = range_val <= float(tolerance)
    response["warnings"] = warnings
    response["result"] = {"range": range_val, "std": std_val, "stable": stable}
    response["explain"] = "Chain stability = range of summed joint angles."
    return response


@mcp.tool()
def gesture_detection(
    session_key: str,
    joint: str,
    side: str | None = None,
    plane: str = "sagittal",
    threshold: float = 0.0,
    direction: str = "above",
    min_duration_s: float | None = None,
    min_frames: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    percent_start: float | None = None,
    percent_end: float | None = None,
):
    label = resolve_label(joint, side, plane)
    input_spec = {
        "session_key": session_key,
        "joint": joint,
        "side": side,
        "plane": plane,
        "threshold": threshold,
        "direction": direction,
        "min_duration_s": min_duration_s,
        "min_frames": min_frames,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "percent_start": percent_start,
        "percent_end": percent_end,
    }
    if not label:
        return {
            **base_response(input_spec, None, None, 0, None, ["label_missing"]),
            "result": None,
            "explain": "No label resolved.",
        }

    fps, series, mask, n_frames, frame_range, warnings = get_series(
        session_key, label, frame_start, frame_end, percent_start, percent_end
    )
    response = base_response(input_spec, label, fps, n_frames, frame_range, warnings)
    if series is None or mask is None or not mask.any():
        response["result"] = None
        response["explain"] = "No finite data in range."
        return response

    if min_frames is None:
        if min_duration_s is not None and fps:
            min_frames = int(math.ceil(float(min_duration_s) * fps))
        else:
            min_frames = 1

    direction_key = direction.lower() if isinstance(direction, str) else "above"
    if direction_key in ("below", "extension", "negative"):
        condition = (series <= threshold) & np.isfinite(series)
    else:
        condition = (series >= threshold) & np.isfinite(series)

    segments = []
    start = None
    for idx, flag in enumerate(condition):
        if flag and start is None:
            start = idx
        if not flag and start is not None:
            end = idx
            if end - start >= min_frames:
                segment = series[start:end]
                peak = float(np.max(segment)) if direction_key not in ("below", "extension", "negative") else float(np.min(segment))
                segments.append(
                    {
                        "start_frame": (frame_range[0] if frame_range else 0) + start,
                        "end_frame": (frame_range[0] if frame_range else 0) + end,
                        "duration_frames": int(end - start),
                        "duration_s": float((end - start) / fps) if fps else None,
                        "peak": peak,
                    }
                )
            start = None
    if start is not None:
        end = len(series)
        if end - start >= min_frames:
            segment = series[start:end]
            peak = float(np.max(segment)) if direction_key not in ("below", "extension", "negative") else float(np.min(segment))
            segments.append(
                {
                    "start_frame": (frame_range[0] if frame_range else 0) + start,
                    "end_frame": (frame_range[0] if frame_range else 0) + end,
                    "duration_frames": int(end - start),
                    "duration_s": float((end - start) / fps) if fps else None,
                    "peak": peak,
                }
            )

    response["result"] = {
        "count": len(segments),
        "segments": segments,
    }
    response["explain"] = "Gesture detection = threshold crossings with minimum duration."
    return response


@mcp.tool()
def detect_cycles(
    session_key: str,
    joint: str = "hip",
    side: str | None = "left",
    plane: str = "sagittal",
    min_period_s: float = 0.3,
):
    label = resolve_label(joint, side, plane)
    input_spec = {
        "session_key": session_key,
        "joint": joint,
        "side": side,
        "plane": plane,
        "min_period_s": min_period_s,
    }
    if not label:
        return {
            **base_response(input_spec, None, None, 0, None, ["label_missing"]),
            "result": None,
            "explain": "No label resolved.",
        }

    fps, values = fetch_signal(session_key, label)
    if values is None:
        return {
            **base_response(input_spec, label, fps, 0, None, ["signal_not_found"]),
            "result": None,
            "explain": "No signal found.",
        }

    safe = [v if v is not None else np.nan for v in values]
    arr = np.asarray(safe, dtype=float)
    n_frames = int(arr.shape[0])
    cycles, warnings = detect_cycles_from_signal(arr, fps, min_period_s)
    response = base_response(input_spec, label, fps, n_frames, (0, n_frames), list(warnings))
    response["result"] = {
        "n_cycles": len(cycles),
        "cycles": cycles,
        "reference_label": label,
    }
    response["explain"] = "Gait cycles = frame intervals between successive peaks of the reference joint angle (default hip flexion)."
    return response


if __name__ == "__main__":
    mcp.run()
