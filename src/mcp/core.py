"""Funzioni di accesso al database condivise da server MCP e alert.

Apre la connessione di sola lettura, legge dal database la serie di un angolo e
i cicli di una sessione insieme a provenienza e copertura, prepara le serie per
il calcolo mantenendo i NaN, calcola le metriche scalari e compone la risposta
nel formato comune a tutti i tool:
input_spec / selected_data / result / explain / warnings.

Prerequisiti: psycopg e numpy; la DSN del ruolo di sola lettura nella variabile
d'ambiente MCP_DB_DSN, senza la quale la connessione non viene tentata.
"""
import os

import numpy as np

from ..conversion import anatomy as bodyparts

# --- Connessione e nomi ---
# DSN letta dall'ambiente e sinonimi accettati per il lato, ricondotti alla
# lettera usata nei nomi dei giunti.
DEFAULT_DSN = os.environ.get("MCP_DB_DSN")

_SIDE_MAP = {"left": "l", "l": "l", "sx": "l", "right": "r", "r": "r", "dx": "r"}


def connect(dsn=None):
    import psycopg
    scelta = dsn or DEFAULT_DSN
    if not scelta:
        raise RuntimeError(
            "MCP_DB_DSN non impostata: il server MCP deve collegarsi come "
            "biomech_mcp, il ruolo di sola lettura che non vede l'anagrafica. "
            "Esempio: MCP_DB_DSN='host=127.0.0.1 dbname=biomech "
            "user=biomech_mcp sslmode=require'")
    return psycopg.connect(scelta)


def canonical_joint(joint, side=None):
    if joint in bodyparts.JOINTS:
        return joint
    suffix = _SIDE_MAP.get(str(side).lower()) if side else None
    key = f"{joint}_{suffix}" if suffix else joint
    return key if key in bodyparts.JOINTS else None


# --- Lettura dal database ---
# Serie di un angolo con la sua provenienza e copertura, e intervalli dei cicli
# di una sessione con metodo e lato.
def fetch_angle(session_key, joint, plane, conn=None, dsn=None):
    if conn is None:
        with connect(dsn) as c:
            return fetch_angle(session_key, joint, plane, conn=c)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT s.fps, a.values, a.provenance, a.coverage FROM sessions s "
            "JOIN joint_angles a ON a.session_id = s.id "
            "WHERE s.session_key = %s AND a.joint = %s AND a.plane = %s",
            (session_key, joint, plane))
        row = cur.fetchone()
    if not row:
        return None, None, None, None
    return float(row[0]), row[1], row[2], (float(row[3]) if row[3] is not None else None)


def fetch_cycles(session_key, conn=None, dsn=None):
    if conn is None:
        with connect(dsn) as c:
            return fetch_cycles(session_key, conn=c)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.start_frame, c.end_frame, c.method FROM cycles c "
            "JOIN sessions s ON s.id = c.session_id "
            "WHERE s.session_key = %s ORDER BY c.cycle_index", (session_key,))
        rows = cur.fetchall()
    return [(r[0], r[1]) for r in rows], (rows[0][2] if rows else None)


# --- Preparazione delle serie e del testo ---
# Ritaglio della finestra di fotogrammi richiesta, con la maschera dei valori
# validi, e riduzione a una riga sola del testo libero letto dal database.
def prepare_series(values, frame_start=None, frame_end=None):
    warnings = []
    if values is None:
        return None, None, 0, None, ["signal_not_found"]
    arr = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    n = arr.shape[0]
    s = 0 if frame_start is None else max(0, int(frame_start))
    e = n if frame_end is None else min(n, int(frame_end))
    if s >= e:
        return None, None, n, (s, e), ["empty_range"]
    sliced = arr[s:e]
    mask = np.isfinite(sliced)
    if not mask.any():
        warnings.append("all_nan")
    return sliced, mask, n, (s, e), warnings


MAX_TESTO = 300


def testo_citato(valore):
    if valore is None:
        return None
    s = " ".join(str(valore).split())
    return s if len(s) <= MAX_TESTO else s[:MAX_TESTO - 1] + "…"


# --- Metriche e provenienza ---
# I nomi calcolabili, il loro calcolo sulla serie e la regola con cui il
# confronto fra serie diverse prende la provenienza meno affidabile.
METRICHE = ("rom", "peak_flexion", "peak_extension", "mean", "mean_abs")


def scalar_metric(series, mask, metric):
    if series is None or mask is None or not mask.any():
        return None
    f = series[mask]
    return {
        "rom": float(f.max() - f.min()),
        "peak_flexion": float(f.max()),
        "peak_extension": float(f.min()),
        "mean": float(f.mean()),
        "mean_abs": float(np.mean(np.abs(f))),
    }.get(metric)


_ORDINE_PROVENIENZA = ("measured", "derived", "filled", "modelled")


def worst_provenance(*valori):
    presenti = [v for v in valori if v]
    if not presenti:
        return None
    return max(presenti, key=lambda v: _ORDINE_PROVENIENZA.index(v)
               if v in _ORDINE_PROVENIENZA else len(_ORDINE_PROVENIENZA))


# --- Contratto di risposta ---
# Copertura sotto la quale la risposta porta un avviso, e la funzione che compone
# ogni risposta nel formato comune a tutti i tool.
MIN_COVERAGE_AFFIDABILE = 0.30


def response(input_spec, signal, fps, n_frames, frame_range, warnings, result, explain):
    w = list(warnings)
    sig = signal or {}
    if sig.get("plane") and sig["plane"] != "sagittal" \
            and sig.get("provenance") == "modelled":
        w.append("modelled_axis")
    copertura = sig.get("coverage")
    if copertura is not None and copertura < MIN_COVERAGE_AFFIDABILE:
        w.append("bassa_copertura")
    return {
        "input_spec": input_spec,
        "selected_data": {**sig, "fps": fps, "n_frames": n_frames,
                          "frame_range": frame_range, "units": "degrees"},
        "result": result,
        "explain": explain,
        "warnings": w,
    }
