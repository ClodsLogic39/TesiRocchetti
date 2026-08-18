"""Helper di query condivisi da server MCP e motore di alert.

Accesso al DB, preparazione delle serie (NaN preservati) e metriche scalari.
Contratto di risposta MCP: input_spec / selected_data / result / explain / warnings.
"""
import os

import numpy as np

from ..conversion import anatomy as bodyparts

# Nessun valore di ripiego: senza DSN esplicita libpq si collegherebbe al socket
# locale come utente del sistema operativo -- che di norma e' il proprietario del
# database, non biomech_mcp. I permessi per colonna che tengono l'identita' fuori
# dalla portata del server verrebbero aggirati in silenzio, e senza sslmode la
# connessione sarebbe anche in chiaro. Meglio non partire.
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
    """'knee' + 'right' -> 'knee_r'. Accetta anche un nome gia' canonico."""
    if joint in bodyparts.JOINTS:
        return joint
    suffix = _SIDE_MAP.get(str(side).lower()) if side else None
    key = f"{joint}_{suffix}" if suffix else joint
    return key if key in bodyparts.JOINTS else None


def fetch_angle(session_key, joint, plane, conn=None, dsn=None):
    """(fps, valori, provenienza, copertura), o quattro None se non c'e'.

    La provenienza dice se il valore veniva gia' dalla sorgente, se lo abbiamo
    calcolato noi, o se dipende da un'assunzione di modello. La copertura viaggia
    con essa e non va persa per strada: un massimo calcolato sul 17% dei frame e'
    il massimo di un sottoinsieme, e presentarlo senza dirlo e' disinformazione.
    """
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
    """Cicli gia' segmentati in conversione: [(start, end)] + metodo."""
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


def prepare_series(values, frame_start=None, frame_end=None):
    """Ritaglia e converte in array; None -> NaN. Restituisce warnings coerenti."""
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


# Testo libero che finisce nel contesto dell'agente (le note di un riferimento,
# la sorgente di una sessione). In locale lo scrive chi ha accesso al caricamento
# ed e' implicitamente fidato; in un sistema multi-utente sarebbe un vettore di
# iniezione indiretta. Si taglia e si appiattisce: un'istruzione su piu' righe
# diventa una riga sola di dato, non un blocco che somiglia a un comando.
MAX_TESTO = 300


def testo_citato(valore):
    """Campo testuale reso innocuo: una riga sola, lunghezza limitata."""
    if valore is None:
        return None
    s = " ".join(str(valore).split())
    return s if len(s) <= MAX_TESTO else s[:MAX_TESTO - 1] + "…"


# I nomi calcolabili, esposti perche' chi chiama possa verificarli PRIMA di
# calcolare: un nome fuori elenco e un segnale senza frame validi darebbero
# entrambi None, e chi legge non potrebbe distinguere "non esiste" da "non c'e'".
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


# Ordine di affidabilita': un confronto fra serie di provenienza diversa vale
# quanto la peggiore delle due.
_ORDINE_PROVENIENZA = ("measured", "derived", "filled", "modelled")


def worst_provenance(*valori):
    presenti = [v for v in valori if v]
    if not presenti:
        return None
    return max(presenti, key=lambda v: _ORDINE_PROVENIENZA.index(v)
               if v in _ORDINE_PROVENIENZA else len(_ORDINE_PROVENIENZA))


# Sotto questa copertura una metrica descrive un sottoinsieme di frame, non il
# gesto: e' la stessa soglia con cui il caricamento decide di non giudicare.
MIN_COVERAGE_AFFIDABILE = 0.30


def response(input_spec, signal, fps, n_frames, frame_range, warnings, result, explain):
    """Risposta standard. signal = {joint, plane, provenance, coverage} o None.

    Due avvisi nascono qui, e nessuno dei due modifica il valore: il numero
    resta quello che e', ma chi legge sa quanto pesa.

    `bassa_copertura` dice che la metrica e' calcolata su una minoranza di
    frame. Un massimo di flessione visto nel 17% dei fotogrammi non e' impreciso,
    e' il massimo di un altro insieme.

    `modelled_axis` dice che il piano dipende da un'assunzione di modello. Non
    scatta quando la sorgente forniva gia' i tre gradi di liberta': li' e' una
    misura come le altre, e avvisare sarebbe fuorviante.
    """
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
