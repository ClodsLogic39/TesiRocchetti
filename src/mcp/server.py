"""Server MCP (FastMCP) che espone le query biomeccaniche come strumenti.

Ogni strumento restituisce sempre la stessa struttura:
input_spec / selected_data / result / explain / warnings, con provenienza e
copertura dei dati su cui la risposta si basa.

Gli strumenti sono divisi per famiglia: metriche e confronti su una serie
angolare, ricerca delle sessioni disponibili, confronti di forma calcolati da
pgvector, alert registrati dal caricamento, apertura del viewer e, quando
abilitata, la scrittura di una nuova sessione dalla chat.

La connessione del server resta di sola lettura in ogni caso: la scrittura, se
attiva, passa da `src.conversion.process` e `src.database.ingest` lanciati come
sottoprocessi, con la DSN di scrittura nell'ambiente e mai negli argomenti.

Avvio: python -m src.mcp.server

Prerequisiti: il pacchetto mcp, psycopg e numpy; la DSN di sola lettura nella
variabile d'ambiente MCP_DB_DSN; per il viewer un interprete con open3d ed ezc3d
e i C3D canonici nella cartella dei dati elaborati; per la scrittura dalla chat,
BIOMECH_INGEST_ENABLED=1 e la DSN di scrittura in INGEST_DB_DSN.
"""
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

from . import core
from ..conversion import anatomy as bodyparts
from ..conversion import canonical
from ..conversion import process
from ..conversion import readers
from ..conversion import segmentation
from ..database import similarity

# --- Percorsi e interruttori letti dall'ambiente ---
# Cartella dei C3D canonici, interprete e modulo con cui il viewer viene lanciato,
# e interruttore che impedisce di aprire finestre sulla macchina del server.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("BIOMECH_DATA_ROOT",
                                REPO_ROOT / process.USCITA_DEFAULT))
VIEWER_PY = os.environ.get("BIOMECH_VIEWER_PYTHON", sys.executable)
VIEWER_MODULE = "src.visualization.viewer_open3d"
VIEWER_DISABLED = bool(os.environ.get("BIOMECH_VIEWER_DISABLED"))

# Scrittura dalla chat: spenta di default. Le radici ammesse per una sorgente
# indicata per percorso, e dove va il log dei tentativi -- un canale che non
# passa dal modello, per quando la risposta in chat non basta.
INGEST_ENABLED = bool(os.environ.get("BIOMECH_INGEST_ENABLED"))
INGEST_DB_DSN = os.environ.get("INGEST_DB_DSN")
INGEST_ROOTS = [r for r in os.environ.get(
    "BIOMECH_INGEST_ROOTS", process.RADICE_DEFAULT).split(os.pathsep) if r]
INGEST_LOG = Path(os.environ.get("BIOMECH_INGEST_LOG", str(DATA_ROOT / "ingest.log")))
INGEST_TIMEOUT = 600

mcp = FastMCP("Biomech MCP")


# --- Funzioni di supporto agli strumenti ---
# Risoluzione di giunto, lato e piano, lettura della serie, copertura peggiore di
# un confronto, e i controlli che producono una risposta di rifiuto: segnale non
# risolto, chiave di sessione non valida, C3D mancante, viewer senza dipendenze,
# nome di metrica inesistente.
def _signal(joint, plane, side=None):
    key = core.canonical_joint(joint, side)
    if not key or plane not in bodyparts.PLANES:
        return None
    return {"joint": key, "plane": plane}


def _load(session_key, sig, frame_start=None, frame_end=None, conn=None):
    fps, values, provenance, coverage = core.fetch_angle(
        session_key, sig["joint"], sig["plane"], conn=conn)
    sig["provenance"] = provenance
    sig["coverage"] = coverage
    return (fps, *core.prepare_series(values, frame_start, frame_end))


def _peggior_copertura(valori):
    presenti = [v for v in valori if v is not None]
    return min(presenti) if presenti else None


def _no_signal(spec, explain="Giunto o piano fuori vocabolario."):
    return core.response(spec, None, None, 0, None, ["signal_unresolved"], None, explain)


def _c3d_sessione(spec, session_key):
    def rifiuto(explain, avviso):
        return None, {"input_spec": spec, "selected_data": {}, "result": None,
                      "explain": explain, "warnings": [avviso]}

    if (not session_key or session_key != os.path.basename(session_key)
            or session_key in (".", "..")):
        return rifiuto("session_key non valida: deve essere un nome semplice, "
                       "senza percorsi.", "invalid_session_key")
    c3d = DATA_ROOT / f"{session_key}.c3d"
    if not c3d.is_file():
        return rifiuto("C3D non trovato: esegui prima la conversione.", "c3d_not_found")
    return c3d, None


def _viewer_pronto(spec):
    try:
        ok = subprocess.run([VIEWER_PY, "-c", "import open3d, ezc3d"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    if ok:
        return None
    return {"input_spec": spec, "selected_data": {}, "result": None,
            "explain": "Interprete viewer senza open3d+ezc3d (imposta BIOMECH_VIEWER_PYTHON).",
            "warnings": ["viewer_deps_missing"]}


def _metrica_ignota(spec, metrica):
    if metrica in core.METRICHE:
        return None
    return core.response(spec, None, None, 0, None, ["unknown_metric"], None,
                         f"Metrica '{metrica}' inesistente. "
                         f"Disponibili: {', '.join(core.METRICHE)}.")


# --- Strumenti: metriche e confronti su una serie angolare ---
@mcp.tool()
def metric(session_key: str, joint: str, side: str | None = None, plane: str = "sagittal",
           metric: str = "rom", frame_start: int | None = None, frame_end: int | None = None):
    """ROM / picco flessione / picco estensione / media su un piano.

    plane: sagittal | frontal | transverse. Il trasverso esiste solo dove e'
    misurato, cioe' nelle sessioni importate da C3D a 3 DOF.
    """
    spec = {"session_key": session_key, "joint": joint, "side": side, "plane": plane,
            "metric": metric, "frame_start": frame_start, "frame_end": frame_end}
    ignota = _metrica_ignota(spec, metric)
    if ignota:
        return ignota
    sig = _signal(joint, plane, side)
    if not sig:
        return _no_signal(spec)
    fps, series, mask, n, fr, warns = _load(session_key, sig, frame_start, frame_end)
    val = core.scalar_metric(series, mask, metric)
    return core.response(spec, sig, fps, n, fr, warns,
                         None if val is None else {"value": val, "metric": metric},
                         f"{metric} sul piano {plane}.")


@mcp.tool()
def angular_kinematics(session_key: str, joint: str, side: str | None = None, plane: str = "sagittal",
                       order: int = 1, frame_start: int | None = None, frame_end: int | None = None):
    """order=1 velocita' angolare, order=2 accelerazione (derivata dell'angolo)."""
    spec = {"session_key": session_key, "joint": joint, "side": side, "plane": plane,
            "order": order, "frame_start": frame_start, "frame_end": frame_end}
    sig = _signal(joint, plane, side)
    if not sig:
        return _no_signal(spec)
    fps, series, mask, n, fr, warns = _load(session_key, sig, frame_start, frame_end)
    if series is None or not fps:
        return core.response(spec, sig, fps, n, fr, warns + ["no_data_or_fps"], None, "No data/fps.")
    d = np.gradient(series, 1.0 / fps)
    if order == 2:
        d = np.gradient(d, 1.0 / fps)
    finite = d[np.isfinite(d)]
    unit = "degrees/s" if order == 1 else "degrees/s^2"
    result = None if finite.size == 0 else {
        "peak_positive": float(finite.max()), "peak_negative": float(finite.min()),
        "mean_abs": float(np.mean(np.abs(finite))), "units": unit}
    return core.response(spec, sig, fps, n, fr, warns, result, f"d^{order}(angle)/dt^{order}.")


@mcp.tool()
def peak_timing(session_key: str, joint: str, side: str | None = None, plane: str = "sagittal",
                peak: str = "flexion", frame_start: int | None = None, frame_end: int | None = None):
    """Istante (frame e secondi) del picco di flessione/estensione."""
    spec = {"session_key": session_key, "joint": joint, "side": side, "plane": plane, "peak": peak}
    sig = _signal(joint, plane, side)
    if not sig:
        return _no_signal(spec)
    fps, series, mask, n, fr, warns = _load(session_key, sig, frame_start, frame_end)
    if series is None or not mask.any():
        return core.response(spec, sig, fps, n, fr, warns, None, "No data.")
    tmp = series.copy()
    tmp[~mask] = -np.inf if peak == "flexion" else np.inf
    idx = int(np.argmax(tmp) if peak == "flexion" else np.argmin(tmp))
    frame_idx = (fr[0] if fr else 0) + idx
    result = {"peak": float(series[idx]), "frame_index": frame_idx,
              "time_s": float(frame_idx / fps) if fps else None}
    return core.response(spec, sig, fps, n, fr, warns, result, "Indice del picco -> secondi.")


@mcp.tool()
def symmetry_index(session_key: str, joint: str, plane: str = "sagittal", metric: str = "rom"):
    """Indice di simmetria sx-dx = 1 - |L-R| / media(|L|,|R|)."""
    spec = {"session_key": session_key, "joint": joint, "plane": plane, "metric": metric}
    ignota = _metrica_ignota(spec, metric)
    if ignota:
        return ignota
    sl, sr = _signal(joint, plane, "left"), _signal(joint, plane, "right")
    if not sl or not sr:
        return _no_signal(spec, "Giunto senza lati distinti.")
    with core.connect() as conn:
        vl = core.scalar_metric(*_load(session_key, sl, conn=conn)[1:3], metric)
        vr = core.scalar_metric(*_load(session_key, sr, conn=conn)[1:3], metric)
    sig = {"joint": f"{sl['joint']},{sr['joint']}", "plane": plane,
           "provenance": core.worst_provenance(sl.get("provenance"),
                                              sr.get("provenance"))}
    if vl is None or vr is None:
        return core.response(spec, sig, None, 0, None, ["metric_unavailable"], None, "n/d")
    denom = (abs(vl) + abs(vr)) / 2
    si = 100.0 * (1 - abs(vl - vr) / denom) if denom else None
    return core.response(spec, sig, None, 0, None, [],
                         {"left": vl, "right": vr, "symmetry_index": si}, "1-|L-R|/mean(|L|,|R|).")


@mcp.tool()
def compare_sessions(session_keys: list[str], joint: str, side: str | None = None,
                     plane: str = "sagittal", metric: str = "rom"):
    """Confronta una metrica tra piu' sessioni (delta rispetto alla prima valida)."""
    spec = {"session_keys": session_keys, "joint": joint, "side": side, "plane": plane,
            "metric": metric}
    ignota = _metrica_ignota(spec, metric)
    if ignota:
        return ignota
    sig = _signal(joint, plane, side)
    if not sig:
        return _no_signal(spec)
    rows, ref, provenienze, coperture = [], None, [], []
    with core.connect() as conn:
        for key in session_keys:
            _, series, mask, _, _, _ = _load(key, sig, conn=conn)
            provenienze.append(sig.get("provenance"))
            coperture.append(sig.get("coverage"))
            v = core.scalar_metric(series, mask, metric)
            if v is not None and ref is None:
                ref = v
            rows.append({"session_key": key, "value": v,
                         "provenance": sig.get("provenance")})
    for r in rows:
        r["delta"] = None if r["value"] is None or ref is None else r["value"] - ref
    sig["provenance"] = core.worst_provenance(*provenienze)
    sig["coverage"] = _peggior_copertura(coperture)
    distinte = {r["provenance"] for r in rows}
    warns = [] if len(distinte) <= 1 else ["mixed_provenance"]
    return core.response(spec, sig, None, 0, None, warns,
                         {"reference": ref, "sessions": rows}, "Delta di metrica tra sessioni.")


@mcp.tool()
def compare_sports(sport_a: str, sport_b: str, joint: str, side: str | None = None,
                   plane: str = "sagittal", metric: str = "rom"):
    """Confronta la media di una metrica tra due sport."""
    spec = {"sport_a": sport_a, "sport_b": sport_b, "joint": joint, "side": side,
            "plane": plane, "metric": metric}
    ignota = _metrica_ignota(spec, metric)
    if ignota:
        return ignota
    sig = _signal(joint, plane, side)
    if not sig:
        return _no_signal(spec)

    def summarize(sport, conn):
        with conn.cursor() as cur:
            cur.execute("SELECT session_key FROM sessions WHERE sport = %s", (sport,))
            keys = [r[0] for r in cur.fetchall()]
        vals = []
        for k in keys:
            _, series, mask, _, _, _ = _load(k, sig, conn=conn)
            provenienze.append(sig.get("provenance"))
            coperture.append(sig.get("coverage"))
            v = core.scalar_metric(series, mask, metric)
            if v is not None:
                vals.append(v)
        arr = np.asarray(vals)
        return {"n": arr.size, "mean": float(arr.mean()) if arr.size else None,
                "std": float(arr.std()) if arr.size else None}

    provenienze, coperture = [], []
    with core.connect() as conn:
        a, b = summarize(sport_a, conn), summarize(sport_b, conn)
    delta = None if a["mean"] is None or b["mean"] is None else a["mean"] - b["mean"]
    sig["provenance"] = core.worst_provenance(*provenienze)
    sig["coverage"] = _peggior_copertura(coperture)
    distinte = {p for p in provenienze if p}
    avvisi = [] if len(distinte) <= 1 else ["mixed_provenance"]
    return core.response(spec, sig, None, 0, None, avvisi,
                         {"sport_a": a, "sport_b": b, "delta_mean": delta}, "Media metrica per sport.")


@mcp.tool()
def relative_phase(session_key: str, joint_a: str, joint_b: str, side_a: str | None = None,
                   side_b: str | None = None, plane: str = "sagittal"):
    """Fase relativa continua (normalizzata in ampiezza) tra due giunti."""
    spec = {"session_key": session_key, "joint_a": joint_a, "joint_b": joint_b,
            "side_a": side_a, "side_b": side_b, "plane": plane}
    sa_sig, sb_sig = _signal(joint_a, plane, side_a), _signal(joint_b, plane, side_b)
    if not sa_sig or not sb_sig:
        return _no_signal(spec)
    with core.connect() as conn:
        fps, sa, ma, n, fr, w = _load(session_key, sa_sig, conn=conn)
        _, sb, mb, _, _, wb = _load(session_key, sb_sig, conn=conn)
    sig = {"joint": f"{sa_sig['joint']},{sb_sig['joint']}", "plane": plane,
           "provenance": core.worst_provenance(sa_sig.get("provenance"),
                                              sb_sig.get("provenance"))}
    if sa is None or sb is None or not fps:
        return core.response(spec, sig, fps, n, fr, w + wb + ["no_data"], None, "No data.")
    length = min(len(sa), len(sb))
    sa, sb = sa[:length], sb[:length]

    def phase(a):
        v = np.gradient(a, 1.0 / fps)
        an = 2 * (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a)) - 1
        vn = v / np.nanmax(np.abs(v)) if np.nanmax(np.abs(v)) else v
        return np.degrees(np.arctan2(vn, an))

    diff = (phase(sa) - phase(sb) + 180) % 360 - 180
    finite = diff[np.isfinite(diff)]
    result = None if finite.size == 0 else {
        "mean_phase_deg": float(finite.mean()), "std_phase_deg": float(finite.std())}
    return core.response(spec, sig, fps, n, fr, w + wb, result,
                         "CRP = phase(a)-phase(b), atan2(v_norm, angle_norm).")


@mcp.tool()
def list_cycles(session_key: str):
    """Cicli della sessione, con il metodo che li ha prodotti."""
    with core.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT c.cycle_index, c.start_frame, c.end_frame, c.method, c.side, "
            "       c.warnings "
            "FROM cycles c JOIN sessions s ON s.id = c.session_id "
            "WHERE s.session_key = %s ORDER BY c.cycle_index", (session_key,))
        righe = cur.fetchall()
        rows = [{"cycle_index": r[0], "start_frame": r[1], "end_frame": r[2],
                 "method": r[3], "side": r[4]} for r in righe]
        avvisi = list(righe[0][5] or []) if righe else []
    method = rows[0]["method"] if rows else None
    avvisi = _avvisi_cicli(avvisi, method)
    spiega = ("Nessun ciclo individuato: la prova intera e' trattata come gesto unico."
              if method == "whole_trial"
              else "Segmentazione deterministica scritta nel C3D canonico in conversione.")
    return {"input_spec": {"session_key": session_key},
            "selected_data": {"n": len(rows), "method": method},
            "result": {"cycles": rows},
            "explain": spiega,
            "warnings": avvisi}


@mcp.tool()
def gesture_detection(session_key: str, joint: str, side: str | None = None, plane: str = "sagittal",
                      threshold: float = 0.0, direction: str = "above", min_frames: int = 1):
    """Segmenti in cui l'angolo supera/scende sotto una soglia per almeno min_frames."""
    spec = {"session_key": session_key, "joint": joint, "side": side, "plane": plane,
            "threshold": threshold, "direction": direction, "min_frames": min_frames}
    sig = _signal(joint, plane, side)
    if not sig:
        return _no_signal(spec)
    fps, series, mask, n, fr, warns = _load(session_key, sig)
    if series is None:
        return core.response(spec, sig, fps, n, fr, warns, None, "No data.")
    below = direction.lower() in ("below", "extension", "negative")
    cond = (series <= threshold if below else series >= threshold) & np.isfinite(series)
    segs, start = [], None
    for i, flag in enumerate(list(cond) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_frames:
                segs.append({"start_frame": start, "end_frame": i, "duration_frames": i - start})
            start = None
    return core.response(spec, sig, fps, n, fr, warns,
                         {"count": len(segs), "segments": segs}, "Attraversamenti di soglia.")


# --- Strumenti: ricerca e descrizione delle sessioni ---
# Quali sessioni esistono, come si risolve un nome parziale e cosa contiene una
# sessione.
@mcp.tool()
def list_sessions(sport: str | None = None, subject_id: str | None = None, limit: int | None = None):
    """Elenca le sessioni (session_key, subject_id pseudonimo, sport, data).

    La ricerca per soggetto usa lo pseudonimo: il server non ha accesso
    all'identita' reale.
    """
    vincoli, params = [], []
    if sport:
        vincoli.append("sport = %s")
        params.append(sport)
    if subject_id:
        vincoli.append("subject_id = %s")
        params.append(subject_id)
    dove = " AND ".join(vincoli) if vincoli else "TRUE"
    limite = ""
    if limit:
        limite = " LIMIT %s"
        params.append(limit)
    q = (f"SELECT session_key, subject_id, sport, ingested_at FROM sessions"
         f" WHERE {dove} ORDER BY ingested_at DESC{limite}")
    with core.connect() as conn, conn.cursor() as cur:
        cur.execute(q, params)
        rows = [{"session_key": r[0], "subject_id": r[1], "sport": r[2],
                 "ingested_at": str(r[3])}
                for r in cur.fetchall()]
    return {"input_spec": {"sport": sport, "subject_id": subject_id, "limit": limit},
            "selected_data": {"n": len(rows)}, "result": {"sessions": rows},
            "explain": "Elenco sessioni dalla tabella sessions.", "warnings": []}


@mcp.tool()
def resolve_session(query: str):
    """Match parziale sul session_key; se ambiguo restituisce i candidati."""
    with core.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT session_key FROM sessions WHERE session_key ILIKE %s "
            "ORDER BY ingested_at DESC",
                    (f"%{query}%",))
        cand = [r[0] for r in cur.fetchall()]
    result = {"match": cand[0]} if len(cand) == 1 else {"candidates": cand}
    return {"input_spec": {"query": query}, "selected_data": {"n": len(cand)},
            "result": result, "explain": "Ricerca ILIKE sul session_key.",
            "warnings": [] if cand else ["no_match"]}


@mcp.tool()
def describe_session(session_key: str):
    """Giunti e piani interrogabili, con l'origine misurata o stimata di ciascuno."""
    with core.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT a.joint, a.plane, a.provenance FROM joint_angles a "
            "JOIN sessions s ON s.id = a.session_id WHERE s.session_key = %s "
            "ORDER BY a.joint, a.plane", (session_key,))
        rows = cur.fetchall()
        cur.execute(
            "SELECT count(*), min(c.method), min(c.warnings) "
            "FROM cycles c JOIN sessions s ON s.id = c.session_id "
            "WHERE s.session_key = %s", (session_key,))
        n_cycles, method, avvisi_cicli = cur.fetchone()
    joints = {}
    for joint, plane, provenance in rows:
        joints.setdefault(joint, {})[plane] = provenance
    return {"input_spec": {"session_key": session_key},
            "selected_data": {"n_signals": len(rows), "n_cycles": n_cycles,
                              "cycle_method": method},
            "result": {"joints": joints},
            "explain": "Segnali presenti in joint_angles, con origine per piano.",
            "warnings": ([] if joints else ["no_signals"])
                        + (_avvisi_cicli(avvisi_cicli, method) if joints else [])}


def _avvisi_cicli(memorizzati, metodo):
    return list(memorizzati or []) or [a for a in (segmentation.avviso_per_metodo(metodo),) if a]


# --- Strumenti: confronti di forma calcolati da pgvector ---
# Cicli simili a uno dato, distanza fra due sessioni, ripetibilita' del gesto e
# somiglianza fra i due lati.
@mcp.tool()
def find_similar_cycles(session_key: str, cycle_index: int = 0,
                        scope: str = "lower_body", planes: list[str] | None = None,
                        top_k: int = 5, min_atoms: int = 2,
                        same_sport_only: bool = True):
    """Cicli di ALTRE sessioni con l'andamento piu' simile a quello dato.

    Il confronto e' sulla forma della curva, non sui valori: gli angoli portano
    scostamenti propri del soggetto e del modello, quindi confrontare i livelli
    fra persone diverse misurerebbe la differenza fra i modelli. La distanza e'
    quella coseno di pgvector: 0 stesso andamento, 1 nessuna relazione.
    """
    planes = planes or ["sagittal"]
    spec = {"session_key": session_key, "cycle_index": cycle_index, "scope": scope,
            "planes": planes, "top_k": top_k, "min_atoms": min_atoms,
            "same_sport_only": same_sport_only}
    with core.connect() as conn, conn.cursor() as cur:
        out = similarity.similar_cycles(cur, session_key, cycle_index, scope,
                                        planes, top_k, min_atoms, same_sport_only)
    righe = out["cycles"]
    return core.response(
        spec, {"scope": out.get("scope"), "planes": planes}, None, None, None,
        out["warnings"] + ([] if righe else ["nessun_candidato"]),
        {"neighbors": righe,
         "sessioni_confrontate": out.get("sessioni_confrontate"),
         "sessioni_compatibili": out.get("sessioni_compatibili")},
        "avg(shape <=> shape_query) sugli atomi in comune, calcolata da pgvector; "
        "gli atomi hanno norma unitaria, quindi la media equivale alla distanza "
        "dei vettori concatenati.")


@mcp.tool()
def compare_session_shapes(session_key_a: str, session_key_b: str,
                           scope: str = "lower_body",
                           planes: list[str] | None = None):
    """Quanto si somigliano DUE sessioni, giunto per giunto.

    Per ogni giunto riporta la distanza mediana fra tutte le coppie di cicli, e
    le escursioni medie dei due lati del confronto: la forma dice *come* si
    muove, l'escursione *quanto*, e le due cose vanno lette insieme.
    """
    planes = planes or ["sagittal"]
    spec = {"session_key_a": session_key_a, "session_key_b": session_key_b,
            "scope": scope, "planes": planes}
    with core.connect() as conn, conn.cursor() as cur:
        out = similarity.compare_sessions(cur, session_key_a, session_key_b,
                                          scope, planes)
    return core.response(
        spec, {"scope": out.get("scope"), "planes": planes}, None, None, None,
        out["warnings"], {"joints": out["joints"]},
        "percentile_cont(0.5) su (shape_a <=> shape_b) per ogni coppia di cicli.")


@mcp.tool()
def session_repeatability(session_key: str, scope: str = "lower_body",
                          planes: list[str] | None = None):
    """Quanto il gesto si ripete uguale DENTRO la stessa sessione.

    Distanza mediana fra tutte le coppie di cicli distinti. Un valore alto puo'
    voler dire variabilita' vera del gesto o tracciamento instabile: la copertura
    minima riportata accanto serve proprio a distinguere i due casi.
    """
    planes = planes or ["sagittal"]
    spec = {"session_key": session_key, "scope": scope, "planes": planes}
    with core.connect() as conn, conn.cursor() as cur:
        out = similarity.repeatability(cur, session_key, scope, planes)
    return core.response(
        spec, {"scope": out.get("scope"), "planes": planes}, None, None, None,
        out["warnings"], {"joints": out["joints"]},
        "percentile_cont(0.5) su (shape_i <=> shape_j) per ogni coppia di cicli "
        "della stessa sessione.")


@mcp.tool()
def side_shape_symmetry(session_key: str, planes: list[str] | None = None):
    """Confronto fra lato destro e sinistro DENTRO la stessa sessione.

    Va letto con cautela: nel cammino e nella corsa i due arti sono in
    controfase, quindi una distanza alta descrive la sfasatura temporale e non
    un'asimmetria. Il dato utile qui e' soprattutto la differenza di escursione,
    riportata accanto alla distanza.
    """
    planes = planes or ["sagittal"]
    spec = {"session_key": session_key, "planes": planes}
    with core.connect() as conn, conn.cursor() as cur:
        out = similarity.symmetry(cur, session_key, planes)
    return core.response(
        spec, {"planes": planes}, None, None, None, out["warnings"],
        {"pairs": out["pairs"]},
        "(shape_sinistra <=> shape_destra) sullo stesso ciclo, piu' la "
        "differenza di escursione.")


# --- Strumento: alert ---
# Gli alert registrati durante il caricamento della sessione.
@mcp.tool()
def list_alerts(session_key: str, severity: str | None = None):
    """Scostamenti dai riferimenti di letteratura, per una sessione.

    Gli alert non si calcolano qui: li produce il caricamento, confrontando le
    metriche della sessione con i valori fissi in `reference_ranges`. Il server
    e' di sola lettura, quindi puo' solo riferirli -- ed e' giusto cosi': un
    giudizio che cambia a seconda di chi lo chiede non e' un giudizio.

    Ogni riga porta la fonte da cui viene l'intervallo, e la copertura e la
    provenienza dell'angolo che lo ha generato: un allarme su una serie modellata
    al 40% di copertura non vale quanto uno su una serie misurata al 100%.
    """
    spec = {"session_key": session_key, "severity": severity}
    q = ("SELECT joint, plane, metric, observed, expected_min, expected_max, "
         "deviation, severity, coverage, provenance, scope, source, note "
         "FROM session_alerts WHERE session_key = %s")
    params = [session_key]
    if severity:
        q += " AND severity = %s"
        params.append(severity)
    q += (" ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'warning' THEN 1 "
          "WHEN 'info' THEN 2 ELSE 3 END, deviation DESC")
    with core.connect() as conn, conn.cursor() as cur:
        cur.execute(q, params)
        colonne = [d[0] for d in cur.description]
        righe = [dict(zip(colonne, r)) for r in cur.fetchall()]
    for r in righe:
        for campo in ("note", "source"):
            if campo in r:
                r[campo] = core.testo_citato(r[campo])
    conteggio = {}
    for r in righe:
        conteggio[r["severity"]] = conteggio.get(r["severity"], 0) + 1
    return core.response(
        spec, None, None, None, None,
        [] if righe else ["nessuno_scostamento"],
        {"alerts": righe, "by_severity": conteggio},
        "Confronto fra le metriche della sessione e reference_ranges, "
        "eseguito al caricamento.")


# --- Strumento: rendering ---
# Apre il viewer sul C3D canonico della sessione.
@mcp.tool()
def visualize_session(session_key: str, show_coord_frames: bool = False,
                      show_alerts: bool = True, show_arrows: bool = True):
    """Apre il viewer Open3D (finestra nativa locale) come processo separato.

    show_alerts colora di rosso i giunti fuori dai riferimenti di letteratura,
    nei soli fotogrammi in cui lo sono; show_arrows aggiunge una freccia su
    ciascuno. show_arrows senza show_alerts non ha effetto.
    """
    spec = {"session_key": session_key, "show_coord_frames": show_coord_frames,
            "show_alerts": show_alerts, "show_arrows": show_arrows}
    if VIEWER_DISABLED:
        return {"input_spec": spec, "selected_data": {}, "result": None,
                "explain": "Viewer disattivato su questa istanza (BIOMECH_VIEWER_DISABLED): "
                           "non apre finestre sulla macchina che ospita il server.",
                "warnings": ["viewer_disabled"]}
    c3d, errore = _c3d_sessione(spec, session_key)
    if errore:
        return errore
    non_pronto = _viewer_pronto(spec)
    if non_pronto:
        return non_pronto
    cmd = [VIEWER_PY, "-m", VIEWER_MODULE, "--c3d", str(c3d)]
    if show_coord_frames:
        cmd.append("--coord")
    if not show_alerts:
        cmd.append("--no-alert")
    elif not show_arrows:
        cmd.append("--no-frecce")
    log = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL,
                                stderr=log, start_new_session=True)
    except OSError as e:
        log.close()
        return {"input_spec": spec, "selected_data": {"c3d": str(c3d)}, "result": None,
                "explain": f"Impossibile avviare il viewer: {e}", "warnings": ["viewer_failed"]}
    try:
        if proc.wait(timeout=1.5) != 0:
            log.seek(0)
            errore = log.read().decode("utf-8", "replace").strip()
            log.close()
            return {"input_spec": spec, "selected_data": {"c3d": str(c3d)}, "result": None,
                    "explain": f"Il viewer e' uscito subito: {errore.splitlines()[-1] if errore else 'nessun dettaglio'}",
                    "warnings": ["viewer_failed"]}
        log.close()
    except subprocess.TimeoutExpired:
        log.close()
    return {"input_spec": spec, "selected_data": {"c3d": str(c3d)},
            "result": {"launched": True, "pid": proc.pid},
            "explain": "Viewer Open3D lanciato come processo separato.", "warnings": []}


# --- Scrittura dalla chat: supporto ---
# La connessione del server resta di sola lettura; questi strumenti scrivono
# lanciando process e ingest come sottoprocessi, con la DSN di scrittura
# nell'ambiente e mai negli argomenti -- lo stesso schema di `src/ui/server.py`.
# Tre funzioni fanno il lavoro: risoluzione della sorgente contro le radici
# ammesse, il sottoprocesso con la sua gestione di errore e timeout, e il log
# su file che non passa dal modello.
def _ingest_disabilitato(spec):
    return {"input_spec": spec, "selected_data": {}, "result": None,
            "explain": "Scrittura disattivata su questa istanza (manca "
                       "BIOMECH_INGEST_ENABLED): il server resta di sola lettura. "
                       "Usa la UI locale (src.ui.server) o la riga di comando.",
            "warnings": ["ingest_disabled"]}


def _radici_ingest():
    radici = []
    for r in INGEST_ROOTS:
        p = Path(r)
        radici.append(p if p.is_absolute() else REPO_ROOT / p)
    return radici


def _sorgenti_disponibili():
    viste = {}
    for radice in _radici_ingest():
        for percorso in readers.discover(str(radice)):
            nome = os.path.splitext(os.path.basename(percorso.rstrip("/\\")))[0]
            viste.setdefault(nome, percorso)
    return viste


def _risolvi_sorgente(source):
    if not source:
        return None, "source mancante."
    disponibili = _sorgenti_disponibili()
    if source in disponibili:
        return disponibili[source], None
    candidato = Path(source)
    if not candidato.is_absolute():
        candidato = REPO_ROOT / candidato
    try:
        candidato = candidato.resolve()
    except OSError:
        return None, f"'{source}' non e' un percorso leggibile."
    for radice in _radici_ingest():
        try:
            candidato.relative_to(radice.resolve())
        except ValueError:
            continue
        if candidato.exists():
            return str(candidato), None
        return None, f"'{source}' e' sotto una radice ammessa ma non esiste."
    return None, (f"'{source}' non e' fra le sorgenti di list_sources() ne' un "
                  f"percorso sotto le radici ammesse "
                  f"({', '.join(str(r) for r in _radici_ingest())}).")


def _chiavi_in_db(nomi):
    nomi = list(nomi)
    if not nomi:
        return set()
    with core.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT session_key FROM sessions WHERE session_key = ANY(%s)",
                    (nomi,))
        return {r[0] for r in cur.fetchall()}


def _ricevuta_sessione(nome):
    with core.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT n_frames, ingested_at FROM sessions "
                    "WHERE session_key = %s", (nome,))
        riga = cur.fetchone()
    return None if not riga else {"n_frames": riga[0], "ingested_at": riga[1]}


def _sport_del_canonico(c3d):
    try:
        dati = canonical.read(str(c3d))
    except Exception:
        return None
    return bodyparts.canonical_sport(dati["meta"].get("sport"))


def _passo(cmd, etichetta, extra_env=None):
    ambiente = dict(os.environ, **(extra_env or {}))
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                              text=True, timeout=INGEST_TIMEOUT, env=ambiente)
    except subprocess.TimeoutExpired:
        return False, "", f"{etichetta}: oltre il tempo massimo ({INGEST_TIMEOUT} s)"
    if proc.returncode == 0:
        return True, proc.stdout, None
    if proc.returncode < 0:
        return False, proc.stdout, (
            f"{etichetta}: terminata dal segnale {-proc.returncode} "
            "(file probabilmente malformato)")
    errore = (proc.stderr.strip().splitlines() or ["errore sconosciuto"])[-1]
    return False, proc.stdout, f"{etichetta}: {errore}"


def _passo_conversione(sorgente, sport):
    cmd = [sys.executable, "-m", "src.conversion.process", sorgente,
           "--uscita", str(DATA_ROOT)]
    if sport:
        cmd += ["--sport", sport]
    return _passo(cmd, "conversione")


def _passo_caricamento(c3d, subject_id):
    cmd = [sys.executable, "-m", "src.database.ingest", str(c3d)]
    if subject_id:
        cmd += ["--subject-id", subject_id]
    return _passo(cmd, "caricamento", {"INGEST_DB_DSN": INGEST_DB_DSN or ""})


def _registra(nome, sport, subject_id, passo, ok, errore):
    try:
        INGEST_LOG.parent.mkdir(parents=True, exist_ok=True)
        ultima = (errore or "").strip().splitlines()[-1] if errore else "-"
        riga = "\t".join([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            nome, sport or "-", subject_id or "-", passo,
            "ok" if ok else "fallito", ultima])
        with open(INGEST_LOG, "a", encoding="utf-8") as f:
            f.write(riga + "\n")
    except OSError:
        pass  # il log e' un aiuto, non deve far fallire lo strumento su un disco pieno


def _suggerimento(passo, errore):
    if passo == "conversione" and errore and "gesto non riconosciuto" in errore:
        return f"Rilancia con sport fra {', '.join(bodyparts.SPORTS)}."
    if passo == "caricamento" and errore and "nome di sessione non ammesso" in errore:
        return "Rinomina il file sorgente con un nome semplice e riconverti."
    return "Controlla l'errore riportato; il tentativo resta in DatiElaborati/ingest.log."


def _fallimento(spec, nome, passo, errore, esplicito):
    return {"input_spec": spec,
            "selected_data": {"sorgente": nome, "passo_fallito": passo,
                              "esito": "fallito"},
            "result": None, "errore": errore, "explain": esplicito,
            "azione_suggerita": _suggerimento(passo, errore),
            "warnings": [f"{passo}_fallita"]}


# --- Strumenti: scrittura dalla chat ---
# Convertire e caricare una sessione senza uscire dalla conversazione. La
# connessione resta di sola lettura: la scrittura vera passa dai sottoprocessi
# sopra. Spenti finche' BIOMECH_INGEST_ENABLED non e' impostata.
@mcp.tool()
def list_sources():
    """Sorgenti sotto le cartelle ammesse (BIOMECH_INGEST_ROOTS), con lo stato.

    Tre stati, non due: 'convertita' non vuol dire 'nel database'.
    da_convertire       nessun C3D canonico ancora
    convertita_non_caricata   C3D presente, nessuna riga in sessions
    nel_database         C3D presente e caricata

    Non scrive nulla; usa `ingest_session` per far avanzare una sorgente.
    """
    spec = {}
    if not INGEST_ENABLED:
        return _ingest_disabilitato(spec)
    disponibili = _sorgenti_disponibili()
    in_db = _chiavi_in_db(disponibili.keys())
    righe = []
    for nome, percorso in sorted(disponibili.items()):
        c3d = DATA_ROOT / f"{nome}.c3d"
        if nome in in_db:
            stato = "nel_database"
        elif c3d.is_file():
            stato = "convertita_non_caricata"
        else:
            stato = "da_convertire"
        righe.append({"source": nome, "percorso": percorso, "stato": stato})
    return {"input_spec": spec,
            "selected_data": {"radici": [str(r) for r in _radici_ingest()]},
            "result": {"sorgenti": righe},
            "explain": "readers.discover sulle radici ammesse; 'nel_database' letto "
                       "da sessions, 'convertita_non_caricata' dal C3D gia' presente "
                       "in DatiElaborati.",
            "warnings": [] if righe else ["nessuna_sorgente"]}


@mcp.tool()
def ingest_session(source: str, sport: str | None = None,
                   subject_id: str | None = None, only_convert: bool = False):
    """Converte e, salvo diversa richiesta, carica una sessione dalla chat.

    source e' un nome elencato da list_sources(), o un percorso sotto una
    radice ammessa (BIOMECH_INGEST_ROOTS). Se dopo la conversione la sessione
    resta senza gesto -- perche' la sorgente non lo dichiara e non ne e' stato
    passato uno -- lo strumento si ferma li' invece di indovinare: risponde
    con i gesti ammessi, ed e' cosi' che chiede di essere richiamato con
    sport=. Il C3D gia' prodotto resta a disposizione: la conversione e'
    deterministica sulla stessa sorgente, quindi il richiamo non la rifa'
    inutilmente.

    subject_id non viene verificato: biomech_mcp non vede l'anagrafica dei
    soggetti (solo la UI locale, con ADMIN_DSN, ce l'ha), quindi un
    identificativo inventato viene accettato in silenzio e la risposta lo
    dichiara.
    """
    spec = {"source": source, "sport": sport, "subject_id": subject_id,
            "only_convert": only_convert}
    if not INGEST_ENABLED:
        return _ingest_disabilitato(spec)

    sorgente, errore = _risolvi_sorgente(source)
    if errore:
        return {"input_spec": spec, "selected_data": {"sorgente": source,
                                                       "esito": "fallito"},
                "result": None, "errore": None, "explain": errore,
                "warnings": ["sorgente_non_valida"]}

    nome = os.path.splitext(os.path.basename(sorgente.rstrip("/\\")))[0]

    ok, _, err = _passo_conversione(sorgente, sport)
    _registra(nome, sport, subject_id, "conversione", ok, err)
    if not ok:
        return _fallimento(spec, nome, "conversione", err,
                           f"La conversione e' fallita: {err}")

    c3d = DATA_ROOT / f"{nome}.c3d"
    if _sport_del_canonico(c3d) is None and sport is None and not only_convert:
        msg = (f"Conversione riuscita ma nessun gesto e' stato riconosciuto: "
               f"la sorgente non lo dichiara e non ne hai passato uno. Il C3D "
               f"resta in {c3d} e verra' riusato, non riconvertito, al "
               f"prossimo tentativo.")
        _registra(nome, sport, subject_id, "sport_mancante", False,
                  "sport non dichiarato")
        return {"input_spec": spec,
                "selected_data": {"sorgente": nome, "passo_fallito": "sport",
                                  "esito": "fallito", "c3d": str(c3d)},
                "result": None, "errore": None, "explain": msg,
                "azione_suggerita": (f"Richiama con sport fra "
                                     f"{', '.join(bodyparts.SPORTS)}."),
                "warnings": ["sport_mancante"]}

    if only_convert:
        _registra(nome, sport, subject_id, "solo_conversione", True, None)
        return {"input_spec": spec,
                "selected_data": {"sorgente": nome, "esito": "riuscito",
                                  "c3d": str(c3d)},
                "result": {"convertita": True, "caricata": False},
                "explain": "Conversione eseguita, caricamento saltato "
                           "(only_convert=True).",
                "warnings": []}

    ok, _, err = _passo_caricamento(c3d, subject_id)
    _registra(nome, sport, subject_id, "caricamento", ok, err)
    if not ok:
        return _fallimento(spec, nome, "caricamento", err,
                           f"Il caricamento e' fallito: {err}")

    ricevuta = _ricevuta_sessione(nome)
    if ricevuta is None:
        # Il sottoprocesso e' uscito senza errori ma la sessione non e' visibile
        # da qui: non ci si fida del codice di uscita, si verifica.
        msg = ("Il caricamento e' terminato senza errori ma la sessione non "
               "risulta in sessions: non dichiaro riuscito cio' che non vedo.")
        _registra(nome, sport, subject_id, "verifica", False, msg)
        return {"input_spec": spec,
                "selected_data": {"sorgente": nome, "passo_fallito": "verifica",
                                  "esito": "fallito"},
                "result": None, "errore": None, "explain": msg,
                "azione_suggerita": "Riprova; se persiste, guarda DatiElaborati/ingest.log.",
                "warnings": ["verifica_fallita"]}

    avvisi = ["sessione_senza_soggetto"] if not subject_id else ["subject_id_non_verificato"]
    return {"input_spec": spec,
            "selected_data": {"sorgente": nome, "esito": "riuscito",
                              "c3d": str(c3d), **ricevuta},
            "result": {"convertita": True, "caricata": True},
            "explain": "Convertita e caricata; la sessione e' verificata in "
                       "sessions (n_frames, ingested_at) prima di dichiarare "
                       "riuscito. biomech_mcp non vede l'anagrafica: subject_id "
                       "non e' verificato contro subjects_identity.",
            "warnings": avvisi}


if __name__ == "__main__":
    mcp.run()
