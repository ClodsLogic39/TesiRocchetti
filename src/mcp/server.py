"""Server MCP (FastMCP) per le query biomeccaniche.

Tool consolidati + discovery + similarita' (calcolata da pgvector) + alert.
Ogni tool restituisce input_spec / selected_data / result / explain / warnings.
Avvio: python -m src.mcp.server   (DSN via env MCP_DB_DSN)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

from . import core
from ..conversion import anatomy as bodyparts
from ..conversion import process
from ..conversion import segmentation
from ..database import similarity

REPO_ROOT = Path(__file__).resolve().parents[2]
# La cartella dei C3D canonici e' quella in cui scrive il convertitore: presa da
# li' e non riscritta a mano, cosi' non puo' piu' divergere dal pipeline.
DATA_ROOT = Path(os.environ.get("BIOMECH_DATA_ROOT",
                                REPO_ROOT / process.USCITA_DEFAULT))
VIEWER_PY = os.environ.get("BIOMECH_VIEWER_PYTHON", sys.executable)
VIEWER_MODULE = "src.visualization.viewer_open3d"
# Aprire una finestra nativa ha senso quando server e utente sono la stessa
# persona sulla stessa macchina. Su un server condiviso e' un processo eseguito
# sull'host per conto di un estraneo: l'interruttore permette di negarlo senza
# togliere lo strumento, che continua a rispondere spiegando perche'.
VIEWER_DISABLED = bool(os.environ.get("BIOMECH_VIEWER_DISABLED"))

mcp = FastMCP("Biomech MCP")


def _signal(joint, plane, side=None):
    """Risolve (giunto, lato, piano) o restituisce None se fuori vocabolario."""
    key = core.canonical_joint(joint, side)
    if not key or plane not in bodyparts.PLANES:
        return None
    return {"joint": key, "plane": plane}


def _load(session_key, sig, frame_start=None, frame_end=None, conn=None):
    """Serie + metadati del segnale, con provenienza e copertura per riga."""
    fps, values, provenance, coverage = core.fetch_angle(
        session_key, sig["joint"], sig["plane"], conn=conn)
    sig["provenance"] = provenance
    sig["coverage"] = coverage
    return (fps, *core.prepare_series(values, frame_start, frame_end))


def _peggior_copertura(valori):
    """La copertura di un confronto e' la peggiore fra quelle che lo compongono."""
    presenti = [v for v in valori if v is not None]
    return min(presenti) if presenti else None


def _no_signal(spec, explain="Giunto o piano fuori vocabolario."):
    return core.response(spec, None, None, 0, None, ["signal_unresolved"], None, explain)


def _c3d_sessione(spec, session_key):
    """(percorso, None) se il C3D canonico c'e', altrimenti (None, risposta).

    `session_key` finisce in un percorso: deve essere un nome semplice, o una
    chiave con `..` leggerebbe fuori dalla cartella dei dati.
    """
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
    """None se l'interprete del viewer ha le dipendenze, altrimenti la risposta."""
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
    """Risposta se il nome della metrica non esiste, altrimenti None.

    Senza questo controllo un nome inventato darebbe `result: None` esattamente
    come un segnale senza fotogrammi validi: chi legge non distinguerebbe "quella
    metrica non esiste" da "non ci sono dati", e l'explain ripeterebbe il nome
    inesistente come se fosse legittimo.
    """
    if metrica in core.METRICHE:
        return None
    return core.response(spec, None, None, 0, None, ["unknown_metric"], None,
                         f"Metrica '{metrica}' inesistente. "
                         f"Disponibili: {', '.join(core.METRICHE)}.")


# --- Metriche scalari ---
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
    # `sig` viene riusato a ogni giro e `_load` lo sovrascrive: senza queste due
    # righe provenienza e copertura resterebbero quelle dell'ultima sessione
    # letta, e l'avviso di bassa copertura dipenderebbe dall'ordine degli
    # argomenti invece che dai dati.
    sig["provenance"] = core.worst_provenance(*provenienze)
    sig["coverage"] = _peggior_copertura(coperture)
    # Confronto fra sorgenti eterogenee: il numero e' calcolabile, non equivalente.
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
    # Come in compare_sessions: senza riparare `sig`, una media costruita su
    # serie misurate e modellate uscirebbe come se fossero omogenee.
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
    # Gli avvisi arrivano dal file attraverso il database. Il ripiego sul metodo
    # copre i casi in cui non ci sono cicli -- li' non esiste una riga da cui
    # leggerli -- e i file convertiti prima che la colonna esistesse.
    avvisi = _avvisi_cicli(avvisi, method)
    spiega = ("Nessun ciclo individuato: la prova intera e' trattata come gesto unico."
              if method == "whole_trial"
              # I confini stanno scritti nel C3D canonico, decisi in conversione:
              # l'ingest li copia e basta. Dirlo importa, perche' significa che
              # ricaricare lo stesso file da' sempre gli stessi cicli.
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


# --- Discovery ---
@mcp.tool()
def list_sessions(sport: str | None = None, subject_id: str | None = None, limit: int | None = None):
    """Elenca le sessioni (session_key, subject_id pseudonimo, sport, data).

    La ricerca per soggetto usa lo pseudonimo: il server non ha accesso
    all'identita' reale.
    """
    # La query si compone in un pezzo solo, con il nome della tabella accanto
    # alle colonne che cita: spezzata in frammenti, un ORDER BY su una colonna
    # scomparsa non sarebbe piu' verificabile contro lo schema.
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
            # Gli stessi avvisi di list_cycles: chi chiede la descrizione della
            # sessione deve sapere se i cicli sono veri o un ripiego.
            "warnings": ([] if joints else ["no_signals"])
                        + (_avvisi_cicli(avvisi_cicli, method) if joints else [])}


def _avvisi_cicli(memorizzati, metodo):
    """Gli avvisi scritti dall'ingest, o quelli deducibili se non ci sono.

    Il ripiego serve ai file convertiti prima che la colonna esistesse e alle
    sessioni senza cicli, dove non c'e' una riga da cui leggerli.
    """
    return list(memorizzati or []) or [a for a in (segmentation.avviso_per_metodo(metodo),) if a]


# --- Similarita' (distanza calcolata da pgvector) ---
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
        # Quante sessioni sono entrate nel confronto. Da quando il tetto sulle
        # candidate non c'e' piu' i due numeri coincidono sempre: restano
        # entrambi perche' dicono su quale base e' stata data la risposta, e
        # perche' se un giorno tornasse un filtro la differenza si vedrebbe.
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
    # `data_quality` va in fondo: non e' un reperto sul gesto ma un avviso che
    # quella misura non e' credibile, e mescolarlo ai reperti clinici toglierebbe
    # peso a entrambi.
    q += (" ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'warning' THEN 1 "
          "WHEN 'info' THEN 2 ELSE 3 END, deviation DESC")
    with core.connect() as conn, conn.cursor() as cur:
        cur.execute(q, params)
        colonne = [d[0] for d in cur.description]
        righe = [dict(zip(colonne, r)) for r in cur.fetchall()]
    # `note` e `source` sono testo libero di un file di riferimenti: vanno
    # all'agente come dato citato, non come blocco che possa somigliare a
    # un'istruzione.
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


# --- Rendering ---
@mcp.tool()
def visualize_session(session_key: str, show_coord_frames: bool = False):
    """Apre il viewer Open3D (finestra nativa locale) come processo separato."""
    spec = {"session_key": session_key, "show_coord_frames": show_coord_frames}
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
    # Con `-m` dalla radice del repo: il viewer importa `anatomy` e `canonical`
    # come parte del pacchetto, senza ritocchi a sys.path.
    cmd = [VIEWER_PY, "-m", VIEWER_MODULE, "--c3d", str(c3d)]
    if show_coord_frames:
        cmd.append("--coord")
    # Lo stderr va su un file temporaneo, non su una pipe: il viewer resta vivo
    # per tutta la sessione e nessuno leggerebbe la pipe, cosi' al primo blocco
    # di avvisi di Open3D il buffer si riempirebbe e la finestra si fermerebbe a
    # meta' rendering. Un file non ha quel limite.
    log = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL,
                                stderr=log, start_new_session=True)
    except OSError as e:
        log.close()
        return {"input_spec": spec, "selected_data": {"c3d": str(c3d)}, "result": None,
                "explain": f"Impossibile avviare il viewer: {e}", "warnings": ["viewer_failed"]}
    # Una finestra grafica resta viva: se il processo e' gia' morto entro la
    # breve attesa, non e' partito e va detto, invece di annunciare un successo
    # che l'utente non vedra' mai.
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
        # Ancora vivo: la finestra e' aperta. Il figlio tiene il suo descrittore
        # del file, quindi qui si puo' chiudere il nostro senza troncare nulla.
        log.close()
    return {"input_spec": spec, "selected_data": {"c3d": str(c3d)},
            "result": {"launched": True, "pid": proc.pid},
            "explain": "Viewer Open3D lanciato come processo separato.", "warnings": []}


# `render_frame` (rendering statico su PNG, senza finestra) e' stato rimosso
# per ora: scriveva nella cartella temporanea di sistema della macchina che
# ospita il server, un percorso che l'agente non ha modo di leggere. Va
# reintrodotto quando avra' una destinazione condivisa (o restituira' i byte
# PNG direttamente nella risposta invece di un percorso su disco).


if __name__ == "__main__":
    mcp.run()
