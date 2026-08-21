"""Segmentazione della prova in cicli, con metodo scelto in base al gesto.

Ogni gesto si ripete per motivi diversi, quindi a ogni sport corrisponde un
elenco ordinato di metodi: si usa il primo il cui segnale e' presente e
abbastanza coperto. Senza sport dichiarato valgono i metodi generici; se nessun
segnale risulta ciclico la prova resta un unico segmento.

Ogni metodo restituisce gli intervalli di ciclo e il lato a cui ciascuno
appartiene. Il nome del metodo usato viaggia insieme ai cicli, insieme agli
avvisi su cio' che e' stato concesso per ottenerli.

Prerequisiti: numpy e scipy. In ingresso il risultato di `angles.compute`, con
angoli, punti e metadati della sessione.
"""
import numpy as np

from . import anatomy
from scipy.signal import find_peaks

# --- Soglie ---
# Durata minima di un ciclo, copertura minima del segnale su cui un metodo
# lavora, escursione minima perche' un angolo o una posizione siano considerati
# ciclici e prominenza richiesta ai picchi, in frazione dell'escursione.
MIN_PERIOD_S = 0.3
MIN_INPUT_COVERAGE = 0.50
ANGLE_FLOOR_DEG = 5.0
POSITION_FLOOR_M = 0.05
PROMINENCE_FRAC = 0.3


# --- Utilita' di segnale ---
# Escursione di una serie, tratti validi abbastanza lunghi, ricerca dei picchi e
# formazione degli intervalli fra picchi successivi.
def _span(series, window=None):
    arr = np.asarray(series, dtype=float)
    if not window or arr.size <= window:
        return float(np.percentile(arr, 95) - np.percentile(arr, 5))
    passo = max(1, window // 2)
    locali = [np.ptp(arr[i:i + window]) for i in range(0, arr.size - window, passo)]
    return float(np.median(locali)) if locali else 0.0


def _valid_runs(valid, minimo):
    out, i, n = [], 0, valid.shape[0]
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and valid[j + 1]:
            j += 1
        if j + 1 - i >= minimo:
            out.append((i, j + 1))
        i = j + 1
    return out


def _peaks(series, min_dist, floor, minimo=2):
    arr = np.asarray(series, dtype=float)
    gruppi = []
    for s, e in _valid_runs(np.isfinite(arr), 2 * min_dist):
        seg = arr[s:e]
        span = _span(seg, window=2 * min_dist)
        if span < floor:
            continue
        pk, _ = find_peaks(seg, distance=min_dist, prominence=PROMINENCE_FRAC * span)
        if len(pk) >= minimo:
            gruppi.append([int(s + p) for p in pk])
    return gruppi


def _cycles(gruppi, min_dist):
    out = [(a, b) for ev in (gruppi or []) for a, b in zip(ev, ev[1:]) if b - a >= min_dist]
    return out or None


# --- Metodi di segmentazione ---
# Ogni metodo restituisce `(cicli, lati)`, con un lato per ciascun ciclo: eventi
# gia' presenti nella sorgente, coordinata della punta del piede, flessione di un
# giunto, periodicita' dominante e prova intera come unico ciclo.
def _da_entrambi_i_lati(per_lato):
    coppie = [(c, lato) for lato, cicli in per_lato for c in cicli]
    if not coppie:
        return None
    coppie.sort(key=lambda x: (x[0][0], x[0][1]))
    return [c for c, _ in coppie], [lato for _, lato in coppie]


def _source_events(ctx):
    eventi = ctx.get("events") or {}
    per_lato = []
    for lato in ("r", "l"):
        colpi = sorted(f for f in eventi.get(lato, [])
                       if 0 <= f < ctx["n_frames"])
        cicli = _cycles([colpi], ctx["min_dist"])
        if cicli:
            per_lato.append((lato, cicli))
    return _da_entrambi_i_lati(per_lato)


def _toe_coordinate(ctx):
    pts = ctx["landmarks"]
    if not all(k in pts for k in ("hip_l", "hip_r")):
        return None
    center = (np.asarray(pts["hip_l"]) + np.asarray(pts["hip_r"])) / 2.0
    avanzamento = {}
    for side in ("r", "l"):
        toe = pts.get(f"toe_{side}")
        if toe is not None:
            avanzamento[side] = (np.asarray(toe) - center) @ ctx["ap"]

    per_lato = []
    for side in ("r", "l"):
        if side not in avanzamento:
            continue
        cycles = _cycles(_peaks(avanzamento[side], ctx["min_dist"], POSITION_FLOOR_M),
                         ctx["min_dist"])
        if cycles:
            per_lato.append((side, cycles))
    return _da_entrambi_i_lati(per_lato)


def _toe_coordinate_step(ctx):
    pts = ctx["landmarks"]
    if not all(k in pts for k in ("hip_l", "hip_r")):
        return None
    center = (np.asarray(pts["hip_l"]) + np.asarray(pts["hip_r"])) / 2.0
    mezzo = max(1, ctx["min_dist"] // 2)

    eventi = []
    for side in ("r", "l"):
        toe = pts.get(f"toe_{side}")
        if toe is None:
            continue
        serie = (np.asarray(toe) - center) @ ctx["ap"]
        for gruppo in _peaks(serie, mezzo, POSITION_FLOOR_M, minimo=1):
            eventi += gruppo
    if len(eventi) < 2:
        return None
    cycles = _cycles([sorted(set(eventi))], mezzo)
    return (cycles, [None] * len(cycles)) if cycles else None


def _joint_flexion(ctx, joint):
    per_lato = []
    for side in ("r", "l"):
        axes = ctx["joints"].get(f"{joint}_{side}")
        if not axes:
            continue
        series = axes.get("sagittal")
        if series is None:
            continue
        cycles = _cycles(_peaks(series, ctx["min_dist"], ANGLE_FLOOR_DEG), ctx["min_dist"])
        if cycles:
            per_lato.append((side, cycles))
    return _da_entrambi_i_lati(per_lato)


def _dominant_periodicity(ctx):
    best = None
    for joint, axes in ctx["joints"].items():
        for values in axes.values():
            arr = np.asarray(values, dtype=float)
            finiti = arr[np.isfinite(arr)]
            if finiti.size < 2 * ctx["min_dist"]:
                continue
            span = _span(finiti, window=2 * ctx["min_dist"])
            if span >= ANGLE_FLOOR_DEG and (best is None or span > best[0]):
                best = (span, joint, arr)
    if best is None:
        return None
    cycles = _cycles(_peaks(best[2], ctx["min_dist"], ANGLE_FLOOR_DEG), ctx["min_dist"])
    return (cycles, [None] * len(cycles)) if cycles else None


def _whole_trial(ctx):
    return ([(0, ctx["n_frames"])], [None]) if ctx["n_frames"] > 0 else None


# --- Registro dei metodi e strategie per gesto ---
# I metodi disponibili per nome, l'ordine in cui si provano per ogni sport, la
# sequenza generica usata senza sport dichiarato e i metodi che non individuano
# un evento fisico del gesto.
METHODS = {
    "source_events": _source_events,
    "toe_coordinate": _toe_coordinate,
    "toe_coordinate_step": _toe_coordinate_step,
    "hip_flexion": lambda ctx: _joint_flexion(ctx, "hip"),
    "knee_flexion": lambda ctx: _joint_flexion(ctx, "knee"),
    "shoulder_flexion": lambda ctx: _joint_flexion(ctx, "shoulder"),
    "dominant_periodicity": _dominant_periodicity,
    "whole_trial": _whole_trial,
}

STRATEGIES = {
    "walking": ("source_events", "toe_coordinate", "toe_coordinate_step",
                "hip_flexion", "shoulder_flexion"),
    "running": ("source_events", "toe_coordinate", "toe_coordinate_step",
                "hip_flexion", "shoulder_flexion"),
    "cycling": ("knee_flexion", "hip_flexion"),
    "swimming": ("shoulder_flexion", "hip_flexion"),
}
SPORTS = tuple(STRATEGIES)
GENERIC = ("source_events", "dominant_periodicity")

_SURROGATE = {"hip_flexion", "knee_flexion", "shoulder_flexion",
              "dominant_periodicity"}


# --- Copertura degli ingressi ---
# Quanto e' visibile il segnale da cui dipende ciascun metodo: INPUT_COVERAGE
# associa a ogni metodo il segnale da controllare prima di sceglierlo.
def _copertura(serie):
    if serie is None:
        return 0.0
    a = np.asarray(serie, dtype=float)
    if a.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(a) if a.ndim == 1 else np.isfinite(a).all(axis=1)))


def _copertura_piede(ctx):
    pts = ctx["landmarks"]
    bacino = min(_copertura(pts.get("hip_l")), _copertura(pts.get("hip_r")))
    return min(bacino, max(_copertura(pts.get("toe_l")), _copertura(pts.get("toe_r"))))


def _copertura_giunto(ctx, giunto):
    return max(_copertura((ctx["joints"].get(f"{giunto}_{lato}") or {}).get("sagittal"))
               for lato in ("r", "l"))


INPUT_COVERAGE = {
    "toe_coordinate": _copertura_piede,
    "toe_coordinate_step": _copertura_piede,
    "hip_flexion": lambda ctx: _copertura_giunto(ctx, "hip"),
    "knee_flexion": lambda ctx: _copertura_giunto(ctx, "knee"),
    "shoulder_flexion": lambda ctx: _copertura_giunto(ctx, "shoulder"),
}


# --- Scelta del metodo e ingresso unico ---
# Preparazione dei segnali disponibili, avviso associato al metodo scelto e
# `detect`, che percorre la strategia del gesto e restituisce cicli, lati, metodo
# e avvisi.
def _joints_by_plane(angles):
    out = {}
    for (giunto, piano), serie in (angles or {}).items():
        out.setdefault(giunto, {})[piano] = serie
    return out


def avviso_per_metodo(metodo):
    if not metodo:
        return "no_cycles"
    if metodo == "whole_trial":
        return "single_gesture"
    if metodo == "toe_coordinate_step":
        return "cycles_from_step"
    if metodo in _SURROGATE:
        return "cycles_from_surrogate"
    return None


def detect(result):
    meta = result.get("meta") or {}
    fps = meta.get("fps") or 0
    sport = anatomy.canonical_sport(meta.get("sport"))
    ctx = {"landmarks": result.get("points") or {},
           "events": result.get("events") or {},
           "joints": _joints_by_plane(result.get("angles")),
           "ap": np.asarray(result.get("world_frame") or np.eye(3))[0],
           "n_frames": int(meta.get("n_frames") or 0),
           "min_dist": max(1, int(round(MIN_PERIOD_S * fps))) if fps else 1}

    warnings = []
    order = STRATEGIES.get(sport)
    if order is None:
        order = GENERIC
        warnings.append("sport_without_strategy")

    scartati = []
    for name in order + ("whole_trial",):
        copertura = INPUT_COVERAGE.get(name)
        if copertura is not None and copertura(ctx) < MIN_INPUT_COVERAGE:
            scartati.append(name)
            continue
        found = METHODS[name](ctx)
        if not found:
            continue
        cycles, sides = found
        avviso = avviso_per_metodo(name)
        if avviso:
            warnings.append(avviso)
        if scartati:
            warnings.append("primary_signal_low_coverage")
        return {"method": name,
                "frames": [[int(s), int(e)] for s, e in cycles],
                "sides": list(sides),
                "warnings": warnings}
    return {"method": None, "sides": [], "frames": [],
            "warnings": warnings + [avviso_per_metodo(None)]
                     + (["primary_signal_low_coverage"] if scartati else [])}
