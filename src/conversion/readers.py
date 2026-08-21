"""Lettori delle sorgenti: dalle etichette originali ai sedici punti canonici.

Ogni lettore restituisce la stessa struttura, cosi' il resto della catena non
dipende dal formato di partenza:

    {"points":  {nome: array (n,3) in metri, NaN dove il punto non e' visto},
     "angles":  {(giunto, piano): array (n,) in gradi}   se la sorgente li ha,
     "absent":  {nome: (motivo, dettaglio)},
     "meta":    {fps, n_frames, source, source_format, ...}}

Un punto assente resta assente: non viene mai sostituito con zero.

Prerequisiti: numpy; ezc3d per le sorgenti .c3d. Una cartella running_* deve
contenere il file `segments_frames.json` prodotto dal tracker MediaPipe, con i
world landmarks di ogni fotogramma.
"""
import json
import os
import re

import numpy as np

from . import anatomy

# --- Vocabolari delle sorgenti ---
# Corrispondenza fra gli indici dei punti MediaPipe e i nomi canonici, etichette
# usate dalle convenzioni C3D, punti che si ottengono come media di due etichette
# ed espressioni che riconoscono gli angoli gia' calcolati dai modelli clinici.
_MEDIAPIPE_INDEX = {
    "ear_l": 7, "ear_r": 8,
    "shoulder_l": 11, "shoulder_r": 12,
    "elbow_l": 13, "elbow_r": 14,
    "wrist_l": 15, "wrist_r": 16,
    "hip_l": 23, "hip_r": 24,
    "knee_l": 25, "knee_r": 26,
    "ankle_l": 27, "ankle_r": 28,
    "heel_l": 29, "heel_r": 30,
    "toe_l": 31, "toe_r": 32,
}
_NOSE = 0
MIN_VISIBILITY = 0.5

_C3D_LABELS = {
    "LSHO": "shoulder_l", "RSHO": "shoulder_r",
    "LELB": "elbow_l", "RELB": "elbow_r",
    "LWRA": "wrist_l", "RWRA": "wrist_r", "LWRI": "wrist_l", "RWRI": "wrist_r",
    "LTOE": "toe_l", "RTOE": "toe_r",
    "LHEE": "heel_l", "RHEE": "heel_r",
    "CLAV": "shoulder_c",
    "LHJC": "hip_l", "RHJC": "hip_r",
    "LKJC": "knee_l", "RKJC": "knee_r",
    "LAJC": "ankle_l", "RAJC": "ankle_r",
    "LFEO": "knee_l", "RFEO": "knee_r",
    "LTIO": "ankle_l", "RTIO": "ankle_r",
    "LTOO": "toe_l", "RTOO": "toe_r",
    "LHUO": "elbow_l", "RHUO": "elbow_r",
    "LRAO": "wrist_l", "RRAO": "wrist_r",
    "C7": "c7", "STRN": "sternum", "T10": "t10", "T8": "t10",
    "LKNE": "knee_l", "RKNE": "knee_r",
    "LANK": "ankle_l", "RANK": "ankle_r",
    "L.KNEE": "knee_l", "R.KNEE": "knee_r",
    "L.ANKLE": "ankle_l", "R.ANKLE": "ankle_r",
    "L.MT2": "toe_l", "R.MT2": "toe_r",
    "L.SHOULDER": "shoulder_l", "R.SHOULDER": "shoulder_r",
    "VHEA": "head", "VNEC": "shoulder_c",
    "VLSH": "shoulder_l", "VRSH": "shoulder_r",
    "VLEL": "elbow_l", "VREL": "elbow_r",
    "VLWR": "wrist_l", "VRWR": "wrist_r",
    "VLHI": "hip_l", "VRHI": "hip_r",
    "VLKN": "knee_l", "VRKN": "knee_r",
    "VLAN": "ankle_l", "VRAN": "ankle_r",
    "LTRO": "hip_l", "RTRO": "hip_r",
    "LEFT_SHOULDER": "shoulder_l", "RIGHT_SHOULDER": "shoulder_r",
    "LEFT_ELBOW": "elbow_l", "RIGHT_ELBOW": "elbow_r",
    "LEFT_WRIST": "wrist_l", "RIGHT_WRIST": "wrist_r",
    "LEFT_HIP": "hip_l", "RIGHT_HIP": "hip_r",
    "LEFT_KNEE": "knee_l", "RIGHT_KNEE": "knee_r",
    "LEFT_ANKLE": "ankle_l", "RIGHT_ANKLE": "ankle_r",
    "LEFT_FOOT_INDEX": "toe_l", "RIGHT_FOOT_INDEX": "toe_r",
}

_C3D_MIDPOINTS = {
    "hip_l": [("LASI", "LPSI"), ("LASI", "SACR")],
    "hip_r": [("RASI", "RPSI"), ("RASI", "SACR")],
    "head": [("LFHD", "RFHD"), ("LHEA", "RHEA"), ("FHEA", "BHEA")],
    "shoulder_c": [("C7", "CLAV"), ("T2", "STER")],
    "toe_l": [("L.MT1", "L.MT5")], "toe_r": [("R.MT1", "R.MT5")],
    "knee_l": [("L.KNEE", "L.KNEE.MEDIAL")], "knee_r": [("R.KNEE", "R.KNEE.MEDIAL")],
    "ankle_l": [("L.ANKLE", "L.ANKLE.MEDIAL")], "ankle_r": [("R.ANKLE", "R.ANKLE.MEDIAL")],
    "hip_l_wbds": [("L.ASIS", "L.PSIS")],
}
_C3D_MIDPOINTS["hip_l"].append(("L.ASIS", "L.PSIS"))
_C3D_MIDPOINTS["hip_r"].append(("R.ASIS", "R.PSIS"))
del _C3D_MIDPOINTS["hip_l_wbds"]

_ANGLE_PATTERN = re.compile(r"^([LR])(HIP|KNEE|ANKLE|SHOULDER|ELBOW)ANGLES?$")
_TRUNK_ANGLE = re.compile(r"^[LR]?(THORAX|TRUNK|SPINE|NECK)ANGLES?$")


# --- Utilita' comuni ai lettori ---
# Normalizzazione di un'etichetta, conversione delle coordinate in metri e
# composizione della struttura restituita, con copertura di ogni punto e
# registrazione delle assenze.
def _key(label):
    s = str(label).replace("\x00", "").strip().upper().replace(" ", "")
    s = s.replace("__PY", "")
    if ":" in s:
        s = s.rsplit(":", 1)[-1]
    return s


def _scale_to_metres(xyz, unit_declared=None):
    fattori = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}
    if unit_declared and str(unit_declared).lower() in fattori:
        return xyz * fattori[str(unit_declared).lower()]
    finiti = xyz[np.isfinite(xyz)]
    return xyz / 1000.0 if finiti.size and np.nanmax(np.abs(finiti)) > 20 else xyz


def _finalise(points, meta, angles=None, events=None):
    for nome, (tipo, a, b) in anatomy.DERIVED_POINTS.items():
        if nome in points or tipo != "mid":
            continue
        if a in points and b in points:
            points[nome] = (points[a] + points[b]) / 2.0

    assenti = {}
    for nome in anatomy.POINTS:
        if nome not in points:
            assenti[nome] = (anatomy.NOT_IN_SOURCE, "")
            continue
        copertura = float(np.all(np.isfinite(points[nome]), axis=1).mean())
        if copertura < anatomy.MIN_COVERAGE:
            assenti[nome] = (anatomy.LOW_COVERAGE, f"{copertura:.1%} dei frame")
            del points[nome]
    return {"points": points, "angles": angles or {}, "absent": assenti,
            "events": events or {}, "meta": meta}


# --- Sorgente: cartella running_* del tracker MediaPipe ---
# Riconoscimento della cartella e lettura dei landmark fotogramma per fotogramma,
# scartando quelli sotto la soglia di visibilita'.
def is_running_folder(path):
    return os.path.isdir(path) and os.path.isfile(
        os.path.join(path, "segments_frames.json"))


def read_running_folder(path):
    with open(os.path.join(path, "segments_frames.json")) as f:
        contenuto = json.load(f)
    meta_file = contenuto.get("metadata", {})
    frames = contenuto.get("frames", [])
    n = len(frames)

    grezzi = {nome: np.full((n, 3), np.nan) for nome in _MEDIAPIPE_INDEX}
    naso = np.full((n, 3), np.nan)
    for i, fr in enumerate(frames):
        lm = fr.get("world_landmarks") or []
        if len(lm) < 33:
            continue
        for nome, k in _MEDIAPIPE_INDEX.items():
            p = lm[k]
            if p.get("visibility", 1.0) >= MIN_VISIBILITY:
                grezzi[nome][i] = (p["x"], p["y"], p["z"])
        if lm[_NOSE].get("visibility", 1.0) >= MIN_VISIBILITY:
            naso[i] = (lm[_NOSE]["x"], lm[_NOSE]["y"], lm[_NOSE]["z"])

    for v in list(grezzi.values()) + [naso]:
        v[:, 1] *= -1
        v[:, 2] *= -1

    punti = {k: v for k, v in grezzi.items() if k not in ("ear_l", "ear_r")}
    orecchie = (grezzi["ear_l"] + grezzi["ear_r"]) / 2.0
    punti["head"] = np.where(np.isfinite(orecchie), orecchie, naso)

    meta = {
        "fps": float(meta_file.get("fps") or 0) or None,
        "n_frames": n,
        "source": os.path.basename(os.path.normpath(path)),
        "source_format": "mediapipe_json",
        "capture_system": "monocular_pose_estimation",
        "view": meta_file.get("view"),
        "sport": meta_file.get("sport"),
        "coordinate_system": meta_file.get("coordinate_system"),
    }
    return _finalise(punti, meta)


# --- Sorgente: file C3D nelle convenzioni standard ---
# Eventi dichiarati nel file, punti riconosciuti dalle etichette e angoli di
# modello quando la sorgente li porta gia' calcolati.
def _eventi_c3d(c):
    p = c["parameters"]
    ev = p.get("EVENT")
    if not ev or "LABELS" not in ev or len(ev["LABELS"]["value"]) == 0:
        return {}
    rate = float(p["POINT"]["RATE"]["value"][0])
    primo = int(c["header"]["points"]["first_frame"])
    tempi = np.asarray(ev["TIMES"]["value"], dtype=float)
    tempi = tempi[1] if tempi.ndim == 2 else tempi
    contesti = [str(x).strip().lower() for x in ev["CONTEXTS"]["value"]]
    etichette = [str(x).strip().lower() for x in ev["LABELS"]["value"]]
    out = {}
    for i in range(len(etichette)):
        if etichette[i] != "foot strike":
            continue
        lato = contesti[i][:1]
        if lato not in ("l", "r"):
            continue
        out.setdefault(lato, []).append(int(round(tempi[i] * rate)) - primo)
    return {k: sorted(v) for k, v in out.items()}


def _read_c3d(path):
    import ezc3d
    c = ezc3d.c3d(path)
    p = c["parameters"]
    etichette = [str(x) for x in p["POINT"]["LABELS"]["value"]]
    P = c["data"]["points"]
    residui = c["data"].get("meta_points", {}).get("residuals")
    try:
        unita = str(p["POINT"]["UNITS"]["value"][0]).strip().lower()
    except Exception:
        unita = None
    fps = float(p["POINT"]["RATE"]["value"][0])
    canali = []
    for i, lb in enumerate(etichette):
        if i >= P.shape[1]:
            break
        xyz = P[:3, i, :].T.copy()
        cattivi = np.all(np.abs(xyz) < 1e-6, axis=1)
        if residui is not None and i < residui.shape[1]:
            r = np.asarray(residui[0, i, :], dtype=float)
            cattivi |= (~np.isfinite(r)) | (r < 0)
        xyz[cattivi] = np.nan
        canali.append((lb, xyz))
    return canali, fps, P.shape[2], unita, _eventi_c3d(c)


def read_c3d(path):
    canali, fps, n, unita, eventi = _read_c3d(path)

    def cerca(chiave):
        candidati = [xyz for lb, xyz in canali if _key(lb) == chiave]
        if not candidati:
            return None
        migliore = max(candidati, key=lambda v: np.all(np.isfinite(v), axis=1).sum())
        return _scale_to_metres(migliore, unita)

    punti = {}
    for chiave, canonico in _C3D_LABELS.items():
        if canonico in punti:
            continue
        v = cerca(chiave)
        if v is not None:
            punti[canonico] = v
    for canonico, ricette in _C3D_MIDPOINTS.items():
        if canonico in punti:
            continue
        for a, b in ricette:
            va, vb = cerca(a), cerca(b)
            if va is not None and vb is not None:
                punti[canonico] = (va + vb) / 2.0
                break

    angoli = {}
    for lb, xyz in canali:
        k = _key(lb)
        m = _ANGLE_PATTERN.match(k) or _TRUNK_ANGLE.match(k)
        if not m:
            continue
        if m.re is _ANGLE_PATTERN:
            giunto = f"{m.group(2).lower()}_{m.group(1).lower()}"
        else:
            giunto = {"THORAX": "trunk", "TRUNK": "trunk",
                      "SPINE": "spine", "NECK": "neck"}[m.group(1)]
        if giunto not in anatomy.JOINTS:
            continue
        for k2, piano in enumerate(anatomy.PLANES):
            angoli[(giunto, piano)] = xyz[:, k2]

    meta = {"fps": fps, "n_frames": n,
            "source": os.path.splitext(os.path.basename(path))[0],
            "source_format": "c3d", "capture_system": "marker_based"}
    return _finalise(punti, meta, angoli, eventi)


# --- Ingresso unico e ricerca delle sorgenti ---
# `read` sceglie il lettore in base al percorso ricevuto; `discover` elenca le
# sorgenti trovate sotto una cartella, fino alla profondita' indicata.
def read(path):
    if is_running_folder(path):
        return read_running_folder(path)
    if os.path.isdir(path):
        raise ValueError(f"cartella non riconosciuta come sessione: {path}")
    if os.path.splitext(path)[1].lower() == ".c3d":
        return read_c3d(path)
    raise ValueError(f"formato non supportato: {path}")


PROFONDITA_MAX = 4


def discover(root, profondita=PROFONDITA_MAX):
    if not os.path.isdir(root):
        return []
    out = []
    for nome in sorted(os.listdir(root)):
        p = os.path.join(root, nome)
        if is_running_folder(p):
            out.append(p)
        elif os.path.isfile(p) and nome.lower().endswith(".c3d"):
            out.append(p)
        elif os.path.isdir(p) and profondita > 0 and not nome.startswith("."):
            out.extend(discover(p, profondita - 1))
    return out
