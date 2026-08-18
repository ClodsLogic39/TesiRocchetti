"""Lettori: da qualunque sorgente ai sedici punti canonici.

Ogni lettore restituisce la stessa struttura, cosi' il resto della catena non sa
piu' da dove vengono i dati:

    {"points":   {nome: array (n,3) in metri, NaN dove il punto non e' visto},
     "angles":   {(giunto, piano): array (n,) in gradi}   se la sorgente li ha
     "absent":   {nome: (motivo, dettaglio)},
     "meta":     {fps, n_frames, source, source_format, ...}}

Un punto assente resta assente: non viene mai sostituito con zero.
"""
import json
import os
import re

import numpy as np

from . import anatomy

# --- MediaPipe Pose: indici dei 33 punti che ci servono ---
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
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
MIN_VISIBILITY = 0.5      # sotto questa soglia MediaPipe sta indovinando

# --- C3D: etichette delle convenzioni standard -> punti canonici ---
# Solo i sedici che servono. Le altre etichette del file vengono ignorate.
_C3D_LABELS = {
    # Plug-in Gait / Vicon e derivati (pyCGM2, Cortex)
    "LSHO": "shoulder_l", "RSHO": "shoulder_r",
    "LELB": "elbow_l", "RELB": "elbow_r",
    "LWRA": "wrist_l", "RWRA": "wrist_r", "LWRI": "wrist_l", "RWRI": "wrist_r",
    "LTOE": "toe_l", "RTOE": "toe_r",
    "LHEE": "heel_l", "RHEE": "heel_r",
    # CLAV (giugulare) e' il centro spalle; C7 e' un punto posteriore e serve
    # invece a definire il torace, quindi non puo' valere per entrambi. Senza
    # CLAV il centro spalle si ricava dai due acromion (vedi DERIVED_POINTS).
    "CLAV": "shoulder_c",
    # centri articolari calcolati dal modello: migliori dei marker di superficie
    "LHJC": "hip_l", "RHJC": "hip_r",
    "LKJC": "knee_l", "RKJC": "knee_r",
    "LAJC": "ankle_l", "RAJC": "ankle_r",
    # Origine delle terne di segmento. In Plug-in Gait l'origine e' il centro
    # del giunto DISTALE, non del prossimale: "The femur origin is taken as the
    # knee joint center" (Vicon, Lower body kinematics). Quindi femore->
    # ginocchio, tibia->caviglia, omero->gomito, radio->polso.
    # Verificato sulle quote di una prova in piedi: LFEO sta a 0.55 m come LKNE
    # (54 mm di distanza, cioe' mezza larghezza di ginocchio), LTIO a 0.11 m
    # come LANK. Leggerli come giunto prossimale sposta tutto l'arto di un
    # giunto: l'anca diventa il ginocchio e il ginocchio la caviglia.
    "LFEO": "knee_l", "RFEO": "knee_r",
    "LTIO": "ankle_l", "RTIO": "ankle_r",
    # LFOO non e' la caviglia: misurato, dista 7 cm dalla punta contro i 22 cm
    # che separano punta e caviglia. Usarlo darebbe un piede lungo un terzo del
    # vero. L'origine del piede non entra quindi nel vocabolario.
    "LTOO": "toe_l", "RTOO": "toe_r",
    "LHUO": "elbow_l", "RHUO": "elbow_r",
    "LRAO": "wrist_l", "RRAO": "wrist_r",
    # Marker del torace: definiscono un segmento con orientamento vero, invece
    # della retta bacino-spalle che il progetto usa quando mancano. Il torace
    # cosi' costruito e' quello della raccomandazione ISB.
    "C7": "c7", "STRN": "sternum", "T10": "t10", "T8": "t10",
    # marker di superficie, usati se i centri non ci sono
    "LKNE": "knee_l", "RKNE": "knee_r",
    "LANK": "ankle_l", "RANK": "ankle_r",
    # WBDS / Visual3D
    "L.KNEE": "knee_l", "R.KNEE": "knee_r",
    "L.ANKLE": "ankle_l", "R.ANKLE": "ankle_r",
    "L.MT2": "toe_l", "R.MT2": "toe_r",
    "L.SHOULDER": "shoulder_l", "R.SHOULDER": "shoulder_r",
    # Motion Analysis Cortex: punti virtuali del modello, gia' centri articolari.
    # Molti sono vuoti in gran parte dei file: il filtro di copertura li scarta
    # da solo, senza bisogno di saperlo in anticipo.
    "VHEA": "head", "VNEC": "shoulder_c",
    "VLSH": "shoulder_l", "VRSH": "shoulder_r",
    "VLEL": "elbow_l", "VREL": "elbow_r",
    "VLWR": "wrist_l", "VRWR": "wrist_r",
    "VLHI": "hip_l", "VRHI": "hip_r",
    "VLKN": "knee_l", "VRKN": "knee_r",
    "VLAN": "ankle_l", "VRAN": "ankle_r",
    "LTRO": "hip_l", "RTRO": "hip_r",
    # MediaPipe esportato come C3D
    "LEFT_SHOULDER": "shoulder_l", "RIGHT_SHOULDER": "shoulder_r",
    "LEFT_ELBOW": "elbow_l", "RIGHT_ELBOW": "elbow_r",
    "LEFT_WRIST": "wrist_l", "RIGHT_WRIST": "wrist_r",
    "LEFT_HIP": "hip_l", "RIGHT_HIP": "hip_r",
    "LEFT_KNEE": "knee_l", "RIGHT_KNEE": "knee_r",
    "LEFT_ANKLE": "ankle_l", "RIGHT_ANKLE": "ankle_r",
    "LEFT_FOOT_INDEX": "toe_l", "RIGHT_FOOT_INDEX": "toe_r",
}

# Punti canonici che si ricavano dalla media di due etichette, quando il singolo
# punto non esiste nella convenzione (l'anca di Plug-in Gait sta fra le spine).
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

# Uscite di modello gia' calcolate: se ci sono, si conservano e non si ricalcola.
_ANGLE_PATTERN = re.compile(r"^([LR])(HIP|KNEE|ANKLE|SHOULDER|ELBOW)ANGLES?$")
# THORAX/TRUNK sono il tronco rispetto al laboratorio (il nostro giunto
# "trunk"); SPINE e' il tronco rispetto al bacino, la grandezza che isola il
# movimento della colonna (il nostro giunto "spine"). PELVIS non ha una voce:
# nel vocabolario non esiste un giunto "bacino rispetto al laboratorio", e
# infilarla in "trunk" (come faceva la vecchia versione, prima che il tronco
# avesse una terna propria) misurerebbe la grandezza sbagliata sotto il nome
# giusto.
# I modelli clinici scrivono queste grandezze una per lato (LSpineAngles,
# RSpineAngles) anche se il giunto e' centrale: e' lo stesso corpo visto dai due
# emilati. Il prefisso va quindi ammesso, o l'etichetta non viene riconosciuta e
# il valore del modello si perde in silenzio. Si tiene la prima occorrenza.
_TRUNK_ANGLE = re.compile(r"^[LR]?(THORAX|TRUNK|SPINE|NECK)ANGLES?$")


def _key(label):
    """Chiave di confronto: maiuscolo, senza spazi ne' prefisso di soggetto."""
    s = str(label).replace("\x00", "").strip().upper().replace(" ", "")
    s = s.replace("__PY", "")            # suffisso dei modelli pyCGM2
    if ":" in s:                         # prefisso "Soggetto:LASI"
        s = s.rsplit(":", 1)[-1]
    return s


def _scale_to_metres(xyz, unit_declared=None):
    """Porta le coordinate in metri secondo l'unita' dichiarata dal file."""
    fattori = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}
    if unit_declared and str(unit_declared).lower() in fattori:
        return xyz * fattori[str(unit_declared).lower()]
    finiti = xyz[np.isfinite(xyz)]
    return xyz / 1000.0 if finiti.size and np.nanmax(np.abs(finiti)) > 20 else xyz


def _finalise(points, meta, angles=None, events=None):
    """Completa i punti derivabili e registra le assenze con il motivo."""
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


# --- Sorgente 1: cartella running_* prodotta dal tracker MediaPipe ---
def is_running_folder(path):
    return os.path.isdir(path) and os.path.isfile(
        os.path.join(path, "segments_frames.json"))


def read_running_folder(path):
    """Legge `segments_frames.json`, campo `world_landmarks`.

    Il file contiene sia i segmenti gia' ricostruiti dal tracker sia i 33 punti
    grezzi di MediaPipe in metri. Si usano i punti: i segmenti sono un prodotto
    derivato del tracker, e ripartire da essi significherebbe ereditarne le
    scelte senza poterle verificare.
    """
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
            # MediaPipe emette una posizione a ogni frame anche quando non vede
            # il punto: la visibilita' dichiara quando e' un'estrapolazione.
            if p.get("visibility", 1.0) >= MIN_VISIBILITY:
                grezzi[nome][i] = (p["x"], p["y"], p["z"])
        if lm[_NOSE].get("visibility", 1.0) >= MIN_VISIBILITY:
            naso[i] = (lm[_NOSE]["x"], lm[_NOSE]["y"], lm[_NOSE]["z"])

    # MediaPipe world: X a destra, Y in basso, Z verso la camera. Portiamo la Y
    # in alto e la Z in avanti, come vuole la convenzione anatomica.
    for v in list(grezzi.values()) + [naso]:
        v[:, 1] *= -1
        v[:, 2] *= -1

    punti = {k: v for k, v in grezzi.items() if k not in ("ear_l", "ear_r")}
    # testa: punto medio delle orecchie, che sta vicino al centro del capo;
    # il naso e' l'ultima risorsa perche' sporge in avanti.
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


# --- Sorgente 2: C3D nelle convenzioni standard ---
def _eventi_c3d(c):
    """Appoggi marcati nella sorgente, per lato, in indici di frame locali.

    I C3D di laboratorio portano il gruppo standard `EVENT` con i contatti gia'
    segnati dal sistema di acquisizione. Sono la definizione del ciclo, non una
    stima: ricavarli di nuovo dalle coordinate della punta significa rifare
    peggio un lavoro gia' fatto, e su queste prove significa leggere un mezzo
    passo dove il laboratorio ha marcato due falcate intere.

    I tempi del gruppo EVENT sono assoluti dall'inizio della prova, mentre le
    coordinate cominciano al frame dichiarato nell'intestazione: senza sottrarre
    quello, gli eventi cadono fuori dal file.
    """
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
    # ezc3d tiene nella quarta riga dei punti la coordinata omogenea (sempre 1);
    # il residuo, che dichiara se il campione e' stato ricostruito, sta a parte.
    residui = c["data"].get("meta_points", {}).get("residuals")
    try:
        unita = str(p["POINT"]["UNITS"]["value"][0]).strip().lower()
    except Exception:
        unita = None
    fps = float(p["POINT"]["RATE"]["value"][0])
    # Un'etichetta puo' comparire piu' volte (marker omonimi su segmenti
    # diversi): si tengono tutte le occorrenze, indicizzate per posizione.
    canali = []
    for i, lb in enumerate(etichette):
        if i >= P.shape[1]:
            break
        xyz = P[:3, i, :].T.copy()
        # residuo negativo = punto non ricostruito; tutte-zero = sentinella
        cattivi = np.all(np.abs(xyz) < 1e-6, axis=1)
        if residui is not None and i < residui.shape[1]:
            r = np.asarray(residui[0, i, :], dtype=float)
            cattivi |= (~np.isfinite(r)) | (r < 0)
        xyz[cattivi] = np.nan
        canali.append((lb, xyz))
    return canali, fps, P.shape[2], unita, _eventi_c3d(c)


def read_c3d(path):
    """Riconosce i sedici punti fra le etichette del file, ignora tutto il resto."""
    canali, fps, n, unita, eventi = _read_c3d(path)

    def cerca(chiave):
        """Fra le occorrenze omonime vince quella vista piu' spesso."""
        candidati = [xyz for lb, xyz in canali if _key(lb) == chiave]
        if not candidati:
            return None
        migliore = max(candidati, key=lambda v: np.all(np.isfinite(v), axis=1).sum())
        return _scale_to_metres(migliore, unita)

    punti = {}
    for chiave, canonico in _C3D_LABELS.items():
        if canonico in punti:
            continue                       # la prima voce della tabella ha precedenza
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


# --- Ingresso unico ---
def read(path):
    """Legge una sorgente riconosciuta dalla struttura o dall'estensione."""
    if is_running_folder(path):
        return read_running_folder(path)
    if os.path.isdir(path):
        raise ValueError(f"cartella non riconosciuta come sessione: {path}")
    if os.path.splitext(path)[1].lower() == ".c3d":
        return read_c3d(path)
    raise ValueError(f"formato non supportato: {path}")


PROFONDITA_MAX = 4


def discover(root, profondita=PROFONDITA_MAX):
    """Sorgenti caricabili sotto una cartella, anche annidate.

    I dataset di laboratorio arrivano organizzati per condizione (piano, salita,
    rampa) in sottocartelle: fermarsi al primo livello significa non vederli.
    La discesa ha un tetto di profondita' perche' un percorso sbagliato non si
    trasformi in una scansione dell'intero disco.

    Una cartella di sessione (`running_*`) e' essa stessa una sorgente e non si
    percorre oltre: dentro ci sono i suoi file, non altre sessioni.
    """
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
