"""Vocabolario anatomico: punti, segmenti, giunti, gesti ed etichette.

Raccoglie i nomi e le definizioni che il resto della catena usa: i sedici punti
canonici, i segmenti che si costruiscono a partire da quelli, i giunti come
coppia di segmenti, i nomi dei gesti riconosciuti e le etichette con cui punti e
angoli vengono scritti nel C3D canonico. Il modulo contiene solo definizioni e
funzioni di lettura: non legge file e non calcola cinematica.

La convenzione codificata qui e' quella ISB (Wu et al. 2002) con sequenza di
Grood e Suntay: terne con Y lungo il segmento dal distale al prossimale, Z verso
destra del soggetto, X = Y x Z; angoli come Eulero ZXY della rotazione del
segmento distale rispetto al prossimale.

Prerequisiti: numpy.
"""
import numpy as np

# --- Provenienza e assenza di un valore ---
# Etichette con cui un valore dichiara come e' stato ottenuto e, quando manca,
# per quale motivo. MIN_COVERAGE e' la quota minima di fotogrammi validi sotto la
# quale un punto non viene considerato una misura.
MEASURED = "measured"
DERIVED = "derived"
MODELLED = "modelled"
FILLED = "filled"

NOT_IN_SOURCE = "not_in_source"
LOW_COVERAGE = "low_coverage"
MISSING_INPUTS = "missing_inputs"
NO_TIME_BASE = "no_time_base"
NOT_OBSERVABLE = "not_observable"

MIN_COVERAGE = 0.05


# --- Punti ---
# I punti canonici del vocabolario, quelli che si ricavano da altri come punto
# medio e i marker ausiliari ammessi in ingresso.
POINTS = (
    "head",
    "shoulder_c",
    "shoulder_l", "shoulder_r",
    "elbow_l", "elbow_r",
    "wrist_l", "wrist_r",
    "hip_l", "hip_r",
    "knee_l", "knee_r",
    "ankle_l", "ankle_r",
    "heel_l", "heel_r",
    "toe_l", "toe_r",
)

DERIVED_POINTS = {
    "shoulder_c": ("mid", "shoulder_l", "shoulder_r"),
    "hip_c": ("mid", "hip_l", "hip_r"),
    "knee_c": ("mid", "knee_l", "knee_r"),
    "thorax_up": ("mid", "shoulder_c", "c7"),
    "thorax_low": ("mid", "sternum", "t10"),
}


AUX_POINTS = (
    "c7",
    "sternum",
    "t10",
)


# --- Segmenti ---
# Per ogni segmento: i due estremi, l'asse che ne discende, le ricette con cui si
# fissa la rotazione attorno a quell'asse e la provenienza del segmento. WORLD e'
# la terna del laboratorio. Le funzioni che seguono leggono queste voci ed
# elencano i punti di cui ogni segmento ha bisogno.
SEGMENTS = {
    "pelvis": {"axis": "lateral", "ends": ("hip_l", "hip_r"),
               "roll": [("direction", "hip_c", "shoulder_c"),
                        ("direction", "knee_c", "hip_c")], "provenance": DERIVED},
    "clavicle_l": {"axis": "lateral", "ends": ("shoulder_l", "shoulder_r"),
                   "roll": [("direction", "hip_c", "shoulder_c"),
                            ("direction", "shoulder_c", "head")],
                   "provenance": DERIVED},
    "clavicle_r": {"axis": "lateral", "ends": ("shoulder_l", "shoulder_r"),
                   "roll": [("direction", "hip_c", "shoulder_c"),
                            ("direction", "shoulder_c", "head")],
                   "provenance": DERIVED},
    "neck": {"axis": "long", "ends": ("shoulder_c", "head"),
             "roll": ("borrow", "clavicle_r"), "provenance": MODELLED},
    "humerus_l": {"axis": "long", "ends": ("elbow_l", "shoulder_l"),
                  "roll": ("plane", "shoulder_l", "elbow_l", "wrist_l"),
                  "provenance": MODELLED},
    "humerus_r": {"axis": "long", "ends": ("elbow_r", "shoulder_r"),
                  "roll": ("plane", "shoulder_r", "elbow_r", "wrist_r"),
                  "provenance": MODELLED},
    "forearm_l": {"axis": "long", "ends": ("wrist_l", "elbow_l"),
                  "roll": ("plane", "shoulder_l", "elbow_l", "wrist_l"),
                  "provenance": MODELLED},
    "forearm_r": {"axis": "long", "ends": ("wrist_r", "elbow_r"),
                  "roll": ("plane", "shoulder_r", "elbow_r", "wrist_r"),
                  "provenance": MODELLED},
    "femur_l": {"axis": "long", "ends": ("knee_l", "hip_l"),
                "roll": ("plane", "hip_l", "knee_l", "ankle_l"),
                "provenance": MODELLED},
    "femur_r": {"axis": "long", "ends": ("knee_r", "hip_r"),
                "roll": ("plane", "hip_r", "knee_r", "ankle_r"),
                "provenance": MODELLED},
    "shank_l": {"axis": "long", "ends": ("ankle_l", "knee_l"),
                "roll": ("plane", "knee_l", "ankle_l", "toe_l"),
                "provenance": MODELLED},
    "shank_r": {"axis": "long", "ends": ("ankle_r", "knee_r"),
                "roll": ("plane", "knee_r", "ankle_r", "toe_r"),
                "provenance": MODELLED},
    "foot_l": {"axis": "forward",
               "ends": [("heel_l", "toe_l"), ("ankle_l", "toe_l")],
               "roll": ("toward", "knee_l"), "provenance": MODELLED},
    "foot_r": {"axis": "forward",
               "ends": [("heel_r", "toe_r"), ("ankle_r", "toe_r")],
               "roll": ("toward", "knee_r"), "provenance": MODELLED},
    "trunk": {"axis": "long",
              "ends": [("thorax_low", "thorax_up"), ("hip_c", "shoulder_c")],
              "roll": [("plane", "c7", "shoulder_c", "thorax_low"),
                       ("direction", "shoulder_l", "shoulder_r"),
                       ("direction", "hip_l", "hip_r")],
              "provenance": DERIVED},
}

WORLD = "world"


def segment_ends_options(segment):
    ends = SEGMENTS[segment]["ends"]
    return list(ends) if isinstance(ends, list) else [ends]


def segment_inputs(segment):
    return sorted(set(segment_ends_options(segment)[-1]))


def roll_recipes(segment):
    roll = SEGMENTS[segment].get("roll")
    if roll is None:
        return []
    return list(roll) if isinstance(roll, list) else [roll]


def roll_inputs(recipe):
    if recipe[0] in ("plane", "direction"):
        return list(recipe[1:])
    if recipe[0] == "toward":
        return [recipe[1]]
    return []


# --- Giunti e piani ---
# Ogni giunto e' la coppia segmento prossimale/segmento distale, con i segni che
# riportano i tre assi alla convenzione clinica e il nome del movimento su
# ciascun piano. Le due funzioni ricavano da queste voci la provenienza di un
# piano e l'elenco dei piani osservabili per un giunto.
PLANES = ("sagittal", "frontal", "transverse")
EULER_SEQUENCE = "ZXY"

JOINTS = {
    "trunk":      {"proximal": WORLD, "distal": "trunk", "signs": (-1, 1, 1),
                   "sagittal": "flexion_extension", "frontal": "lateral_bending",
                   "transverse": "axial_rotation"},
    "spine":      {"proximal": "pelvis", "distal": "trunk", "signs": (-1, 1, 1),
                   "sagittal": "flexion_extension", "frontal": "lateral_bending",
                   "transverse": "axial_rotation"},
    "neck":       {"proximal": "clavicle_r", "distal": "neck", "signs": (-1, 1, 1),
                   "sagittal": "flexion_extension", "frontal": "lateral_bending",
                   "transverse": "axial_rotation"},
    "shoulder_l": {"proximal": "clavicle_l", "distal": "humerus_l", "signs": (1, -1, -1),
                   "sagittal": "flexion_extension", "frontal": "abduction_adduction",
                   "transverse": "internal_external_rotation"},
    "shoulder_r": {"proximal": "clavicle_r", "distal": "humerus_r", "signs": (1, 1, 1),
                   "sagittal": "flexion_extension", "frontal": "abduction_adduction",
                   "transverse": "internal_external_rotation"},
    "elbow_l":    {"proximal": "humerus_l", "distal": "forearm_l", "signs": (1, -1, -1),
                   "sagittal": "flexion_extension", "frontal": "varus_valgus",
                   "transverse": "pronation_supination"},
    "elbow_r":    {"proximal": "humerus_r", "distal": "forearm_r", "signs": (1, 1, 1),
                   "sagittal": "flexion_extension", "frontal": "varus_valgus",
                   "transverse": "pronation_supination"},
    "hip_l":      {"proximal": "pelvis", "distal": "femur_l", "signs": (1, -1, -1),
                   "sagittal": "flexion_extension", "frontal": "abduction_adduction",
                   "transverse": "internal_external_rotation"},
    "hip_r":      {"proximal": "pelvis", "distal": "femur_r", "signs": (1, 1, 1),
                   "sagittal": "flexion_extension", "frontal": "abduction_adduction",
                   "transverse": "internal_external_rotation"},
    "knee_l":     {"proximal": "femur_l", "distal": "shank_l", "signs": (-1, -1, -1),
                   "sagittal": "flexion_extension", "frontal": "varus_valgus",
                   "transverse": "internal_external_rotation"},
    "knee_r":     {"proximal": "femur_r", "distal": "shank_r", "signs": (-1, 1, 1),
                   "sagittal": "flexion_extension", "frontal": "varus_valgus",
                   "transverse": "internal_external_rotation"},
    "ankle_l":    {"proximal": "shank_l", "distal": "foot_l", "signs": (1, -1, -1),
                   "sagittal": "dorsi_plantarflexion", "frontal": "inversion_eversion",
                   "transverse": "internal_external_rotation"},
    "ankle_r":    {"proximal": "shank_r", "distal": "foot_r", "signs": (1, 1, 1),
                   "sagittal": "dorsi_plantarflexion", "frontal": "inversion_eversion",
                   "transverse": "internal_external_rotation"},
}


def plane_provenance(joint, plane):
    jd = JOINTS[joint]
    pieno = {s for s, d in SEGMENTS.items() if d["provenance"] == DERIVED}
    pieno.add(WORLD)
    prossimale, distale = jd["proximal"] in pieno, jd["distal"] in pieno
    if joint == "spine" and plane == "sagittal":
        return MODELLED
    if prossimale and distale:
        return DERIVED
    if plane == "transverse":
        return MODELLED
    if prossimale or distale:
        return DERIVED
    return DERIVED if plane == "sagittal" else MODELLED


def observable_planes(joint):
    return [p for p in PLANES if plane_provenance(joint, p) == DERIVED]


# --- Gesti ---
# Nomi canonici dei gesti, sinonimi accettati e la funzione che riconduce un nome
# dichiarato al nome canonico.
SPORTS = ("walking", "running", "cycling", "swimming")

_SPORT_ALIASES = {
    "camminata": "walking", "cammino": "walking", "marcia": "walking",
    "corsa": "running", "corsetta": "running",
    "bicicletta": "cycling", "ciclismo": "cycling", "bici": "cycling",
    "nuoto": "swimming",
}


def canonical_sport(sport):
    if not sport:
        return None
    chiave = str(sport).strip().lower()
    if chiave in SPORTS:
        return chiave
    return _SPORT_ALIASES.get(chiave)


# --- Etichette e utilita' ---
# Come punti e angoli vengono nominati nel C3D canonico, l'ordine delle tre
# componenti angolari, i colori associati ai segmenti e la normalizzazione di un
# vettore.
def point_label(name):
    return name.upper()


def angle_label(joint, plane):
    return f"{joint.upper()}_ANGLE"


ANGLE_COMPONENT_ORDER = PLANES


def colours():
    return {
        "pelvis": (0.85, 0.55, 0.20), "neck": (0.90, 0.80, 0.60),
        "clavicle_l": (0.40, 0.60, 0.85), "clavicle_r": (0.30, 0.50, 0.80),
        "humerus_l": (0.85, 0.40, 0.40), "humerus_r": (0.40, 0.80, 0.50),
        "forearm_l": (0.90, 0.55, 0.55), "forearm_r": (0.55, 0.85, 0.62),
        "femur_l": (0.85, 0.30, 0.30), "femur_r": (0.30, 0.75, 0.40),
        "shank_l": (0.90, 0.45, 0.45), "shank_r": (0.45, 0.85, 0.55),
        "foot_l": (0.75, 0.35, 0.55), "foot_r": (0.35, 0.70, 0.70),
        "trunk": (0.75, 0.70, 0.45),
    }


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = v / n
    u[np.broadcast_to(n, u.shape) < 1e-9] = np.nan
    return u
