"""Vocabolario anatomico: i punti che servono, i segmenti che ne discendono.

Sedici punti, nessuno in piu'. Sono i centri articolari che qualunque sistema di
cattura -- marker fisici o stima di posa da video -- e' in grado di fornire.
Tutto il resto (segmenti, terne, angoli) si costruisce da questi.

Convenzione delle terne, raccomandazione ISB (Wu et al. 2002):
    Y  asse lungo del segmento, dal distale al prossimale (verso l'alto)
    Z  asse mediolaterale, verso destra del soggetto
    X  asse anteroposteriore, in avanti,  X = Y x Z
Angoli come Eulero ZXY della rotazione relativa, secondo Grood e Suntay:
    prima la flessione (asse Z), poi ab/adduzione (X), poi rotazione assiale (Y).
"""
import numpy as np

# --- Provenienza di un valore ---
MEASURED = "measured"     # letto dalla sorgente cosi' com'e'
DERIVED = "derived"       # calcolato in modo deterministico da punti misurati
MODELLED = "modelled"     # dipende da un'assunzione di modello dichiarata
FILLED = "filled"         # interpolato su un buco breve

# --- Motivi di assenza ---
NOT_IN_SOURCE = "not_in_source"       # la sorgente non contiene il punto
LOW_COVERAGE = "low_coverage"         # presente ma visto troppo poco
MISSING_INPUTS = "missing_inputs"     # mancano i punti da cui si costruisce
NO_TIME_BASE = "no_time_base"         # senza fps non esistono derivate temporali
NOT_OBSERVABLE = "not_observable"     # il modello di segmento non vede quel piano

MIN_COVERAGE = 0.05       # sotto il 5% di frame validi il punto non e' una misura


# --- I sedici punti ---
POINTS = (
    "head",                          # centro del capo
    "shoulder_c",                    # centro delle spalle (giugulare/C7)
    "shoulder_l", "shoulder_r",
    "elbow_l", "elbow_r",
    "wrist_l", "wrist_r",
    "hip_l", "hip_r",
    "knee_l", "knee_r",
    "ankle_l", "ankle_r",
    "heel_l", "heel_r",
    "toe_l", "toe_r",
)

# Punti che si ricavano da altri quando la sorgente non li da'.
#   ("mid", a, b)  punto medio fra due punti
DERIVED_POINTS = {
    "shoulder_c": ("mid", "shoulder_l", "shoulder_r"),
    # centri costruiti: non sono punti da riconoscere ma centri di segmento, e
    # servono come base degli assi del tronco e del corpo.
    "hip_c": ("mid", "hip_l", "hip_r"),
    "knee_c": ("mid", "knee_l", "knee_r"),
    # Estremi del torace secondo la raccomandazione ISB: l'asse lungo unisce il
    # punto medio fra sterno e vertebra bassa a quello fra giugulare e C7.
    "thorax_up": ("mid", "shoulder_c", "c7"),
    "thorax_low": ("mid", "sternum", "t10"),
}

# Marker ausiliari: non sono centri articolari e non entrano fra i punti
# canonici, perche' non tutte le sorgenti li hanno. Servono a definire meglio un
# segmento quando ci sono. Restano un vocabolario chiuso: un segmento non puo'
# riferirsi a un punto che nessuno ha dichiarato.
#
# Le bacchette laterali di coscia e gamba (LTHI, LTIB) sono state provate e
# tolte: la ricetta "plane" prende la NORMALE al piano dei tre punti come asse
# di flessione, e un piano che contiene una bacchetta laterale e' quasi
# frontale, quindi la sua normale punta in avanti invece che di lato. Misurato,
# peggioravano la terna e spostavano la dorsiflessione nel piano sbagliato.
AUX_POINTS = (
    "c7",         # processo spinoso di C7
    "sternum",    # processo xifoideo
    "t10",        # processo spinoso toracico basso
)


# --- I segmenti che i punti danno per costruzione ---
# Ogni terna ha bisogno di tre punti non allineati: due danno l'asse lungo, il
# terzo fissa la rotazione attorno a esso. Con soli centri articolari il terzo
# punto e' spesso il giunto adiacente, e allora l'assunzione e' che il segmento
# ruoti nel piano dei tre -- e va dichiarata.
#
#   axis="long"    Y dall'estremo distale al prossimale (arti, collo)
#   axis="lateral" Z fra i due estremi (bacino, clavicole): segmenti trasversali
#   axis="forward" X dall'estremo prossimale al distale (piede)
#
#   roll:  come si fissa la rotazione attorno all'asse lungo. Puo' essere una
#          lista di ricette: vince la prima i cui punti sono disponibili, cosi'
#          un'acquisizione dei soli arti inferiori resta orientabile.
#     ("plane", a, b, c)      normale al piano dei tre punti -> asse di flessione
#     ("toward", p)           verso un punto
#     ("direction", da, a)    lungo la congiungente di due punti
#     ("borrow", segmento)    presa da un altro segmento gia' costruito
SEGMENTS = {
    # cinto pelvico: tre punti non allineati -> orientamento interamente osservato
    "pelvis": {"axis": "lateral", "ends": ("hip_l", "hip_r"),
               "roll": [("direction", "hip_c", "shoulder_c"),
                        ("direction", "knee_c", "hip_c")], "provenance": DERIVED},
    # cinto scapolare: le due clavicole condividono un'unica terna, perche' con
    # questi punti il movimento scapolare non e' osservabile e fingere due
    # orientamenti distinti sarebbe inventare un grado di liberta'.
    # Il rollio della clavicola serve solo a dire da che parte e' l'alto. Il
    # tronco lo dice meglio, ma non c'e' sempre: basta un'inquadratura che tagli
    # sotto le spalle perche' le anche restino fuori dal fotogramma, e allora
    # senza ripiego cadeva tutto il cingolo scapolare -- e con esso l'angolo di
    # spalla, che e' l'ultimo segnale ciclico disponibile in quel caso.
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
    # arti: l'asse di flessione viene dal piano dei tre giunti della catena
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
    # Coscia e gamba: la bacchetta laterale, quando c'e', misura la rotazione
    # attorno all'asse lungo; il piano dei tre centri articolari la assume
    # soltanto. Le due ricette hanno percio' provenienza diversa, dichiarata in
    # `roll_provenance` nello stesso ordine: la terna vale quanto il modo in cui
    # e' stata fissata, e una sorgente senza bacchette non deve sembrare
    # equivalente a una che le ha.
    # Il femore NON usa la bacchetta: misurato contro le terne del laboratorio,
    # il piano dei tre centri articolari sbaglia di 9.6 gradi e la bacchetta di
    # 25.7. Il piano anca-ginocchio-caviglia e' il piano di progressione della
    # gamba, stabile per tutto il passo; la bacchetta di coscia sta su tessuto
    # molle e va interpretata con le larghezze articolari che qui non abbiamo.
    "femur_l": {"axis": "long", "ends": ("knee_l", "hip_l"),
                "roll": ("plane", "hip_l", "knee_l", "ankle_l"),
                "provenance": MODELLED},
    "femur_r": {"axis": "long", "ends": ("knee_r", "hip_r"),
                "roll": ("plane", "hip_r", "knee_r", "ankle_r"),
                "provenance": MODELLED},
    # Nemmeno la gamba usa la bacchetta, e per una ragione geometrica: la
    # ricetta "plane" prende la NORMALE al piano dei tre punti come asse di
    # flessione, e il piano ginocchio-caviglia-bacchetta e' quasi frontale
    # (la bacchetta sta di lato), quindi la sua normale punta in avanti invece
    # che di lato. Misurato: con la bacchetta la terna della gamba esce con X e
    # Z scambiati rispetto a tutti gli altri segmenti, e la dorsiflessione
    # finisce nel piano frontale. Il piano ginocchio-caviglia-punta e' invece
    # sagittale, e la sua normale e' l'asse di flessione vero.
    "shank_l": {"axis": "long", "ends": ("ankle_l", "knee_l"),
                "roll": ("plane", "knee_l", "ankle_l", "toe_l"),
                "provenance": MODELLED},
    "shank_r": {"axis": "long", "ends": ("ankle_r", "knee_r"),
                "roll": ("plane", "knee_r", "ankle_r", "toe_r"),
                "provenance": MODELLED},
    # piede: l'asse lungo e' orizzontale, in avanti; il ginocchio fissa il piano
    # piede: l'asse lungo e' misurato, ma il piano lo fissa il ginocchio, quindi
    # inversione ed eversione discendono da un'assunzione, non da un'osservazione.
    # Asse lungo dal tallone alla punta, come il modello clinico: "The main
    # foot segment is constructed using the TOE-HEE line as the primary axis"
    # (Vicon, Plug-in Gait Reference Guide). Con caviglia->punta l'escursione
    # misurata fra gamba e piede scende a 28.8 gradi contro i 32.5 che da'
    # tallone->punta, a fronte di 33.3 dichiarati dal modello.
    # Senza tallone si ripiega su caviglia->punta: peggiore, ma non assente.
    "foot_l": {"axis": "forward",
               "ends": [("heel_l", "toe_l"), ("ankle_l", "toe_l")],
               "roll": ("toward", "knee_l"), "provenance": MODELLED},
    "foot_r": {"axis": "forward",
               "ends": [("heel_r", "toe_r"), ("ankle_r", "toe_r")],
               "roll": ("toward", "knee_r"), "provenance": MODELLED},
    # tronco: asse lungo dal centro del bacino al centro delle spalle. hip_c e'
    # un centro costruito (vedi DERIVED_POINTS), non un punto da riconoscere:
    # entra qui solo per fissare l'asse, e non compare mai fra i punti scritti
    # nel C3D. Il terzo asse viene dalla linea delle spalle o, in mancanza,
    # da quella del bacino: due punti indipendenti dall'asse lungo, non un
    # piano assunto, quindi la rotazione attorno all'asse lungo (il piano
    # trasverso del giunto "spine") e' osservata, non modellata.
    # Torace. Con i marker toracici la terna e' quella della raccomandazione
    # ISB: asse lungo fra il punto medio sterno-vertebra e quello giugulare-C7,
    # asse di flessione normale al piano sagittale che quei tre punti
    # definiscono. Senza, si ripiega sulla retta bacino-spalle, che e' una
    # direzione e non un corpo: da' l'inclinazione ma non la torsione, e il
    # terzo asse va preso dalla linea delle spalle.
    "trunk": {"axis": "long",
              "ends": [("thorax_low", "thorax_up"), ("hip_c", "shoulder_c")],
              "roll": [("plane", "c7", "shoulder_c", "thorax_low"),
                       ("direction", "shoulder_l", "shoulder_r"),
                       ("direction", "hip_l", "hip_r")],
              "provenance": DERIVED},
}

# Terna del laboratorio: verticale e direzione di avanzamento. Serve come
# riferimento per l'inclinazione del tronco.
WORLD = "world"


def segment_ends_options(segment):
    """Coppie di estremi in ordine di preferenza: vince la prima disponibile."""
    ends = SEGMENTS[segment]["ends"]
    return list(ends) if isinstance(ends, list) else [ends]


def segment_inputs(segment):
    """Punti minimi per costruire la terna: quelli dell'ultimo ripiego.

    Le alternative precedenti danno un asse migliore ma non sono obbligatorie:
    chiedere anche quelle renderebbe il segmento impossibile su una sorgente
    che non le ha.
    """
    return sorted(set(segment_ends_options(segment)[-1]))


def roll_recipes(segment):
    """Ricette per il terzo asse, in ordine di preferenza."""
    roll = SEGMENTS[segment].get("roll")
    if roll is None:
        return []
    return list(roll) if isinstance(roll, list) else [roll]


def roll_inputs(recipe):
    """Punti che una ricetta richiede."""
    if recipe[0] in ("plane", "direction"):
        return list(recipe[1:])
    if recipe[0] == "toward":
        return [recipe[1]]
    return []


# --- I giunti ---
PLANES = ("sagittal", "frontal", "transverse")
EULER_SEQUENCE = "ZXY"          # Grood e Suntay: flessione, ab/adduzione, rotazione

# I segni portano ogni asse alla convenzione clinica: flessione, adduzione e
# rotazione interna positive. Le terne hanno Z sempre verso destra del soggetto,
# quindi il lato sinistro va specchiato sui piani frontale e trasverso.
JOINTS = {
    # tronco rispetto al laboratorio: prima di avere un segmento di tronco vero
    # questo giunto usava il bacino come surrogato (proximal=world, distal=
    # pelvis): un'approssimazione dichiarata, non un errore, ma ora che il
    # tronco e' un segmento con terna propria misura cio' che il nome promette.
    "trunk":      {"proximal": WORLD, "distal": "trunk", "signs": (-1, 1, 1),
                   "sagittal": "flexion_extension", "frontal": "lateral_bending",
                   "transverse": "axial_rotation"},
    # tronco rispetto al bacino: la grandezza clinica "Spine Angles" dei modelli
    # Plug-in Gait. E' quella che isola il movimento della colonna dal
    # movimento del bacino sotto di essa.
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
    """Quanto e' osservato un piano, quando l'angolo lo calcoliamo noi.

    Un segmento la cui rotazione attorno all'asse lungo viene da un'assunzione
    di piano non puo' fornire la rotazione assiale del giunto. La flessione e
    l'ab/adduzione restano determinate finche' almeno uno dei due segmenti ha
    orientamento pieno; fra due segmenti a piano il modello e' una cerniera e
    resta solo la flessione.

    I piani che risultano MODELLED non vengono scritti come valori ma registrati
    come assenza: un'escursione di novanta gradi che nasce dal rumore della
    normale a un piano non e' una misura imprecisa, e' un numero inventato.
    """
    jd = JOINTS[joint]
    pieno = {s for s, d in SEGMENTS.items() if d["provenance"] == DERIVED}
    pieno.add(WORLD)
    prossimale, distale = jd["proximal"] in pieno, jd["distal"] in pieno
    if joint == "spine" and plane == "sagittal":
        # il bacino prende il suo terzo asse dalla stessa direzione (hip_c ->
        # shoulder_c) che definisce l'asse lungo del tronco: la flessione
        # relativa fra i due -- la prima rotazione della sequenza ZXY -- e'
        # quindi zero per costruzione, in ogni posa, non perche' il soggetto
        # non flette la colonna. Verificato sui dati: scarto dell'ordine di
        # 1e-15 gradi su un'intera sessione di corsa. Un valore sempre nullo
        # che sembra una misura sarebbe peggio di un'assenza dichiarata, e
        # cambiare il riferimento del bacino per liberare questo piano
        # romperebbe la flessione d'anca, che lo stesso riferimento tiene
        # indipendente dal movimento delle gambe.
        return MODELLED
    if prossimale and distale:
        return DERIVED
    if plane == "transverse":
        return MODELLED
    if prossimale or distale:
        return DERIVED
    # fra due segmenti a piano il modello e' una cerniera: solo la flessione
    return DERIVED if plane == "sagittal" else MODELLED


def observable_planes(joint):
    """Piani che il modello di segmento osserva davvero, per questo giunto."""
    return [p for p in PLANES if plane_provenance(joint, p) == DERIVED]


# --- Vocabolario dei gesti ---
# Il gesto decide due cose: come si segmenta in cicli e quali riferimenti di
# letteratura si applicano. Se le due parti usassero nomi diversi per la stessa
# cosa, i cicli funzionerebbero e le soglie no -- e una sessione senza alert
# sembrerebbe sana invece che non giudicata. Un nome solo, quindi, con i sinonimi
# che vi confluiscono.
SPORTS = ("walking", "running", "cycling", "swimming")

_SPORT_ALIASES = {
    "camminata": "walking", "cammino": "walking", "marcia": "walking",
    "corsa": "running", "corsetta": "running",
    "bicicletta": "cycling", "ciclismo": "cycling", "bici": "cycling",
    "nuoto": "swimming",
}


def canonical_sport(sport):
    """Nome del gesto nella forma unica, o None se non lo riconosciamo.

    Un nome ignoto non diventa un gesto inventato: resta None, e la sessione
    riceve i soli controlli anatomici -- quelli che valgono comunque.
    """
    if not sport:
        return None
    chiave = str(sport).strip().lower()
    if chiave in SPORTS:
        return chiave
    return _SPORT_ALIASES.get(chiave)


# --- Etichette per il C3D canonico ---
def point_label(name):
    """Nome del punto nel C3D. Maiuscolo, come vuole la prassi del formato."""
    return name.upper()


def angle_label(joint, plane):
    """Canale angolare: un punto a tre componenti per giunto, come i modelli."""
    return f"{joint.upper()}_ANGLE"


ANGLE_COMPONENT_ORDER = PLANES     # X sagittale, Y frontale, Z trasverso


def colours():
    """Colori dei segmenti, per il rendering."""
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
    """Versore, con NaN dove il vettore e' nullo o non finito."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = v / n
    u[np.broadcast_to(n, u.shape) < 1e-9] = np.nan
    return u
