"""Segmentazione in cicli, con strategia scelta in base allo sport.

Ogni gesto si ripete per motivi diversi: il passo si riconosce dal contatto al
suolo, la pedalata dal ginocchio, la bracciata dalla spalla. Senza sport
dichiarato si usa un metodo generico che cerca il giunto con l'escursione
dominante. Se nessun segnale e' ciclico, la prova resta un segmento unico.

La segmentazione avviene una volta in conversione: e' un dato della sessione,
non una scelta di chi interroga. Il metodo usato viaggia con i cicli, perche' un
ciclo da forza di reazione e uno da periodicita' non hanno lo stesso valore.
"""
import numpy as np

from . import anatomy
from scipy.signal import find_peaks

MIN_PERIOD_S = 0.3        # durata minima di un ciclo
# Quanto del gesto deve vedere il segnale su cui un metodo lavora. Senza questo
# limite la lista e' a prima-che-riesce e non guarda quanto dato c'e' sotto: una
# punta visibile nel 5% dei frame batteva una spalla visibile nel 100%, e i cicli
# uscivano da due secondi di una ripresa di sessanta. Il metodo migliore vince
# solo se ha di che lavorare.
#
# Meta' della ripresa. Misurato sul materiale del progetto, ogni valore fra 0.30
# e 0.75 da' lo stesso esito: il caso da escludere sta al 5%, il peggiore da
# tenere al 79%, e in mezzo c'e' un altopiano largo. 0.50 ne sta al centro, cioe'
# il piu' lontano possibile da entrambi i bordi. A 0.80 si perdeva una sessione
# per un punto percentuale, che e' il segno di una soglia appoggiata su un ciglio.
MIN_INPUT_COVERAGE = 0.50
ANGLE_FLOOR_DEG = 5.0     # escursione minima perche' un angolo sia ciclico
POSITION_FLOOR_M = 0.05   # escursione minima per una posizione
PROMINENCE_FRAC = 0.3     # prominenza richiesta, in frazione dell'escursione


def _span(series, window=None):
    """Escursione tipica DENTRO una finestra lunga come un ciclo.

    L'escursione globale non distingue un gesto ciclico da una lenta deriva: una
    registrazione statica di sei minuti accumula centimetri di spostamento e
    sembrerebbe ampia quanto un passo. La mediana delle escursioni locali no.
    """
    arr = np.asarray(series, dtype=float)
    if not window or arr.size <= window:
        return float(np.percentile(arr, 95) - np.percentile(arr, 5))
    passo = max(1, window // 2)
    locali = [np.ptp(arr[i:i + window]) for i in range(0, arr.size - window, passo)]
    return float(np.median(locali)) if locali else 0.0


def _valid_runs(valid, minimo):
    """Tratti contigui di dati veri, lunghi almeno `minimo` campioni."""
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
    """Massimi ciclici, cercati SOLO nei tratti realmente coperti.

    Interpolare attraverso un buco lungo produce un tratto rettilineo che non e'
    un gesto: i cicli si cercano dove il dato esiste, e i buchi separano.

    `minimo` e' quanti massimi deve avere un tratto per essere tenuto. Due
    servono a chiudere un ciclo con un solo segnale; uno basta quando i massimi
    di due segnali alterni (i due piedi) vengono uniti dopo.
    """
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
    """Eventi -> intervalli consecutivi. Nessun ciclo attraversa un buco."""
    out = [(a, b) for ev in (gruppi or []) for a, b in zip(ev, ev[1:]) if b - a >= min_dist]
    return out or None


# --- Metodi ---
# Ogni metodo restituisce `(cicli, lati)`, con `lati` lungo quanto `cicli`: il
# lato appartiene al singolo ciclo, non alla sessione. I due piedi producono
# falcate ugualmente valide e sfasate di mezzo periodo, e tenerne una sola
# buttava meta' del gesto osservato.
def _da_entrambi_i_lati(per_lato):
    """Unisce i cicli dei due lati in un solo elenco, ordinato nel tempo.

    L'ordinamento conta: l'indice del ciclo nel database segue il tempo, e due
    serie concatenate senza riordinare lo farebbero saltare all'indietro.
    """
    coppie = [(c, lato) for lato, cicli in per_lato for c in cicli]
    if not coppie:
        return None
    coppie.sort(key=lambda x: (x[0][0], x[0][1]))
    return [c for c, _ in coppie], [lato for _, lato in coppie]


def _source_events(ctx):
    """Falcata fra due appoggi dello stesso piede, come marcati dalla sorgente.

    Quando il file li porta, questi non sono una stima: il ciclo e' definito
    dall'evento, e ogni metodo che segue lo deduce dalle coordinate. Si prendono
    le falcate di entrambi i piedi: si sovrappongono nel tempo, com'e' normale
    nell'analisi del passo, e ognuna porta il proprio lato.
    """
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
    """Appoggio: massimo anteriore del piede rispetto al bacino (Zeni 2008).

    Il metodo originale usa il tallone; il nostro vocabolario ha la punta, che
    descrive lo stesso andamento anteroposteriore con un anticipo di fase
    costante. I cicli restano corretti in numero e durata, e il loro inizio e'
    spostato di una quantita' fissa: e' un limite dichiarato, non un errore.
    """
    pts = ctx["landmarks"]
    if not all(k in pts for k in ("hip_l", "hip_r")):
        return None
    center = (np.asarray(pts["hip_l"]) + np.asarray(pts["hip_r"])) / 2.0
    avanzamento = {}
    for side in ("r", "l"):
        toe = pts.get(f"toe_{side}")
        if toe is not None:
            avanzamento[side] = (np.asarray(toe) - center) @ ctx["ap"]

    # Falcata: due appoggi dello stesso piede. E' il ciclo del gesto, e resta il
    # primo tentativo perche' e' quello che la letteratura definisce. Si tengono
    # quelle di entrambi i piedi, come per gli eventi della sorgente.
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
    """Passo: appoggi alterni dei due piedi, quando la falcata non si chiude.

    Una passata di laboratorio dura una falcata per piede: due massimi dello
    stesso piede non ci sono, e cercarli soltanto la' rende non segmentabile
    ogni acquisizione breve. Gli appoggi ci sono comunque -- uno per piede -- e
    uniti in ordine descrivono i passi.

    Il passo e' mezzo ciclo, quindi vale meta' della distanza minima. Non e' una
    falcata e non va confuso con essa: chi lo produce lo dichiara nel nome del
    metodo, e cicli di passo e di falcata non sono confrontabili fra loro.
    """
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
    """Cicli dai massimi di flessione di un giunto, da entrambi i lati.

    Prima vinceva il destro se dava qualcosa, altrimenti il sinistro: una scelta
    per posizione, che su una stessa corsa prendeva 54 cicli da un braccio e ne
    ignorava 28 dall'altro. I due lati descrivono lo stesso gesto sfasato di
    mezzo periodo, quindi valgono entrambi e ognuno porta il proprio lato.

    Attenzione: qui i due lati NON sono ugualmente attendibili come lo sono gli
    appoggi marcati dalla sorgente. Sono surrogati, e uno dei due puo' essere
    molto piu' disturbato dell'altro; la dispersione delle durate, che il lato
    porta con se' nel database, e' cio' che permette di accorgersene.
    """
    per_lato = []
    # Solo giunti con lato: `_joint_flexion` viene chiamata con hip, knee e
    # shoulder, che in JOINTS esistono unicamente come coppia sinistra/destra.
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
    """Generico: usa il giunto con l'escursione maggiore fra quelli disponibili."""
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
    # Il lato resta vuoto: qui il ciclo viene dal giunto con l'escursione
    # maggiore, che puo' non averne uno. Prima ci finiva il nome del giunto, che
    # nella colonna `side` del database non significa niente.
    return (cycles, [None] * len(cycles)) if cycles else None


def _whole_trial(ctx):
    """Gesto non ciclico: la prova intera e' un segmento unico."""
    return ([(0, ctx["n_frames"])], [None]) if ctx["n_frames"] > 0 else None


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

# Sport supportati: quelli per cui esiste una definizione di ciclo motivata.
#
# Nel passo la spalla chiude la lista. Il braccio oscilla una volta per falcata,
# quindi il suo ritmo e' quello del gesto anche quando l'arto inferiore non e'
# stato ripreso -- ed e' una condizione comune, perche' basta un'inquadratura
# stretta perche' le caviglie restino fuori dal fotogramma. Senza questo passaggio
# una sessione di sessanta secondi con solo la parte alta finiva in `whole_trial`,
# cioe' perdeva tutti i cicli avendo un segnale periodico buono a disposizione.
#
# Sta in coda e non prima dell'anca perche' e' il piu' lontano dal contatto al
# suolo: si usa quando non e' rimasto altro. Il ginocchio non compare affatto qui,
# perche' il suo massimo di flessione cade a meta' oscillazione e non ha relazione
# stabile con l'appoggio: misurato, sbaglia il confine piu' di tutti gli altri.
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

# Metodi che non individuano un evento fisico del gesto.
_SURROGATE = {"hip_flexion", "knee_flexion", "shoulder_flexion",
              "dominant_periodicity"}


def _copertura(serie):
    """Frazione di frame in cui il dato esiste."""
    if serie is None:
        return 0.0
    a = np.asarray(serie, dtype=float)
    if a.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(a) if a.ndim == 1 else np.isfinite(a).all(axis=1)))


def _copertura_piede(ctx):
    """Il contatto si legge dalla punta RISPETTO al bacino: servono entrambi."""
    pts = ctx["landmarks"]
    bacino = min(_copertura(pts.get("hip_l")), _copertura(pts.get("hip_r")))
    return min(bacino, max(_copertura(pts.get("toe_l")), _copertura(pts.get("toe_r"))))


def _copertura_giunto(ctx, giunto):
    """Il migliore dei due lati: al metodo ne basta uno."""
    return max(_copertura((ctx["joints"].get(f"{giunto}_{lato}") or {}).get("sagittal"))
               for lato in ("r", "l"))


# Da quale segnale dipende ogni metodo. Chi non compare qui non ha un ingresso
# fisso da controllare: `dominant_periodicity` sceglie il giunto da se', e
# `whole_trial` non guarda nessun segnale.
INPUT_COVERAGE = {
    "toe_coordinate": _copertura_piede,
    "toe_coordinate_step": _copertura_piede,
    "hip_flexion": lambda ctx: _copertura_giunto(ctx, "hip"),
    "knee_flexion": lambda ctx: _copertura_giunto(ctx, "knee"),
    "shoulder_flexion": lambda ctx: _copertura_giunto(ctx, "shoulder"),
}


def _joints_by_plane(angles):
    """{(giunto, piano): serie} -> {giunto: {piano: serie}}."""
    out = {}
    for (giunto, piano), serie in (angles or {}).items():
        out.setdefault(giunto, {})[piano] = serie
    return out


def avviso_per_metodo(metodo):
    """L'avviso che accompagna un metodo di segmentazione, o None se non serve.

    Il nome del metodo viaggia col dato fino al database, l'avviso no: chi legge
    una sessione dal server non deve dover sapere che `whole_trial` significa
    "nessun ciclo trovato". I nomi sono gli stessi che `detect` emette, cosi' le
    due strade dicono la stessa cosa.
    """
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
    """Cicli della sessione: {method, frames, sides, warnings}."""
    meta = result.get("meta") or {}
    fps = meta.get("fps") or 0
    sport = anatomy.canonical_sport(meta.get("sport"))
    ctx = {"landmarks": result.get("points") or {},
           "events": result.get("events") or {},
           "joints": _joints_by_plane(result.get("angles")),
           # righe della terna: 0 anteriore, 1 alto, 2 destra
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
            # Non e' un fallimento del metodo: e' che il suo segnale non c'e'.
            # Va distinto, perche' cicli trovati su una frazione della ripresa
            # non descrivono la ripresa.
            scartati.append(name)
            continue
        found = METHODS[name](ctx)
        if not found:
            continue
        cycles, sides = found
        # L'avviso viene da `avviso_per_metodo` e non da un elenco ripetuto qui:
        # ora che finisce scritto nel C3D, chi legge il file e chi interroga il
        # database devono trovare la stessa frase, non due formulazioni parallele
        # che possono divergere.
        avviso = avviso_per_metodo(name)
        if avviso:
            warnings.append(avviso)
        if scartati:
            # Perche' non e' stato usato il metodo migliore: senza dirlo, un
            # ciclo da surrogato sembrerebbe una scelta e non un ripiego.
            warnings.append("primary_signal_low_coverage")
        return {"method": name,
                "frames": [[int(s), int(e)] for s, e in cycles],
                # Un lato per ciclo, non uno per sessione: le falcate dei due
                # piedi convivono nella stessa prova.
                "sides": list(sides),
                "warnings": warnings}
    return {"method": None, "sides": [], "frames": [],
            "warnings": warnings + [avviso_per_metodo(None)]
                     + (["primary_signal_low_coverage"] if scartati else [])}
