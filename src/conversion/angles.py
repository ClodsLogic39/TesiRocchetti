"""Dai punti agli angoli: terne di segmento, quaternioni, Eulero ZXY.

Si calcola solo cio' che la sorgente non fornisce gia'. Se il file contiene
angoli di modello a tre gradi di liberta', quelli restano: sono stati prodotti
da un modello calibrato sul soggetto, e ricalcolarli sarebbe una perdita.

Convenzione, raccomandazione ISB (Wu et al. 2002) con sequenza di Grood e Suntay:
terne con Y lungo il segmento verso l'alto, Z verso destra, X = Y x Z; angoli
come Eulero ZXY della rotazione del distale rispetto al prossimale, cioe' prima
la flessione, poi l'ab/adduzione, infine la rotazione assiale.
"""
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

from . import anatomy

CUTOFF_HZ = 6.0            # passa-basso di prassi per la cinematica del cammino
FILTER_ORDER = 4
MAX_GAP_S = 0.20           # buchi piu' lunghi restano buchi
MIN_RUN = 15               # campioni contigui minimi perche' un tratto sia usabile
MIN_PLANE_DEG = 10.0       # sotto questa flessione il piano del giunto non esiste
MAX_ANGULAR_DEG_S = 1000.0  # oltre non e' movimento umano ma rumore

_unit = anatomy.unit


# --- Pulizia delle traiettorie ---
def _runs(valid):
    out, i, n = [], 0, len(valid)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and valid[j + 1]:
            j += 1
        out.append((i, j + 1))
        i = j + 1
    return out


def interpolate_gaps(series, max_gap):
    """Colma i buchi interni brevi. Bordi e buchi lunghi restano NaN."""
    out = np.asarray(series, dtype=float).copy()
    valid = np.all(np.isfinite(out), axis=1)
    if valid.sum() < 2:
        return out
    idx = np.arange(len(out))
    mancanti = np.where(~valid)[0]
    if mancanti.size == 0:
        return out
    for run in np.split(mancanti, np.where(np.diff(mancanti) != 1)[0] + 1):
        s, e = run[0], run[-1]
        if s == 0 or e == len(out) - 1 or (e - s + 1) > max_gap:
            continue
        for c in range(3):
            out[s:e + 1, c] = np.interp(idx[s:e + 1], idx[valid], out[valid, c])
    return out


def drop_short_runs(series, minimo):
    """Scarta i tratti troppo corti: qualche frame isolato non e' una misura."""
    out = np.asarray(series, dtype=float).copy()
    tenuti = np.zeros(len(out), dtype=bool)
    for s, e in _runs(np.all(np.isfinite(out), axis=1)):
        if e - s >= minimo:
            tenuti[s:e] = True
    out[~tenuti] = np.nan
    return out


def lowpass(series, fps, cutoff=CUTOFF_HZ, order=FILTER_ORDER):
    """Butterworth a fase nulla, applicato a ogni tratto contiguo separatamente.

    Filtrare attraverso un buco spalmerebbe valori inventati sui campioni buoni
    che lo circondano.
    """
    out = np.asarray(series, dtype=float).copy()
    if not fps or cutoff <= 0 or cutoff >= fps / 2:
        return out
    b, a = butter(order, cutoff / (fps / 2), btype="low")
    minimo = max(MIN_RUN, 3 * max(len(a), len(b)))
    for s, e in _runs(np.all(np.isfinite(out), axis=1)):
        if e - s > minimo:
            out[s:e] = filtfilt(b, a, out[s:e], axis=0)
    return out


def clean(points, fps):
    """Buchi brevi colmati, tratti corti scartati, poi passa-basso.

    Restituisce (punti puliti, quali sono stati interpolati).
    """
    if not fps:
        return dict(points), set()
    max_gap = max(1, int(round(MAX_GAP_S * fps)))
    out, interpolati = {}, set()
    for nome, v in points.items():
        prima = np.all(np.isfinite(v), axis=1).sum()
        colmato = interpolate_gaps(v, max_gap)
        if np.all(np.isfinite(colmato), axis=1).sum() > prima:
            interpolati.add(nome)
        out[nome] = lowpass(drop_short_runs(colmato, MIN_RUN), fps)
    return out, interpolati


# --- Terna del laboratorio ---
def world_frame(points):
    """Terna di riferimento: verticale del laboratorio e destra del soggetto.

    Serve come prossimale per l'inclinazione del tronco. La verticale si prende
    dall'asse del laboratorio piu' allineato all'asse del corpo: e' l'assunzione
    che la camera o il laboratorio siano in bolla, e va dichiarata.
    """
    def mediana(nome):
        v = points.get(nome)
        if v is None:
            return None
        f = v[np.all(np.isfinite(v), axis=1)]
        return np.median(f, axis=0) if len(f) else None

    hip_l, hip_r = mediana("hip_l"), mediana("hip_r")
    if hip_l is None or hip_r is None:
        return np.eye(3), ["world_frame_missing"]
    hip_c = (hip_l + hip_r) / 2.0
    # L'alto viene dal tronco quando c'e'; un'acquisizione dei soli arti
    # inferiori non ha le spalle, ma il bacino sopra le ginocchia basta.
    alto = None
    for sopra, sotto in ((mediana("shoulder_c"), hip_c), (hip_c, mediana("knee_c")),
                         (hip_c, mediana("ankle_l"))):
        if sopra is None or sotto is None:
            continue
        v = _unit(sopra - sotto)
        if np.all(np.isfinite(v)):
            alto = v
            break
    if alto is None:
        return np.eye(3), ["world_frame_missing"]
    destra = _unit(hip_r - hip_l)
    k = int(np.argmax(np.abs(alto)))
    verticale = np.zeros(3)
    verticale[k] = np.sign(alto[k])
    destra = _unit(destra - np.dot(destra, verticale) * verticale)
    return np.vstack([np.cross(verticale, destra), verticale, destra]), []


# --- Terne dei segmenti ---
def _utilizzabile(punti, chiavi):
    """Vero se quei punti sono validi in almeno `MIN_COVERAGE` dei frame.

    La presenza della chiave non basta. Un punto derivato esiste nel dizionario
    anche quando cio' da cui si ricava non e' mai stato visto: `hip_c` di una
    ripresa che taglia sotto le spalle e' un vettore di soli NaN. Provandolo
    comunque, la prima ricetta restituiva NaN invece di fallire, e nessun
    ripiego veniva mai raggiunto -- il segmento cadeva pur avendo un'alternativa
    del tutto valida piu' in basso nell'elenco.
    """
    for k in chiavi:
        v = punti.get(k)
        if v is None:
            return False
        if np.mean(np.isfinite(np.asarray(v, dtype=float)).all(axis=1)) < anatomy.MIN_COVERAGE:
            return False
    return True


def _roll_axis(points, nome, spec, destra_rif, estremi):
    """(terzo asse, indice della ricetta usata). (None, None) se nessuna vale.

    L'indice dice quale ripiego e' stato usato: serve a chi vuole sapere se la
    rotazione e' stata fissata dalla ricetta migliore o da una di riserva.
    """
    for i, roll in enumerate(anatomy.roll_recipes(nome)):
        if not _utilizzabile(points, anatomy.roll_inputs(roll)):
            continue
        v = _roll_single(points, estremi, roll, destra_rif)
        if v is not None:
            return v, i
    return None, None


def _roll_single(points, estremi, roll, destra_rif):
    if roll[0] == "plane":
        a, b, c = (points[k] for k in roll[1:])
        normale = np.cross(a - b, c - b)
        # |n| = |ab||cb| sin(theta): sotto MIN_PLANE_DEG i tre punti sono
        # allineati, il piano non esiste e l'asse si prende dal soggetto.
        scala = np.linalg.norm(a - b, axis=1) * np.linalg.norm(c - b, axis=1)
        degenere = np.linalg.norm(normale, axis=1) < (
            np.sin(np.radians(MIN_PLANE_DEG)) * np.maximum(scala, 1e-9))
        z = np.where(degenere[:, None], destra_rif, normale)
        # l'asse di flessione va sempre verso destra del soggetto: e' il segno
        # in JOINTS a distinguere i due lati, non l'orientamento della terna
        return np.where(np.sum(z * destra_rif, axis=1, keepdims=True) < 0, -z, z)
    if roll[0] == "toward":
        p = points.get(roll[1])
        return None if p is None else p - points[estremi[0]]
    if roll[0] == "direction":
        a, b = points.get(roll[1]), points.get(roll[2])
        return None if a is None or b is None else b - a
    return None


def segment_frames(points, R_world):
    """Terne per segmento. Restituisce (terne, provenienza, assenze, avvisi).

    Ogni terna ha righe [X anteriore, Y lungo, Z destra].
    """
    n = len(next(iter(points.values()))) if points else 0
    destra_rif = np.tile(R_world[2], (n, 1))
    terne, prov, assenze, avvisi, rimandati = {}, {}, {}, [], []

    for nome, spec in anatomy.SEGMENTS.items():
        mancanti = [k for k in anatomy.segment_inputs(nome) if k not in points]
        if mancanti:
            assenze[nome] = (anatomy.MISSING_INPUTS, ", ".join(mancanti))
            continue
        ricette = anatomy.roll_recipes(nome)
        if ricette and ricette[0][0] == "borrow":
            rimandati.append((nome, spec))
            continue
        estremi = next((c for c in anatomy.segment_ends_options(nome)
                        if _utilizzabile(points, c)), None)
        if estremi is None:
            assenze[nome] = (anatomy.MISSING_INPUTS, "estremi del segmento")
            continue
        a, b = (points[k] for k in estremi)
        asse = spec["axis"]
        terza, ricetta = _roll_axis(points, nome, spec, destra_rif, estremi)
        if terza is None:
            assenze[nome] = (anatomy.MISSING_INPUTS, "riferimento per la rotazione")
            continue
        if asse == "long":                       # Y lungo il segmento
            y = _unit(b - a)
            z = _unit(terza - np.sum(terza * y, axis=1, keepdims=True) * y)
            x = np.cross(y, z)
        elif asse == "lateral":                  # Z fra i due estremi (cinti)
            z = _unit(b - a)
            y = _unit(terza - np.sum(terza * z, axis=1, keepdims=True) * z)
            x = np.cross(y, z)
        else:                                    # X in avanti (piede)
            x = _unit(b - a)
            y = _unit(terza - np.sum(terza * x, axis=1, keepdims=True) * x)
            z = np.cross(x, y)
        R = np.stack([x, y, z], axis=1)
        R[~np.all(np.isfinite(R), axis=(1, 2))] = np.nan
        terne[nome] = R
        prov[nome] = spec["provenance"]

    for nome, spec in rimandati:
        sorgente = anatomy.roll_recipes(nome)[0][1]
        padre = terne.get(sorgente)
        if padre is None:
            assenze[nome] = (anatomy.MISSING_INPUTS, sorgente)
            continue
        a, b = (points[k] for k in spec["ends"])
        y = _unit(b - a)
        z = padre[:, 2, :]
        z = _unit(z - np.sum(z * y, axis=1, keepdims=True) * y)
        R = np.stack([np.cross(y, z), y, z], axis=1)
        R[~np.all(np.isfinite(R), axis=(1, 2))] = np.nan
        terne[nome] = R
        prov[nome] = spec["provenance"]

    # le due clavicole condividono l'orientamento del cinto scapolare
    for lato in ("l", "r"):
        altro = "r" if lato == "l" else "l"
        if f"clavicle_{lato}" not in terne and f"clavicle_{altro}" in terne:
            terne[f"clavicle_{lato}"] = terne[f"clavicle_{altro}"]
            prov[f"clavicle_{lato}"] = anatomy.SEGMENTS[f"clavicle_{lato}"]["provenance"]
            assenze.pop(f"clavicle_{lato}", None)

    terne[anatomy.WORLD] = np.tile(R_world, (n, 1, 1))
    prov[anatomy.WORLD] = anatomy.DERIVED
    return terne, prov, assenze, avvisi


# --- Quaternioni e derivate ---
def quaternions(R):
    """Terne -> quaternioni [x, y, z, w], con segno continuo nel tempo."""
    n = len(R)
    q = np.full((n, 4), np.nan)
    valid = np.all(np.isfinite(R), axis=(1, 2))
    if valid.any():
        # le righe di R sono gli assi, quindi R e' la matrice mondo->locale:
        # la rotazione del segmento e' la sua trasposta
        q[valid] = Rotation.from_matrix(np.transpose(R[valid], (0, 2, 1))).as_quat()
    ordine = np.where(valid)[0]
    for k in range(1, len(ordine)):
        a, b = ordine[k - 1], ordine[k]
        if np.dot(q[a], q[b]) < 0:
            q[b] *= -1
    return q


def fill_quaternions(q, max_gap):
    """SLERP sui buchi brevi: interpolare le componenti darebbe rotazioni invalide."""
    out = q.copy()
    valid = np.all(np.isfinite(out), axis=1)
    mancanti = np.where(~valid)[0]
    if mancanti.size == 0 or valid.sum() < 2:
        return out
    for run in np.split(mancanti, np.where(np.diff(mancanti) != 1)[0] + 1):
        s, e = run[0], run[-1]
        if s == 0 or e >= len(out) - 1 or (e - s + 1) > max_gap:
            continue
        if not (valid[s - 1] and valid[e + 1]):
            continue
        chiavi = Rotation.from_quat([out[s - 1], out[e + 1]])
        t = np.linspace(0, 1, e - s + 3)[1:-1]
        out[s:e + 1] = Slerp([0, 1], chiavi)(t).as_quat()
    return out


def angular_kinematics(q, fps):
    """Velocita' angolare dalla derivata del quaternione, e accelerazione da essa."""
    n = len(q)
    omega = np.full((n, 3), np.nan)
    if not fps:
        return omega, omega.copy()
    dt = 1.0 / fps
    buono = np.all(np.isfinite(q), axis=1)
    for i in range(1, n):
        if not (buono[i] and buono[i - 1]):
            continue
        a, b = q[i - 1], q[i].copy()
        if np.dot(a, b) < 0:
            b = -b
        omega[i] = (Rotation.from_quat(a).inv() * Rotation.from_quat(b)).as_rotvec() / dt
    alpha = np.full_like(omega, np.nan)
    ok = np.all(np.isfinite(omega), axis=1)
    alpha[1:][ok[1:] & ok[:-1]] = np.diff(omega, axis=0)[ok[1:] & ok[:-1]] / dt
    return omega, alpha


# --- Angoli articolari ---
def despike(series, limite):
    """Toglie i frame isolati oltre il limite fisiologico rispetto a entrambi i vicini."""
    out = np.asarray(series, dtype=float).copy()
    a, b, c = out[:-2], out[1:-1], out[2:]
    sospetti = (np.abs(b - a) > limite) & (np.abs(b - c) > limite)
    out[1:-1][np.nan_to_num(sospetti, nan=False).astype(bool)] = np.nan
    return out


def implausible_rate(series, limite):
    """Quota di variazioni oltre il limite: quanto e' gesto e quanto rumore."""
    v = np.asarray(series, dtype=float)
    f = v[np.isfinite(v)]
    return None if f.size < 2 else float(np.mean(np.abs(np.diff(f)) > limite))


def joint_angles(terne, fps, esistenti=None):
    """Angoli per (giunto, piano). Restituisce (angoli, provenienza, assenze).

    Gli angoli che la sorgente fornisce gia' non vengono ricalcolati.
    """
    angoli = dict(esistenti or {})
    prov = {k: anatomy.MEASURED for k in angoli}
    assenze = {}
    limite = MAX_ANGULAR_DEG_S / fps if fps else 90.0

    for nome, jd in anatomy.JOINTS.items():
        prossimale, distale = jd["proximal"], jd["distal"]
        if prossimale not in terne or distale not in terne:
            mancanti = [s for s in (prossimale, distale) if s not in terne]
            for piano in anatomy.PLANES:
                if (nome, piano) not in angoli:
                    assenze[(nome, piano)] = (anatomy.MISSING_INPUTS, ", ".join(mancanti))
            continue
        if all((nome, p) in angoli for p in anatomy.PLANES):
            continue
        Rp, Rd = terne[prossimale], terne[distale]
        ang = np.full((len(Rp), 3), np.nan)
        valid = (np.all(np.isfinite(Rp), axis=(1, 2))
                 & np.all(np.isfinite(Rd), axis=(1, 2)))
        if valid.any():
            # cambio di base dal distale al prossimale: e' la rotazione che il
            # distale ha subito rispetto al prossimale
            rel = np.matmul(Rp[valid], np.transpose(Rd[valid], (0, 2, 1)))
            ang[valid] = Rotation.from_matrix(rel).as_euler(
                anatomy.EULER_SEQUENCE, degrees=True)
        ang *= np.asarray(jd["signs"], dtype=float)
        for k, piano in enumerate(anatomy.PLANES):
            if (nome, piano) in angoli:
                continue
            if anatomy.plane_provenance(nome, piano) != anatomy.DERIVED:
                # il piano non e' osservabile da queste terne: registrarlo come
                # valore darebbe un numero credibile ma senza significato
                if nome == "spine" and piano == "sagittal":
                    dettaglio = ("il bacino prende il suo rollio dalla stessa "
                                 "direzione che definisce l'asse del tronco: "
                                 "la flessione relativa e' zero per costruzione")
                else:
                    dettaglio = (f"{jd['proximal']}/{jd['distal']}: la rotazione "
                                 "attorno all'asse lungo dipende da un'assunzione "
                                 "di piano")
                assenze[(nome, piano)] = (anatomy.NOT_OBSERVABLE, dettaglio)
                continue
            angoli[(nome, piano)] = despike(ang[:, k], limite)
            prov[(nome, piano)] = anatomy.plane_provenance(nome, piano)
    return angoli, prov, assenze


# --- Ingresso unico ---
def compute(letto):
    """Da quanto letto dalla sorgente al pacchetto completo per il C3D."""
    meta = dict(letto["meta"])
    fps = meta.get("fps")
    max_gap = max(1, int(round(MAX_GAP_S * fps))) if fps else 1

    punti, interpolati = clean(letto["points"], fps)
    R_world, avvisi = world_frame(punti)
    terne, prov_seg, assenze_seg, av2 = segment_frames(punti, R_world)
    avvisi += av2

    segmenti = {}
    for nome, R in terne.items():
        if nome == anatomy.WORLD:
            continue
        q = fill_quaternions(quaternions(R), max_gap)
        omega, alpha = angular_kinematics(q, fps)
        segmenti[nome] = {"quat": q, "omega": omega, "alpha": alpha,
                          "provenance": prov_seg[nome]}

    angoli, prov_ang, assenze_ang = joint_angles(terne, fps, letto.get("angles"))

    assenze = {f"point:{k}": v for k, v in letto["absent"].items()}
    assenze.update({f"segment:{k}": v for k, v in assenze_seg.items()})
    assenze.update({f"angle:{g}:{p}": v for (g, p), v in assenze_ang.items()})
    if not fps:
        assenze["kinematics:angular_velocity"] = (anatomy.NO_TIME_BASE, "fps assente")

    limite = MAX_ANGULAR_DEG_S / fps if fps else 90.0
    # I centri costruiti (bacino) servono a costruire le terne ma non sono punti
    # da riconoscere: fuori restano i sedici canonici.
    punti = {k: v for k, v in punti.items() if k in anatomy.POINTS}
    prov_punti = {k: (anatomy.FILLED if k in interpolati else anatomy.MEASURED)
                  for k in punti}
    return {
        "points": punti, "point_provenance": prov_punti,
        "segments": segmenti,
        "angles": angoli, "angle_provenance": prov_ang,
        "absences": assenze,
        "coverage": {k: float(np.all(np.isfinite(v), axis=1).mean())
                     for k, v in punti.items()},
        "implausible": {k: implausible_rate(v, limite) for k, v in angoli.items()},
        "world_frame": np.asarray(R_world).tolist(),
        # Gli appoggi marcati dalla sorgente passano di qui senza essere toccati:
        # la segmentazione li preferisce a qualunque cosa sappia ricavare.
        "events": letto.get("events") or {},
        "warnings": avvisi,
        "meta": meta,
    }
