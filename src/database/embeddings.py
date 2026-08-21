"""Atomi di ciclo: la forma di una curva articolare come vettore.

Un atomo corrisponde alla terna (ciclo, giunto, piano). Il ciclo viene
ricampionato a lunghezza fissa, ruotato in modo da partire dal suo minimo, quindi
portato a media nulla e norma unitaria: il vettore che ne risulta descrive la
sola forma della curva, mentre ampiezza e livello viaggiano accanto come colonne
numeriche.

Su vettori a media nulla e norma unitaria la distanza coseno vale 1 meno la
correlazione di Pearson fra le due curve: 0 stesso andamento, 1 nessuna
relazione, 2 andamento opposto.

Prerequisiti: numpy. La dimensione del vettore deve coincidere con quella
dichiarata in `schema.sql`, e `insert_for_cycles` richiede un cursore aperto su
una connessione in scrittura.
"""
import numpy as np

from ..conversion import anatomy

# --- Parametri dell'encoder ---
# Lunghezza del ciclo normalizzato, dimensione del vettore, versione dell'encoder
# su cui filtrano le query, copertura minima del ciclo e limiti sotto i quali la
# forma non e' definita.
CYCLE_POINTS = 101
ATOM_DIM = 32
ENCODER_VERSION = "shape-v2"

MIN_COVERAGE = 0.80
_MIN_FINITE = 8
_MIN_NORM = 1e-9


# --- Costruzione di un atomo ---
# Copertura della serie, ricampionamento del ciclo a lunghezza fissa, rotazione
# della curva sul suo minimo e riduzione a media nulla e norma unitaria, con la
# resa del vettore nella forma testuale attesa dal database.
def _nan(values):
    return np.asarray([np.nan if v is None else v for v in values], dtype=float)


def coverage(series):
    a = np.asarray(series, dtype=float)
    return float(np.isfinite(a).mean()) if a.size else 0.0


def normalized_cycle(series, points=CYCLE_POINTS):
    arr = np.asarray(series, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() < _MIN_FINITE:
        return None
    idx = np.where(mask)[0]
    return np.interp(np.linspace(idx[0], idx[-1], points), idx, arr[mask])


def atom(series):
    curva = normalized_cycle(series)
    if curva is None:
        return None
    picco, valle = float(curva.max()), float(curva.min())
    curva = np.roll(curva, -int(np.argmin(curva)))
    forma = np.interp(np.linspace(0, 1, ATOM_DIM),
                      np.linspace(0, 1, curva.size), curva)
    forma = forma - forma.mean()
    norma = float(np.linalg.norm(forma))
    if norma < _MIN_NORM:
        return None
    return forma / norma, picco - valle, picco, valle


def vector_literal(v):
    return "[" + ",".join(f"{x:.6g}" for x in v) + "]"


# --- Atomi di un ciclo e scrittura in tabella ---
# Produce un atomo per ogni coppia giunto/piano dell'intervallo indicato e li
# inserisce, ciclo per ciclo, nella tabella degli embedding.
def atoms_for_cycle(angles, provenance, start, end):
    righe, scarti = [], {}
    for (giunto, piano), serie in angles.items():
        if giunto not in anatomy.JOINTS:
            continue
        tratto = _nan(serie)[start:end]
        cop = coverage(tratto)
        if cop < MIN_COVERAGE:
            scarti[(giunto, piano)] = ("low_coverage", f"{cop:.0%} nel ciclo")
            continue
        costruito = atom(tratto)
        if costruito is None:
            scarti[(giunto, piano)] = ("flat_or_short", "curva piatta o troppo corta")
            continue
        forma, rom, picco, valle = costruito
        righe.append({
            "joint": giunto, "plane": piano, "shape": forma, "rom": rom,
            "peak": picco, "trough": valle, "coverage": cop,
            "provenance": provenance.get((giunto, piano), anatomy.DERIVED),
        })
    return righe, scarti


def insert_for_cycles(cur, dati, cicli):
    scritti = 0
    for cycle_id, inizio, fine in cicli:
        righe, _ = atoms_for_cycle(dati["angles"], dati["angle_provenance"],
                                   inizio, fine)
        for r in righe:
            cur.execute(
                """
                INSERT INTO cycle_atoms (cycle_id, joint, plane, encoder_version,
                                         shape, rom, peak, trough, provenance,
                                         coverage)
                VALUES (%s,%s,%s,%s,%s::vector,%s,%s,%s,%s,%s)
                """,
                (cycle_id, r["joint"], r["plane"], ENCODER_VERSION,
                 vector_literal(r["shape"]), r["rom"], r["peak"], r["trough"],
                 r["provenance"], r["coverage"]))
            scritti += 1
    return scritti


# --- Confronto fra vettori ---
# Distanza coseno fra due atomi e confronto fra la distanza di due atomi
# concatenati e la media delle distanze dei singoli.
def cosine(a, b):
    return float(np.dot(a, b))


def concat_equals_mean(a1, a2, b1, b2):
    concat_a = np.concatenate([a1, a2]) / np.sqrt(2.0)
    concat_b = np.concatenate([b1, b2]) / np.sqrt(2.0)
    return cosine(concat_a, concat_b), (cosine(a1, b1) + cosine(a2, b2)) / 2.0
