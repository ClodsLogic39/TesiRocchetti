"""Encoder degli atomi di ciclo: la forma di una curva articolare come vettore.

Un atomo = (ciclo, giunto, piano). Il vettore descrive la sola **forma** della
curva: il ciclo viene ricampionato a lunghezza fissa, gli si toglie la media e lo
si porta a norma unitaria.

Perche' solo la forma, e non i valori. Gli angoli che produciamo portano
scostamenti propri del soggetto e del modello: la caviglia non ha una prova
statica di calibrazione, il bacino nasce dai centri articolari e non dalle spine
iliache. Confrontare i valori assoluti fra due persone misurerebbe la differenza
fra i due modelli, non fra i due gesti. L'andamento nel tempo, invece, e'
confrontabile: e' quello che descrive *come* si muove un giunto.

Ampiezza e livello non spariscono: viaggiano come colonne numeriche accanto al
vettore, dove si confrontano con operatori ordinari e solo quando ha senso.

Su vettori a media nulla e norma unitaria la distanza coseno di pgvector vale
1 meno la correlazione di Pearson fra le due curve:
    0  stesso andamento     1  nessuna relazione     2  andamento opposto
"""
import numpy as np

from ..conversion import anatomy

CYCLE_POINTS = 101        # ciclo normalizzato 0-100%, convenzione del dominio
ATOM_DIM = 32             # dimensione del vettore, deve combaciare con schema.sql
# v2: la forma e' allineata di fase (vedi `atom`). Gli atomi v1 restano in
# tabella ma nessuna query li vede, perche' filtrano tutte su encoder_version:
# per usarli di nuovo basta ricaricare le sessioni.
ENCODER_VERSION = "shape-v2"

MIN_COVERAGE = 0.80       # sotto questa quota di frame validi il ciclo non basta
_MIN_FINITE = 8           # campioni finiti minimi per ricampionare
_MIN_NORM = 1e-9          # curva costante: la forma non e' definita


def _nan(values):
    """Serie dal database (con None) o da numpy -> array float con NaN."""
    return np.asarray([np.nan if v is None else v for v in values], dtype=float)


def coverage(series):
    a = np.asarray(series, dtype=float)
    return float(np.isfinite(a).mean()) if a.size else 0.0


def normalized_cycle(series, points=CYCLE_POINTS):
    """Curva del ciclo ricampionata a 0-100%. None se i dati non bastano.

    L'interpolazione lavora sui soli campioni validi e non esce dal loro
    intervallo: un buco interno viene attraversato, ma un ciclo che comincia o
    finisce nel vuoto non viene esteso per inventare i bordi.
    """
    arr = np.asarray(series, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() < _MIN_FINITE:
        return None
    idx = np.where(mask)[0]
    return np.interp(np.linspace(idx[0], idx[-1], points), idx, arr[mask])


def atom(series):
    """Serie di un ciclo -> (forma, rom, picco, valle). None se degenere.

    La forma viene ruotata perche' cominci dal suo minimo. Un ciclo e' chiuso --
    finisce dove ricomincia -- quindi la rotazione non altera il gesto, ma lo
    rende indipendente dal punto in cui la registrazione e' partita. Senza,
    poiche' la forma e' centrata sulla media e normalizzata, la distanza coseno
    vale `1 - correlazione`: due passate identiche riprese a mezzo ciclo di
    distanza risultavano anti-correlate, cioe' massimamente diverse.
    """
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
        # curva piatta: un vettore nullo non e' indicizzabile per coseno e non
        # significa "simile a tutto", significa "nessuna forma"
        return None
    return forma / norma, picco - valle, picco, valle


def vector_literal(v):
    """Vettore -> letterale che pgvector accetta."""
    return "[" + ",".join(f"{x:.6g}" for x in v) + "]"


def atoms_for_cycle(angles, provenance, start, end):
    """Atomi di un ciclo. Restituisce (righe, scarti con il motivo)."""
    righe, scarti = [], {}
    for (giunto, piano), serie in angles.items():
        if giunto not in anatomy.JOINTS:
            continue
        tratto = _nan(serie)[start:end]     # `end` esclusivo, come in ingest
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
    """Scrive gli atomi di tutti i cicli. `cicli` = [(cycle_id, start, end)]."""
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


# --- Verifiche della proprieta' su cui si regge il confronto composito ---
def cosine(a, b):
    """Similarita' coseno fra due vettori gia' normalizzati."""
    return float(np.dot(a, b))


def concat_equals_mean(a1, a2, b1, b2):
    """cos(a1||a2, b1||b2) == media delle due similarita', se tutti unitari.

    E' la proprieta' che permette di confrontare piu' giunti insieme sommando le
    distanze dei singoli atomi invece di costruire un vettore composito: il
    database puo' quindi rispondere su qualunque insieme di giunti senza doverne
    indicizzare uno per combinazione.
    """
    concat_a = np.concatenate([a1, a2]) / np.sqrt(2.0)
    concat_b = np.concatenate([b1, b2]) / np.sqrt(2.0)
    return cosine(concat_a, concat_b), (cosine(a1, b1) + cosine(a2, b2)) / 2.0
