"""Riferimenti di letteratura per gli alert e valutazione di una sessione.

I valori sono fissi e vengono letti da `reference_ranges.json`, che sta accanto a
questo modulo. Sono divisi in due livelli:
  anatomical  limiti articolari validi per chiunque, indipendenti dal gesto
  sport       intervalli attesi per un gesto specifico

`righe` porta il file in una forma tabellare, `metriche` calcola le grandezze
osservate di una serie e `valuta` confronta le une con gli altri producendo gli
alert, ciascuno con gravita' e fonte. `frames_alert` risale da un alert ai
fotogrammi che lo producono, quando ne esistono.

Prerequisiti: numpy; il file `reference_ranges.json` presente nella cartella del
modulo.
"""
import json
import os

import numpy as np

# --- Sorgente dei riferimenti ---
# Percorso del file dei valori di letteratura e lati su cui un riferimento senza
# lato viene replicato.
PERCORSO = os.path.join(os.path.dirname(__file__), "reference_ranges.json")

_LATI = ("l", "r")


# --- Lettura dei riferimenti ---
# `carica` apre il file, `righe` lo appiattisce in una riga per giunto, piano,
# metrica e livello.
def carica(percorso=None):
    with open(percorso or PERCORSO) as f:
        return json.load(f)


def righe(dati=None):
    dati = dati or carica()
    fonti = dati["sources"]
    out = []

    def espandi(voce, scope, sport):
        fonte = fonti[voce["source"]]
        base = voce["joint"]
        nomi = [base] if base in ("trunk", "spine", "neck") else [
            f"{base}_{s}" for s in _LATI]
        for giunto in nomi:
            out.append({
                "scope": scope, "sport": sport, "joint": giunto,
                "plane": voce["plane"], "metric": voce["metric"],
                "min_value": voce.get("min"), "max_value": voce.get("max"),
                "unit": "deg", "severity": voce.get("severity", "warning"),
                "source": fonte["label"], "source_url": fonte.get("url"),
                "note": voce.get("note"),
            })

    for voce in dati.get("anatomical", []):
        espandi(voce, "anatomical", None)
    for sport, voci in dati.get("sports", {}).items():
        for voce in voci:
            espandi(voce, "sport", sport)
    return out


# --- Metriche osservate ---
# Le grandezze calcolate su una serie e confrontate con i riferimenti.
def metriche(serie):
    v = np.asarray(serie, dtype=float)
    f = v[np.isfinite(v)]
    if f.size < 2:
        return {}
    return {"peak_max": float(f.max()), "peak_min": float(f.min()),
            "rom": float(f.max() - f.min())}

# --- Valutazione e alert ---
# Gravita' di uno scostamento, confronto fra metriche osservate e riferimenti con
# gli avvisi legati a copertura e provenienza, e selezione dei riferimenti che si
# applicano a un dato gesto.
FATTORE_IMPLAUSIBILE = 2.0
DATA_QUALITY = "data_quality"


def _gravita(riferimento, lo, hi, scarto):
    ampiezza = None
    if lo is not None and hi is not None:
        ampiezza = abs(hi - lo)
    elif lo is not None or hi is not None:
        ampiezza = abs(lo if lo is not None else hi)
    if ampiezza and scarto > FATTORE_IMPLAUSIBILE * ampiezza:
        return DATA_QUALITY
    return riferimento["severity"]


def valuta(osservate, riferimenti, coverage=None, provenance=None,
           min_coverage=0.30):
    per_chiave = {}
    for r in riferimenti:
        per_chiave.setdefault((r["joint"], r["plane"], r["metric"]), []).append(r)

    fuori = []
    for (giunto, piano), valori in osservate.items():
        cop = (coverage or {}).get((giunto, piano))
        if cop is not None and cop < min_coverage:
            continue
        for metrica, valore in valori.items():
            for r in per_chiave.get((giunto, piano, metrica), []):
                lo, hi = r.get("min_value"), r.get("max_value")
                if lo is not None and valore < lo:
                    scarto = lo - valore
                elif hi is not None and valore > hi:
                    scarto = valore - hi
                else:
                    continue
                fuori.append({
                    "reference": r, "joint": giunto, "plane": piano,
                    "metric": metrica, "observed": valore,
                    "expected_min": lo, "expected_max": hi,
                    "deviation": float(scarto),
                    "severity": _gravita(r, lo, hi, float(scarto)),
                    "coverage": cop,
                    "provenance": (provenance or {}).get((giunto, piano)),
                })
    return fuori


# --- Dai valori ai fotogrammi ---
# Quali fotogrammi producono uno scostamento. La regola e' una sola: l'eccesso e'
# puntuale, la carenza no. Un valore che supera un limite lo supera in fotogrammi
# precisi, e quelli si indicano; una serie che non arriva a un minimo non ha
# fotogrammi colpevoli -- il difetto sta nell'intera serie, e indicare i due
# fotogrammi estremi vorrebbe dire additare proprio quelli in cui il soggetto si
# muove di piu'. In quel caso l'alert resta, ma dichiarato non localizzabile.
def frames_alert(serie, alert):
    """Maschera dei fotogrammi che producono l'alert, e se l'alert si localizza.

    Ritorna `(maschera, localizzabile)`: un array di booleani lungo quanto la
    serie e il flag che dice se quella maschera ha un senso. Con
    `localizzabile` falso la maschera e' tutta falsa e l'alert va riferito come
    proprieta' dell'intera serie, non di un istante.

    `alert` e' una voce come quelle prodotte da `valuta`: servono `metric`,
    `observed`, `expected_min` ed `expected_max`.
    """
    v = np.asarray(serie, dtype=float)
    vuota = np.zeros(v.shape, dtype=bool)
    finiti = np.isfinite(v)
    if not finiti.any():
        return vuota, False

    metrica = alert.get("metric")
    osservato = alert.get("observed")
    lo, hi = alert.get("expected_min"), alert.get("expected_max")
    if osservato is None:
        return vuota, False

    # Eccesso su un picco: i fotogrammi oltre la soglia sono esattamente quelli.
    if metrica == "peak_max" and hi is not None and osservato > hi:
        return finiti & (v > hi), True
    if metrica == "peak_min" and lo is not None and osservato < lo:
        return finiti & (v < lo), True

    # Eccesso di escursione: la producono i due fotogrammi degli estremi, che
    # sono i due campioni da cui il numero e' calcolato.
    if metrica == "rom" and hi is not None and osservato > hi:
        indici = np.flatnonzero(finiti)
        maschera = vuota.copy()
        maschera[indici[np.argmax(v[indici])]] = True
        maschera[indici[np.argmin(v[indici])]] = True
        return maschera, True

    # Tutto il resto e' una carenza: nessun fotogramma la produce.
    return vuota, False


def applicabili(riferimenti, sport):
    return [r for r in riferimenti
            if r["scope"] == "anatomical" or (sport and r["sport"] == sport)]
