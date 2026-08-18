"""Caricamento dei riferimenti per gli alert e valutazione di una sessione.

I riferimenti sono valori fissi presi da letteratura, in `reference_ranges.json`.
Non vengono calcolati dalle sessioni caricate: un riferimento che nasce dagli
stessi dati che deve giudicare non e' un riferimento, e' un'eco.

Due livelli, con significati diversi:
  anatomical  limiti articolari validi per chiunque, indipendenti dal gesto.
              Uscirne significa movimento fuori dai limiti fisiologici o errore
              di misura.
  sport       intervalli attesi per un gesto specifico. Uscirne non e' di per
              se' pericoloso: e' un'anomalia da guardare.

L'alert dice solo *cosa* si scosta e *da quale riferimento*. Non dice cosa fare:
quello e' compito di chi legge, umano o agente che sia.
"""
import json
import os

import numpy as np

PERCORSO = os.path.join(os.path.dirname(__file__), "reference_ranges.json")

# I riferimenti parlano di giunti senza lato: valgono per entrambi.
_LATI = ("l", "r")


def carica(percorso=None):
    with open(percorso or PERCORSO) as f:
        return json.load(f)


def righe(dati=None):
    """Espande il file in righe pronte per `reference_ranges`, una per lato."""
    dati = dati or carica()
    fonti = dati["sources"]
    out = []

    def espandi(voce, scope, sport):
        fonte = fonti[voce["source"]]
        base = voce["joint"]
        # trunk, spine e neck sono giunti centrali, senza lato.
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
def metriche(serie):
    """Metriche confrontabili con i riferimenti, da una serie di angoli."""
    v = np.asarray(serie, dtype=float)
    f = v[np.isfinite(v)]
    if f.size < 2:
        return {}
    return {"peak_max": float(f.max()), "peak_min": float(f.min()),
            "rom": float(f.max() - f.min())}


# Oltre questo multiplo dell'ampiezza dell'intervallo, lo scostamento non
# descrive piu' un corpo: un gomito che iperestende di 137 gradi non e' un
# atleta fuori norma, e' una misura rotta. Chiamarlo `critical` con una fonte
# clinica a fianco darebbe dignita' medica a un errore di tracciamento.
FATTORE_IMPLAUSIBILE = 2.0
DATA_QUALITY = "data_quality"


def _gravita(riferimento, lo, hi, scarto):
    """La gravita' dichiarata, o `data_quality` se il valore e' impossibile."""
    ampiezza = None
    if lo is not None and hi is not None:
        ampiezza = abs(hi - lo)
    elif lo is not None or hi is not None:
        # Intervallo aperto da un lato: l'unico riferimento di scala e' il
        # limite stesso.
        ampiezza = abs(lo if lo is not None else hi)
    if ampiezza and scarto > FATTORE_IMPLAUSIBILE * ampiezza:
        return DATA_QUALITY
    return riferimento["severity"]


def valuta(osservate, riferimenti, coverage=None, provenance=None,
           min_coverage=0.30):
    """Confronta le metriche osservate con i riferimenti applicabili.

    `osservate` = {(giunto, piano): {metrica: valore}}
    Restituisce la lista degli scostamenti, ciascuno con il riferimento che lo
    ha generato. Un angolo visto in troppo pochi frame non viene giudicato: il
    suo massimo e' il massimo di un sottoinsieme, non del gesto.
    """
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


def applicabili(riferimenti, sport):
    """I riferimenti che valgono per questa sessione.

    Quelli anatomici valgono sempre; quelli di gesto solo se lo sport e'
    dichiarato e riconosciuto. Una sessione senza sport riceve comunque i
    controlli anatomici: quelli non dipendono da cosa si sta facendo.
    """
    return [r for r in riferimenti
            if r["scope"] == "anatomical" or (sport and r["sport"] == sport)]
