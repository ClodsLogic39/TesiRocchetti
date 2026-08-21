"""Confronta gli angoli calcolati dal progetto con quelli di un modello clinico.

I C3D di laboratorio con marker set Plug-in Gait contengono sia i punti sia le
uscite del modello. Lo strumento ricalcola gli angoli a partire dai soli punti,
dopo aver svuotato quelli che la sorgente porta gia', e misura lo scarto rispetto
alle uscite del modello presenti nello stesso file. Stampa un riepilogo per
giunto e piano, con la possibilita' di raggrupparlo per condizione di ripresa.

Uso:  python -m tools.validazione_clinica <cartella con file .c3d>

Prerequisiti: numpy ed ezc3d; file C3D che contengano insieme i marker e le
uscite del modello clinico.
"""
import argparse
import os
import sys

import numpy as np

from src.conversion import angles, readers

# --- Soglia di confronto ---
# Numero minimo di fotogrammi validi perche' il confronto su un piano venga
# considerato.
MIN_CAMPIONI = 30


# --- Confronto di un file ---
# Rilegge la sorgente, azzera gli angoli che porta gia' calcolati, li ricalcola
# dai punti e misura lo scarto rispetto a quelli del modello.
def confronta_file(percorso):
    letto = readers.read(percorso)
    riferimento = dict(letto.get("angles") or {})
    if not riferimento:
        return {}
    senza = dict(letto)
    senza["angles"] = {}
    calcolato = angles.compute(senza)["angles"]

    out = {}
    for chiave, serie in riferimento.items():
        if chiave not in calcolato:
            continue
        a = np.asarray(serie, dtype=float)
        b = np.asarray(calcolato[chiave], dtype=float)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < MIN_CAMPIONI or np.std(b[ok]) < 1e-6:
            continue
        A, B = a[ok], b[ok]
        scala, offset = np.polyfit(B, A, 1)
        residuo = A - (scala * B + offset)
        out[chiave] = {
            "corr": float(np.corrcoef(A, B)[0, 1]),
            "rom_clinico": float(A.max() - A.min()),
            "rom_calcolato": float(B.max() - B.min()),
            "scala": float(scala),
            "offset": float(offset),
            "residuo_rms": float(np.sqrt((residuo ** 2).mean())),
        }
    return out


# --- Riepilogo su piu' file ---
# Raccoglie gli scarti di tutti i file trovati e li aggrega per giunto e piano.
def riepilogo(radici, per_condizione=0):
    raccolta = {}
    for radice in radici:
        presi = {}
        for percorso in readers.discover(radice):
            if not percorso.lower().endswith(".c3d"):
                continue
            condizione = os.path.basename(os.path.dirname(percorso))
            if per_condizione:
                presi[condizione] = presi.get(condizione, 0) + 1
                if presi[condizione] > per_condizione:
                    continue
            try:
                misure = confronta_file(percorso)
            except Exception:
                continue
            for chiave, v in misure.items():
                raccolta.setdefault(chiave, []).append(v)

    out = {}
    for chiave, valori in raccolta.items():
        out[chiave] = {
            "n_file": len(valori),
            "corr": float(np.median([abs(v["corr"]) for v in valori])),
            "scala": float(np.median([v["scala"] for v in valori])),
            "residuo_rms": float(np.median([v["residuo_rms"] for v in valori])),
            "rom_clinico": float(np.median([v["rom_clinico"] for v in valori])),
            "rom_calcolato": float(np.median([v["rom_calcolato"] for v in valori])),
        }
    return out


# --- Riga di comando ---
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("radici", nargs="+", help="cartelle con C3D del modello clinico")
    args = ap.parse_args(argv)

    presenti = [r for r in args.radici if os.path.isdir(r)]
    if not presenti:
        print("nessuna cartella trovata", file=sys.stderr)
        return 1
    tabella = riepilogo(presenti)
    if not tabella:
        print("nessun file con uscite di modello", file=sys.stderr)
        return 1

    print(f"{'giunto/piano':24s} {'file':>5s} {'|corr|':>7s} {'scala':>7s} "
          f"{'ROM clin':>9s} {'ROM calc':>9s} {'residuo':>8s}")
    for (giunto, piano), v in sorted(tabella.items(), key=lambda kv: kv[1]["residuo_rms"]):
        print(f"{giunto + '/' + piano:24s} {v['n_file']:5d} {v['corr']:7.3f} "
              f"{v['scala']:+7.2f} {v['rom_clinico']:9.1f} {v['rom_calcolato']:9.1f} "
              f"{v['residuo_rms']:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
