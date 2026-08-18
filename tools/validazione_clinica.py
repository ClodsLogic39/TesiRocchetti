"""Confronta gli angoli calcolati dal progetto con quelli di un modello clinico.

I C3D di laboratorio con marker set Plug-in Gait portano sia i punti sia le
uscite del modello. Si possono quindi ricalcolare gli angoli dai soli punti e
misurare lo scarto rispetto alla verita' che sta nello stesso file: e' l'unica
condizione in cui il calcolo del progetto e' verificabile, perche' sui dati
MediaPipe non esiste niente con cui confrontarlo.

Attenzione a come il convertitore tratta questi file: quando la sorgente porta
gia' gli angoli, li ricopia invece di ricalcolarli (provenienza `measured`).
Il confronto qui sotto svuota apposta quel campo, altrimenti misurerebbe un
valore contro se stesso.

    python -m tools.validazione_clinica DatiGrezzi/Ref_1 DatiGrezzi/Ref_2
"""
import argparse
import os
import sys

import numpy as np

from src.conversion import angles, readers

# Il fit lineare separa due cose diverse: una convenzione diversa (segno,
# offset, guadagno) lascia un residuo piccolo perche' la forma della curva e'
# la stessa; un modello diverso lascia un residuo grande, e nessuna costante
# puo' ripararlo.
MIN_CAMPIONI = 30


def confronta_file(percorso):
    """{(giunto, piano): metriche} confrontando ricalcolo e modello clinico."""
    letto = readers.read(percorso)
    riferimento = dict(letto.get("angles") or {})
    if not riferimento:
        return {}
    senza = dict(letto)
    senza["angles"] = {}          # costringe il ricalcolo dai punti
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


def riepilogo(radici, per_condizione=0):
    """Mediana delle metriche per (giunto, piano) sui file trovati.

    `per_condizione` limita quanti file prendere da ogni sottocartella: le
    mediane si stabilizzano con pochi file per condizione, e un confronto che
    dura mezzo minuto non viene eseguito. Zero significa tutti.
    """
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
