"""Processazione: da una sorgente al C3D canonico pronto per l'ingest.

    python -m src.conversion.process --elenco
    python -m src.conversion.process DatiGrezzi/running_20260220_121922
    python -m src.conversion.process DatiGrezzi --tutte --uscita DatiElaborati

Tre passi nell'ordine: lettura della sorgente e riconoscimento dei punti,
calcolo di terne, quaternioni, angoli e cicli, scrittura del C3D canonico nella
cartella di uscita. Al termine stampa un riepilogo della sessione convertita.

Prerequisiti: numpy, scipy ed ezc3d; una sorgente leggibile sotto la cartella
dei dati grezzi, cartella running_* oppure file .c3d.
"""
import argparse
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from src.conversion import (anatomy, angles, canonical,
                            readers, segmentation)

# --- Cartelle di lavoro predefinite ---
RADICE_DEFAULT = "DatiGrezzi"
USCITA_DEFAULT = "DatiElaborati"


# --- Conversione di una sessione ---
# Normalizzazione del gesto dichiarato, calcolo di angoli e cicli, scrittura del
# file e riepilogo numerico di cio' che ne e' uscito.
def _gesto_canonico(dichiarato):
    if not dichiarato:
        return None
    canonico = anatomy.canonical_sport(dichiarato)
    if canonico is None:
        raise ValueError(f"gesto non riconosciuto: {dichiarato!r}. "
                         f"Ammessi: {', '.join(anatomy.SPORTS)}")
    return canonico


def analizza(letto):
    risultato = angles.compute(letto)
    risultato["cycles"] = segmentation.detect(risultato) or {}
    return risultato


def process(sorgente, cartella_uscita, sport=None):
    _gesto_canonico(sport)
    letto = readers.read(sorgente)
    dichiarato = sport or letto["meta"].get("sport")
    if dichiarato:
        letto["meta"]["sport"] = _gesto_canonico(dichiarato)
    risultato = analizza(letto)
    nome = risultato["meta"]["source"]
    os.makedirs(cartella_uscita, exist_ok=True)
    percorso = canonical.write(risultato, os.path.join(cartella_uscita, f"{nome}.c3d"))

    misurati = sum(1 for v in risultato["angle_provenance"].values()
                   if v == anatomy.MEASURED)
    return percorso, {
        "sessione": nome,
        "sport": risultato["meta"].get("sport"),
        "formato": risultato["meta"]["source_format"],
        "frame": risultato["meta"]["n_frames"],
        "fps": risultato["meta"]["fps"],
        "punti": len(risultato["points"]),
        "segmenti": len(risultato["segments"]),
        "angoli": len(risultato["angles"]),
        "angoli_misurati": misurati,
        "assenze": len(risultato["absences"]),
        "cicli": len(risultato["cycles"].get("frames") or []),
        "metodo_cicli": risultato["cycles"].get("method"),
        "avvisi": risultato["warnings"] + (risultato["cycles"].get("warnings") or []),
        "copertura_media": (float(np.mean(list(risultato["coverage"].values())))
                            if risultato["coverage"] else 0.0),
    }


# --- Riga di comando ---
# Stampa del riepilogo, elenco delle sorgenti disponibili e conversione di una
# sola sorgente o di tutte quelle trovate sotto la cartella indicata.
def _stampa(r):
    print(f"  {r['sessione']}")
    print(f"     {r['formato']}, {r['frame']} frame"
          + (f", {r['sport']}" if r.get("sport") else "")
          + (f" @ {r['fps']:g} Hz" if r['fps'] else " (nessuna base temporale)"))
    print(f"     {r['punti']}/{len(anatomy.POINTS)} punti "
          f"(copertura media {r['copertura_media']:.0%}), "
          f"{r['segmenti']} segmenti, {r['angoli']} angoli"
          + (f" di cui {r['angoli_misurati']} gia' nella sorgente"
             if r['angoli_misurati'] else ""))
    print(f"     {r['cicli']} cicli"
          + (f" ({r['metodo_cicli']})" if r.get("metodo_cicli") else " (nessuno)"))
    print(f"     {r['assenze']} assenze registrate"
          + (f", avvisi: {', '.join(r['avvisi'])}" if r['avvisi'] else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sorgente", nargs="?", default=RADICE_DEFAULT,
                    help="una sessione: cartella running_* o file .c3d. "
                         "Per convertire tutto cio' che sta sotto una cartella "
                         "serve --tutte")
    ap.add_argument("--tutte", action="store_true",
                    help="converte tutte le sorgenti trovate sotto la cartella")
    ap.add_argument("--elenco", action="store_true",
                    help="elenca le sorgenti disponibili senza convertire")
    ap.add_argument("--uscita", default=USCITA_DEFAULT, help="cartella di destinazione")
    ap.add_argument("--sport", help="dichiara il gesto quando la sorgente non lo sa "
                                    "(walking, running, ...)")
    args = ap.parse_args(argv)

    if args.elenco:
        for p in readers.discover(args.sorgente):
            print(os.path.basename(p))
        return 0

    sorgenti = (readers.discover(args.sorgente) if args.tutte else [args.sorgente])
    if not sorgenti:
        print(f"nessuna sorgente sotto {args.sorgente}", file=sys.stderr)
        return 1

    fatte, fallite = 0, []
    for s in sorgenti:
        try:
            percorso, riepilogo = process(s, args.uscita, sport=args.sport)
        except Exception as e:
            fallite.append((os.path.basename(s), f"{type(e).__name__}: {e}"))
            continue
        fatte += 1
        _stampa(riepilogo)
        print(f"     -> {percorso}")

    if fallite:
        print(f"\n{len(fallite)} non convertite:", file=sys.stderr)
        for nome, motivo in fallite:
            print(f"  {nome}: {motivo}", file=sys.stderr)
    print(f"\n{fatte}/{len(sorgenti)} convertite in {args.uscita}")
    return 0 if fatte else 1


if __name__ == "__main__":
    raise SystemExit(main())
