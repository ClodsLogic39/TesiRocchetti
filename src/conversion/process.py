#!/usr/bin/env python3
"""Processazione: da una sorgente al C3D canonico pronto per l'ingest.

    python -m src.conversion.process --elenco
    python -m src.conversion.process DatiGrezzi/running_20260220_121922
    python -m src.conversion.process DatiGrezzi --tutte --uscita DatiElaborati

Tre passi, nell'ordine:
  1. leggere       riconoscere i sedici punti fra le etichette della sorgente
  2. calcolare     terne, quaternioni e angoli, ma solo di cio' che manca
  3. scrivere      un C3D canonico con i dati e le annotazioni
"""
import argparse
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from src.conversion import (anatomy, angles, canonical,  # noqa: E402
                            readers, segmentation)

RADICE_DEFAULT = "DatiGrezzi"
USCITA_DEFAULT = "DatiElaborati"


def _gesto_canonico(dichiarato):
    """Il nome canonico del gesto, o un errore che elenca quelli ammessi."""
    if not dichiarato:
        return None
    canonico = anatomy.canonical_sport(dichiarato)
    if canonico is None:
        raise ValueError(f"gesto non riconosciuto: {dichiarato!r}. "
                         f"Ammessi: {', '.join(anatomy.SPORTS)}")
    return canonico


def analizza(letto):
    """Tutto cio' che il C3D canonico deve portare: angoli, poi cicli.

    Esiste come funzione a se' perche' il contenuto del file canonico va deciso
    in un posto solo. Chi scrive un C3D passando direttamente da `angles.compute`
    ne produce uno senza cicli, che `canonical.read` rifiuta.
    """
    risultato = angles.compute(letto)
    # I cicli si individuano qui, non al caricamento: sono parte del dato, non
    # del modo in cui lo si legge. Lo stesso C3D deve dare gli stessi cicli a
    # chiunque lo apra, oggi e con una versione futura del codice.
    risultato["cycles"] = segmentation.detect(risultato) or {}
    return risultato


def process(sorgente, cartella_uscita, sport=None):
    """Converte una sorgente. Restituisce (percorso del C3D, riepilogo).

    `sport` dichiara il gesto quando la sorgente non lo sa: i C3D di laboratorio
    non lo scrivono, e senza di esso non si possono applicare ne' la strategia di
    segmentazione in cicli ne' i riferimenti di quel gesto. Se la sorgente lo
    dichiara gia', il valore passato lo sovrascrive: e' l'operatore a sapere cosa
    e' stato ripreso.

    L'identificativo del soggetto NON entra qui: resta un'associazione del
    database. Il C3D canonico e' un file che puo' essere copiato o condiviso, e
    non deve portarsi dietro nemmeno lo pseudonimo.
    """
    # Il gesto passato a mano si controlla PRIMA di leggere: e' un errore di
    # comando, e non ha senso caricare qualche megabyte di coordinate per poi
    # rifiutare un argomento. Quello che arriva dalla sorgente si controlla
    # dopo, perche' prima non lo si conosce.
    _gesto_canonico(sport)
    letto = readers.read(sorgente)
    dichiarato = sport or letto["meta"].get("sport")
    if dichiarato:
        # Nel file finisce il nome canonico: e' l'unico che segmentazione e
        # riferimenti riconoscono entrambi.
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
