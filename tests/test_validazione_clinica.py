"""Il calcolo degli angoli, misurato contro un modello clinico.

I C3D con marker set Plug-in Gait portano punti e uscite di modello insieme:
sono l'unico posto dove il calcolo del progetto e' verificabile, perche' sui
dati MediaPipe non c'e' nessuna verita' con cui confrontarlo.

C'e' un'insidia che questi test evitano apposta. Quando la sorgente porta gia'
gli angoli, il convertitore li **ricopia** invece di ricalcolarli (provenienza
`measured`): confrontare quei valori col modello darebbe scarto zero e
sembrerebbe una validazione, mentre sarebbe un numero confrontato con se
stesso. Qui gli angoli della sorgente vengono svuotati prima del calcolo.

    python -m pytest tests/test_validazione_clinica.py -q
"""
import os

import pytest

from tools.validazione_clinica import riepilogo

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = [os.path.join(RADICE, "DatiGrezzi", n) for n in ("Ref_1", "Ref_2")]
pytestmark = pytest.mark.skipif(
    not any(os.path.isdir(d) for d in DATASET),
    reason="servono i C3D di laboratorio con uscite di modello (DatiGrezzi/Ref_*)")


@pytest.fixture(scope="module")
def misure():
    # Due file per condizione bastano: le mediane sono stabili e la suite resta
    # veloce. `python -m tools.validazione_clinica` gira su tutti e 118.
    return riepilogo([d for d in DATASET if os.path.isdir(d)], per_condizione=2)


# Giunti la cui forma d'onda coincide con quella del modello clinico: restano
# differenze di convenzione (segno, guadagno, offset), non di gesto.
CONVENZIONE = (("shoulder_l", "frontal"), ("shoulder_r", "frontal"),
               ("elbow_l", "sagittal"), ("elbow_r", "sagittal"),
               ("shoulder_l", "sagittal"), ("shoulder_r", "sagittal"))


def test_l_arto_superiore_ha_la_forma_giusta(misure):
    """Correlazione quasi perfetta e residuo minimo dopo il fit lineare.

    Un residuo piccolo dice che la curva e' la stessa: cio' che resta e' come
    la si nomina, non cosa descrive.
    """
    for chiave in CONVENZIONE:
        assert chiave in misure, f"manca il confronto per {chiave}"
        v = misure[chiave]
        assert v["corr"] > 0.95, f"{chiave}: correlazione {v['corr']:.3f}"
        assert v["residuo_rms"] < 3.0, f"{chiave}: residuo {v['residuo_rms']:.1f} gradi"


def test_la_spalla_frontale_e_invertita_rispetto_al_modello(misure):
    """Differenza di convenzione dichiarata, non ancora riconciliata.

    La scala e' vicina a -1 ma non vale -1: oltre al segno c'e' un guadagno del
    10-20%, quindi non esiste una costante che riconcili le due definizioni e
    invertire il segno non basterebbe. Il test fissa cio' che e' stato misurato,
    cosi' il giorno in cui la convenzione cambia il cambiamento e' visibile.
    """
    for lato in ("shoulder_l", "shoulder_r"):
        scala = misure[(lato, "frontal")]["scala"]
        assert scala < -1.0, f"{lato}: scala {scala:+.2f}, atteso negativo"
        assert abs(scala) < 1.5, f"{lato}: scala {scala:+.2f} fuori dal misurato"


def test_anca_e_ginocchio_coincidono_col_modello_clinico(misure):
    """Anca e ginocchio devono seguire il modello clinico, non un altro giunto.

    Questi valori erano a 13-17 gradi di residuo finche' il lettore ha preso
    l'origine delle terne per il centro del giunto prossimale: in Plug-in Gait
    e' quello distale, quindi tutto l'arto entrava sfalsato di un giunto e il
    progetto calcolava il ginocchio chiamandolo anca. La soglia bassa e' cio'
    che impedisce a quell'errore di tornare senza farsi notare.
    """
    for chiave in (("knee_l", "sagittal"), ("knee_r", "sagittal")):
        v = misure[chiave]
        assert v["corr"] > 0.95, f"{chiave}: correlazione {v['corr']:.3f}"
        assert v["residuo_rms"] < 6.0, f"{chiave}: residuo {v['residuo_rms']:.1f} gradi"
        # scala vicino a 1: stessa ampiezza, non solo stessa forma
        assert 0.8 < v["scala"] < 1.2, f"{chiave}: scala {v['scala']:+.2f}"

    for chiave in (("hip_l", "sagittal"), ("hip_r", "sagittal"),
                   ("hip_l", "frontal"), ("hip_r", "frontal")):
        v = misure[chiave]
        assert v["corr"] > 0.95, f"{chiave}: correlazione {v['corr']:.3f}"
        assert v["residuo_rms"] < 3.0, f"{chiave}: residuo {v['residuo_rms']:.1f} gradi"


def test_i_punti_canonici_stanno_nell_ordine_anatomico():
    """Spalla sopra anca sopra ginocchio sopra caviglia, in una posa eretta.

    E' il controllo che avrebbe intercettato subito lo sfasamento di un giunto:
    con la mappatura sbagliata l'anca finiva a 0.55 m, cioe' all'altezza del
    ginocchio. Non serve un modello per accorgersene, basta guardare le quote.
    """
    import numpy as np
    from src.conversion import readers

    prova = os.path.join(RADICE, "DatiGrezzi", "Ref_1", "level_up",
                         "Ref01_sss_le_shoes04.c3d")
    if not os.path.isfile(prova):
        pytest.skip("serve una prova di laboratorio")
    punti = readers.read(prova)["points"]
    quote = {}
    for nome in ("shoulder_l", "hip_l", "knee_l", "ankle_l"):
        assert nome in punti, f"manca {nome}"
        quote[nome] = float(np.nanmedian(np.asarray(punti[nome], dtype=float)[:, 2]))
    assert quote["shoulder_l"] > quote["hip_l"] > quote["knee_l"] > quote["ankle_l"], quote


def test_la_caviglia_segue_il_modello_clinico(misure):
    """Il piede va orientato sull'asse tallone-punta e la gamba sul piano sagittale.

    Due difetti si sommavano qui. Il piede era definito caviglia-punta, un asse
    che in appoggio plantare non e' orizzontale: la posa neutra leggeva -12/-25
    gradi. E la gamba prendeva l'asse di flessione dal piano che contiene la
    bacchetta laterale, che e' quasi frontale, quindi la sua normale puntava in
    avanti invece che di lato e la dorsiflessione finiva nel piano frontale.
    Corretti entrambi, l'escursione calcolata coincide con quella clinica.
    """
    for lato in ("ankle_l", "ankle_r"):
        v = misure[(lato, "sagittal")]
        assert v["corr"] > 0.95, f"{lato}: correlazione {v['corr']:.3f}"
        assert v["residuo_rms"] < 3.0, f"{lato}: residuo {v['residuo_rms']:.1f} gradi"
        assert 0.8 < v["scala"] < 1.2, f"{lato}: scala {v['scala']:+.2f}"


def test_il_torace_segue_la_raccomandazione_isb(misure):
    """Con i marker toracici il tronco e' un corpo, non una retta.

    Finche' il tronco era la congiungente bacino-spalle dava l'inclinazione ma
    non la torsione, e la colonna ne ereditava l'imprecisione. Con i quattro
    marker (giugulare, C7, sterno, vertebra bassa) la terna e' quella della
    raccomandazione ISB e coincide col modello clinico.
    """
    for chiave in (("trunk", "sagittal"), ("trunk", "frontal"),
                   ("trunk", "transverse"), ("spine", "transverse")):
        v = misure[chiave]
        assert v["corr"] > 0.95, f"{chiave}: correlazione {v['corr']:.3f}"
        assert v["residuo_rms"] < 1.0, f"{chiave}: residuo {v['residuo_rms']:.2f} gradi"
        assert 0.9 < abs(v["scala"]) < 1.1, f"{chiave}: scala {v['scala']:+.2f}"


def test_gli_angoli_del_tronco_della_sorgente_vengono_letti(misure):
    """I modelli clinici li scrivono per lato (LSpineAngles): il prefisso conta.

    Senza ammetterlo l'etichetta non veniva riconosciuta e il valore del modello
    spariva in silenzio, senza che nulla lo segnalasse.
    """
    for giunto in ("trunk", "spine", "neck"):
        assert any(k[0] == giunto for k in misure), f"{giunto} non letto dalla sorgente"


def test_ogni_giunto_confrontabile_sta_entro_pochi_gradi(misure):
    """Nessun giunto deve tornare a divergere senza che qualcuno se ne accorga."""
    peggiore = max(misure.items(), key=lambda kv: kv[1]["residuo_rms"])
    chiave, v = peggiore
    assert v["residuo_rms"] < 4.0, (
        f"il caso peggiore e' {chiave} con {v['residuo_rms']:.1f} gradi")
