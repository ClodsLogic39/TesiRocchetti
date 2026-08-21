"""Il calcolo degli angoli, misurato contro le uscite di un modello clinico.

I C3D con marker set Plug-in Gait contengono punti e uscite di modello insieme.
Le prove ricalcolano gli angoli dai soli punti, dopo aver svuotato quelli che la
sorgente porta gia' calcolati, e confrontano il risultato con le uscite del
modello presenti nello stesso file.

    python -m pytest tests/test_validazione_clinica.py -q

Prerequisiti: pytest, numpy ed ezc3d; i C3D di laboratorio sotto DatiGrezzi/Ref_1
e Ref_2. Senza quelle cartelle l'intero file viene saltato.
"""
import os

import pytest

from tools.validazione_clinica import riepilogo

# --- Dataset e misure ---
# Cartelle dei C3D di laboratorio, salto dell'intero file quando non ci sono, e
# la fixture che calcola una volta sola gli scarti rispetto al modello.
RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = [os.path.join(RADICE, "DatiGrezzi", n) for n in ("Ref_1", "Ref_2")]
pytestmark = pytest.mark.skipif(
    not any(os.path.isdir(d) for d in DATASET),
    reason="servono i C3D di laboratorio con uscite di modello (DatiGrezzi/Ref_*)")


@pytest.fixture(scope="module")
def misure():
    return riepilogo([d for d in DATASET if os.path.isdir(d)], per_condizione=2)


# --- Confronto con il modello clinico ---
# I giunti la cui forma d'onda coincide con quella del modello a meno di segno,
# guadagno e scostamento, e le prove su arto superiore, anca, ginocchio,
# caviglia, torace e ordine anatomico dei punti.
CONVENZIONE = (("shoulder_l", "frontal"), ("shoulder_r", "frontal"),
               ("elbow_l", "sagittal"), ("elbow_r", "sagittal"),
               ("shoulder_l", "sagittal"), ("shoulder_r", "sagittal"))


def test_l_arto_superiore_ha_la_forma_giusta(misure):
    for chiave in CONVENZIONE:
        assert chiave in misure, f"manca il confronto per {chiave}"
        v = misure[chiave]
        assert v["corr"] > 0.95, f"{chiave}: correlazione {v['corr']:.3f}"
        assert v["residuo_rms"] < 3.0, f"{chiave}: residuo {v['residuo_rms']:.1f} gradi"


def test_la_spalla_frontale_e_invertita_rispetto_al_modello(misure):
    for lato in ("shoulder_l", "shoulder_r"):
        scala = misure[(lato, "frontal")]["scala"]
        assert scala < -1.0, f"{lato}: scala {scala:+.2f}, atteso negativo"
        assert abs(scala) < 1.5, f"{lato}: scala {scala:+.2f} fuori dal misurato"


def test_anca_e_ginocchio_coincidono_col_modello_clinico(misure):
    for chiave in (("knee_l", "sagittal"), ("knee_r", "sagittal")):
        v = misure[chiave]
        assert v["corr"] > 0.95, f"{chiave}: correlazione {v['corr']:.3f}"
        assert v["residuo_rms"] < 6.0, f"{chiave}: residuo {v['residuo_rms']:.1f} gradi"
        assert 0.8 < v["scala"] < 1.2, f"{chiave}: scala {v['scala']:+.2f}"

    for chiave in (("hip_l", "sagittal"), ("hip_r", "sagittal"),
                   ("hip_l", "frontal"), ("hip_r", "frontal")):
        v = misure[chiave]
        assert v["corr"] > 0.95, f"{chiave}: correlazione {v['corr']:.3f}"
        assert v["residuo_rms"] < 3.0, f"{chiave}: residuo {v['residuo_rms']:.1f} gradi"


def test_i_punti_canonici_stanno_nell_ordine_anatomico():
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
    for lato in ("ankle_l", "ankle_r"):
        v = misure[(lato, "sagittal")]
        assert v["corr"] > 0.95, f"{lato}: correlazione {v['corr']:.3f}"
        assert v["residuo_rms"] < 3.0, f"{lato}: residuo {v['residuo_rms']:.1f} gradi"
        assert 0.8 < v["scala"] < 1.2, f"{lato}: scala {v['scala']:+.2f}"


def test_il_torace_segue_la_raccomandazione_isb(misure):
    for chiave in (("trunk", "sagittal"), ("trunk", "frontal"),
                   ("trunk", "transverse"), ("spine", "transverse")):
        v = misure[chiave]
        assert v["corr"] > 0.95, f"{chiave}: correlazione {v['corr']:.3f}"
        assert v["residuo_rms"] < 1.0, f"{chiave}: residuo {v['residuo_rms']:.2f} gradi"
        assert 0.9 < abs(v["scala"]) < 1.1, f"{chiave}: scala {v['scala']:+.2f}"


def test_gli_angoli_del_tronco_della_sorgente_vengono_letti(misure):
    for giunto in ("trunk", "spine", "neck"):
        assert any(k[0] == giunto for k in misure), f"{giunto} non letto dalla sorgente"


def test_ogni_giunto_confrontabile_sta_entro_pochi_gradi(misure):
    peggiore = max(misure.items(), key=lambda kv: kv[1]["residuo_rms"])
    chiave, v = peggiore
    assert v["residuo_rms"] < 4.0, (
        f"il caso peggiore e' {chiave} con {v['residuo_rms']:.1f} gradi")
