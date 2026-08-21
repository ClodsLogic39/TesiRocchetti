"""Test della processazione: vocabolario, angoli, robustezza, C3D canonico.

Le prove partono da una posa costruita a mano, con il soggetto in piedi e fermo,
e ne modificano un punto alla volta per verificare il valore che ne deve uscire.
Le prove sul C3D canonico scrivono e rileggono un file temporaneo.

    python -m pytest tests/test_conversione.py -q

Prerequisiti: pytest, numpy, scipy ed ezc3d. L'ultima prova richiede un C3D di
laboratorio sotto DatiGrezzi e viene saltata se il file non c'e'.
"""
import os

import numpy as np
import pytest

from src.conversion import anatomy, angles, canonical, process, readers

# --- Posa di riferimento e utilita' ---
# La posa in piedi da cui partono le prove, la funzione che ne modifica un punto
# alla volta e le scorciatoie per calcolare e leggere un angolo.
_IN_PIEDI = {
    "head": (0, 1.70, 0), "shoulder_c": (0, 1.45, 0),
    "shoulder_l": (-.20, 1.45, 0), "shoulder_r": (.20, 1.45, 0),
    "elbow_l": (-.20, 1.15, 0), "elbow_r": (.20, 1.15, 0),
    "wrist_l": (-.20, .90, 0), "wrist_r": (.20, .90, 0),
    "hip_l": (-.10, .90, 0), "hip_r": (.10, .90, 0),
    "knee_l": (-.10, .50, 0), "knee_r": (.10, .50, 0),
    "ankle_l": (-.10, .08, 0), "ankle_r": (.10, .08, 0),
    "heel_l": (-.10, .03, .06), "heel_r": (.10, .03, .06),
    "toe_l": (-.10, .03, -.15), "toe_r": (.10, .03, -.15),
}


def posa(**mod):
    p = dict(_IN_PIEDI)
    p.update(mod)
    p["hip_c"] = tuple((np.array(p["hip_l"]) + np.array(p["hip_r"])) / 2)
    p["knee_c"] = tuple((np.array(p["knee_l"]) + np.array(p["knee_r"])) / 2)
    p["shoulder_c"] = tuple((np.array(p["shoulder_l"]) + np.array(p["shoulder_r"])) / 2) \
        if "shoulder_l" in mod or "shoulder_r" in mod else p["shoulder_c"]
    return {k: np.tile(np.asarray(v, dtype=float), (8, 1)) for k, v in p.items()}


def calcola(punti):
    R, _ = angles.world_frame(punti)
    terne, _, _, _ = angles.segment_frames(punti, R)
    return angles.joint_angles(terne, fps=30)[0], terne


def val(a, giunto, piano):
    return float(a[(giunto, piano)][3])


# --- Vocabolario ---
# I punti canonici e i segmenti che se ne ricavano, la provenienza dei piani di
# ogni giunto e la registrazione dei piani non osservabili come assenza.
def test_i_punti_canonici_e_i_segmenti_che_ne_discendono():
    assert len(anatomy.POINTS) == 18
    assert {"heel_l", "heel_r"} <= set(anatomy.POINTS)
    punti = posa()
    _, terne = calcola(punti)
    costruiti = {s for s in terne if s != anatomy.WORLD}
    assert costruiti == set(anatomy.SEGMENTS), costruiti ^ set(anatomy.SEGMENTS)
    ammessi = set(anatomy.POINTS) | set(anatomy.DERIVED_POINTS) | set(anatomy.AUX_POINTS)
    for segmento in anatomy.SEGMENTS:
        richiesti = set(anatomy.segment_inputs(segmento))
        for ricetta in anatomy.roll_recipes(segmento):
            if ricetta[0] != "borrow":
                richiesti |= set(anatomy.roll_inputs(ricetta))
        assert richiesti <= ammessi, f"{segmento}: {richiesti - ammessi}"


def test_provenienza_dei_piani():
    D, M = anatomy.DERIVED, anatomy.MODELLED
    assert anatomy.plane_provenance("hip_r", "sagittal") == D
    assert anatomy.plane_provenance("hip_r", "frontal") == D
    assert anatomy.plane_provenance("hip_r", "transverse") == M
    assert anatomy.plane_provenance("knee_r", "sagittal") == D
    assert anatomy.plane_provenance("knee_r", "frontal") == M
    assert all(anatomy.plane_provenance("trunk", p) == D for p in anatomy.PLANES)
    assert anatomy.plane_provenance("spine", "sagittal") == M
    assert anatomy.plane_provenance("spine", "frontal") == D
    assert anatomy.plane_provenance("spine", "transverse") == D


def test_piani_non_osservabili_diventano_assenza():
    punti = posa()
    R, _ = angles.world_frame(punti)
    terne, _, _, _ = angles.segment_frames(punti, R)
    valori, prov, assenze = angles.joint_angles(terne, fps=30)

    assert all(v == anatomy.DERIVED for v in prov.values())
    for giunto in anatomy.JOINTS:
        for piano in anatomy.PLANES:
            osservabile = piano in anatomy.observable_planes(giunto)
            assert ((giunto, piano) in valori) == osservabile, (giunto, piano)
            if not osservabile:
                motivo, _ = assenze[(giunto, piano)]
                assert motivo == anatomy.NOT_OBSERVABLE

    assert ("knee_r", "sagittal") in valori
    assert ("knee_r", "transverse") not in valori
    assert ("hip_r", "frontal") in valori
    assert ("trunk", "transverse") in valori
    assert ("spine", "transverse") in valori


# --- Convenzione angolare ---
# Che la posa neutra dia zero, che i segni delle flessioni siano quelli clinici e
# simmetrici fra i due lati, e che tronco e colonna misurino cio' che dichiarano.
def test_posa_neutra_a_zero():
    a, _ = calcola(posa())
    for (giunto, piano), serie in a.items():
        if giunto.startswith("ankle"):
            continue
        assert abs(float(serie[3])) < 1.0, f"{giunto} {piano} = {serie[3]:.2f}"


def test_la_caviglia_legge_zero_in_appoggio_plantare():
    a, _ = calcola(posa())
    sx, dx = val(a, "ankle_l", "sagittal"), val(a, "ankle_r", "sagittal")
    assert abs(sx - dx) < 1.0, f"scostamento asimmetrico: {sx:.1f} vs {dx:.1f}"
    assert abs(sx) < 2.0, f"la posa neutra non legge zero: {sx:.1f}"


def test_flessioni_positive_e_simmetriche():
    prove = [
        ("knee", dict(ankle_l=(-.10, .20, .35), toe_l=(-.10, .14, .20),
                      ankle_r=(.10, .20, .35), toe_r=(.10, .14, .20))),
        ("hip", dict(knee_l=(-.10, .62, -.28), ankle_l=(-.10, .20, -.28),
                     toe_l=(-.10, .14, -.43), knee_r=(.10, .62, -.28),
                     ankle_r=(.10, .20, -.28), toe_r=(.10, .14, -.43))),
        ("elbow", dict(wrist_l=(-.20, 1.15, -.25), wrist_r=(.20, 1.15, -.25))),
        ("shoulder", dict(elbow_l=(-.20, 1.45, -.30), wrist_l=(-.20, 1.45, -.55),
                          elbow_r=(.20, 1.45, -.30), wrist_r=(.20, 1.45, -.55))),
        ("ankle", dict(toe_l=(-.10, .16, -.13), toe_r=(.10, .16, -.13))),
    ]
    riposo, _ = calcola(posa())
    for giunto, mod in prove:
        a, _ = calcola(posa(**mod))
        sx = val(a, f"{giunto}_l", "sagittal") - val(riposo, f"{giunto}_l", "sagittal")
        dx = val(a, f"{giunto}_r", "sagittal") - val(riposo, f"{giunto}_r", "sagittal")
        assert sx > 10 and dx > 10, f"{giunto}: {sx:.1f} / {dx:.1f}"
        assert abs(sx - dx) < 2.0, f"{giunto} asimmetrico: {sx:.1f} vs {dx:.1f}"


def test_tronco_riferito_alla_verticale():
    avanti, _ = calcola(posa(shoulder_c=(0, 1.42, -.30), head=(0, 1.62, -.40),
                             shoulder_l=(-.20, 1.42, -.30), shoulder_r=(.20, 1.42, -.30)))
    assert 25 < val(avanti, "trunk", "sagittal") < 35


def test_spine_isola_la_rotazione_del_bacino_dal_tronco():
    n = 8
    punti = {k: np.tile(np.asarray(v, dtype=float), (n, 1))
             for k, v in _IN_PIEDI.items()}
    theta = np.radians(15.0)
    ruota = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
    ultimo = n - 1
    punti["hip_l"][ultimo] = ruota @ np.asarray(_IN_PIEDI["hip_l"])
    punti["hip_r"][ultimo] = ruota @ np.asarray(_IN_PIEDI["hip_r"])
    punti["hip_c"] = (punti["hip_l"] + punti["hip_r"]) / 2.0
    punti["knee_c"] = (punti["knee_l"] + punti["knee_r"]) / 2.0

    a, _ = calcola(punti)
    assert abs(a[("trunk", "transverse")][ultimo]) < 3.0, "le spalle non si sono mosse"
    assert abs(a[("spine", "transverse")][ultimo]) > 8.0, \
        "il bacino e' ruotato sotto un tronco fermo"


# --- Robustezza dei dati ---
# Tratti troppo brevi scartati, buchi lunghi lasciati aperti e punti visti troppo
# poco registrati come assenza invece che usati.
def test_tratti_troppo_corti_scartati():
    serie = np.full((100, 3), np.nan)
    serie[10:14] = [1.0, 2.0, 3.0]
    serie[40:80] = [1.0, 2.0, 3.0]
    out = angles.drop_short_runs(serie, angles.MIN_RUN)
    assert not np.isfinite(out[10:14]).any()
    assert np.isfinite(out[40:80]).all()


def test_buchi_lunghi_restano_buchi():
    serie = np.zeros((50, 3))
    serie[20:35] = np.nan
    out = angles.interpolate_gaps(serie, max_gap=5)
    assert not np.isfinite(out[20:35]).any()
    corto = np.zeros((50, 3))
    corto[20:23] = np.nan
    assert np.isfinite(angles.interpolate_gaps(corto, max_gap=5)).all()


def test_punto_poco_coperto_diventa_assenza():
    n = 200
    punti = {k: np.tile(np.asarray(v, dtype=float), (n, 1))
             for k, v in _IN_PIEDI.items()}
    punti["knee_l"][:] = np.nan
    punti["knee_l"][:4] = _IN_PIEDI["knee_l"]
    letto = readers._finalise(punti, {"fps": 30, "n_frames": n, "source": "prova"})
    assert "knee_l" not in letto["points"]
    assert letto["absent"]["knee_l"][0] == anatomy.LOW_COVERAGE


# --- C3D canonico ---
# Scrittura e rilettura del file, cicli letti dal file e non ricalcolati, scelta
# del metodo di segmentazione, e rifiuto dei file che non rispettano il formato.
def test_c3d_canonico_andata_e_ritorno(tmp_path):
    n = 60
    punti = {k: np.tile(np.asarray(v, dtype=float), (n, 1))
             for k, v in _IN_PIEDI.items()}
    punti["wrist_l"][10:20] = np.nan
    letto = readers._finalise(punti, {"fps": 60.0, "n_frames": n, "source": "prova",
                                      "source_format": "sintetico",
                                      "capture_system": "prova", "sport": "test"})
    originale = process.analizza(letto)
    percorso = canonical.write(originale, str(tmp_path / "prova.c3d"))
    riletto = canonical.read(percorso)

    assert set(riletto["points"]) == set(originale["points"])
    assert set(riletto["angles"]) == set(originale["angles"])
    assert set(riletto["segments"]) == set(originale["segments"])
    assert riletto["absences"] == originale["absences"]
    assert riletto["cycles"] == originale["cycles"]
    assert riletto["angle_provenance"] == originale["angle_provenance"]
    assert riletto["meta"]["sport"] == "test"
    assert riletto["meta"]["fps"] == 60.0

    for nome, v in originale["points"].items():
        d = np.abs(np.asarray(v) - riletto["points"][nome])
        assert np.nanmax(d) < 1e-4, nome
        assert (np.isfinite(riletto["points"][nome]).all(axis=1)
                == np.isfinite(v).all(axis=1)).all(), nome
    for chiave, v in originale["angles"].items():
        d = np.abs(np.asarray(v) - riletto["angles"][chiave])
        assert np.nanmax(d) < 1e-3, chiave


def test_i_cicli_sono_scritti_nel_file_non_ricavati_al_caricamento(tmp_path):
    n = 120
    t = np.linspace(0, 4 * np.pi, n)
    punti = {k: np.tile(np.asarray(v, dtype=float), (n, 1))
             for k, v in _IN_PIEDI.items()}
    punti["toe_l"][:, 2] -= 0.30 * np.sin(t)
    punti["toe_r"][:, 2] -= 0.30 * np.sin(t + np.pi)
    letto = readers._finalise(punti, {"fps": 30.0, "n_frames": n, "source": "prova",
                                      "source_format": "sintetico",
                                      "capture_system": "prova", "sport": "walking"})
    cicli = canonical.read(canonical.write(
        process.analizza(letto), str(tmp_path / "prova.c3d")))["cycles"]

    assert cicli["method"] in ("toe_coordinate", "toe_coordinate_step"), (
        f"il contatto doveva essere rilevato, invece: {cicli['method']!r}")
    assert cicli["frames"], "nessun ciclo scritto nel file"
    for inizio, fine in cicli["frames"]:
        assert 0 <= inizio < fine <= n, (inizio, fine)


def test_una_ripresa_della_sola_parte_alta_non_perde_i_cicli(tmp_path):
    n, fps = 300, 30.0
    t = np.linspace(0, 10 * 2 * np.pi, n)
    punti = {k: np.tile(np.asarray(v, dtype=float), (n, 1))
             for k, v in _IN_PIEDI.items()
             if not any(k.startswith(p) for p in
                        ("knee", "ankle", "heel", "toe"))}
    for lato, fase in (("l", 0.0), ("r", np.pi)):
        for punto, ampiezza in (("elbow", 0.18), ("wrist", 0.30)):
            punti[f"{punto}_{lato}"][:, 2] -= ampiezza * np.sin(t + fase)
    letto = readers._finalise(punti, {"fps": fps, "n_frames": n, "source": "prova",
                                      "source_format": "sintetico",
                                      "capture_system": "prova", "sport": "running"})
    cicli = process.analizza(letto)["cycles"]

    assert cicli["method"] == "shoulder_flexion", (
        f"atteso il ripiego sulla spalla, ottenuto {cicli['method']!r}")
    assert len(cicli["frames"]) >= 5, f"solo {len(cicli['frames'])} cicli"
    assert "cycles_from_surrogate" in cicli["warnings"]


def test_un_segnale_quasi_assente_non_batte_uno_completo():
    n, fps = 300, 30.0
    t = np.linspace(0, 10 * 2 * np.pi, n)
    punti = {k: np.tile(np.asarray(v, dtype=float), (n, 1))
             for k, v in _IN_PIEDI.items()}
    for lato, fase in (("l", 0.0), ("r", np.pi)):
        punti[f"toe_{lato}"][:, 2] -= 0.30 * np.sin(t + fase)
        for punto, ampiezza in (("elbow", 0.18), ("wrist", 0.30)):
            punti[f"{punto}_{lato}"][:, 2] -= ampiezza * np.sin(t + fase)
    meta = {"fps": fps, "n_frames": n, "source": "prova", "source_format": "sintetico",
            "capture_system": "prova", "sport": "running"}

    intero = process.analizza(readers._finalise(
        {k: v.copy() for k, v in punti.items()}, dict(meta)))["cycles"]
    assert intero["method"] in ("toe_coordinate", "toe_coordinate_step"), intero["method"]

    bucato = {k: v.copy() for k, v in punti.items()}
    for lato in ("l", "r"):
        bucato[f"toe_{lato}"][:int(n * 0.9)] = np.nan
    cicli = process.analizza(readers._finalise(bucato, dict(meta)))["cycles"]
    assert cicli["method"] == "shoulder_flexion", (
        f"il contatto andava scartato per copertura, invece: {cicli['method']!r}")
    assert "primary_signal_low_coverage" in cicli["warnings"], (
        "il ripiego per scarsa copertura va dichiarato")


def test_scrivere_un_c3d_senza_cicli_e_un_errore(tmp_path):
    n = 20
    punti = {k: np.tile(np.asarray(v, dtype=float), (n, 1))
             for k, v in _IN_PIEDI.items()}
    letto = readers._finalise(punti, {"fps": 30.0, "n_frames": n, "source": "prova",
                                      "source_format": "sintetico",
                                      "capture_system": "prova", "sport": "walking"})
    with pytest.raises(ValueError, match="cicli"):
        canonical.write(angles.compute(letto), str(tmp_path / "monco.c3d"))


def test_un_canonico_di_prima_dei_cicli_viene_rifiutato(tmp_path):
    import ezc3d
    c = ezc3d.c3d()
    c["parameters"]["POINT"]["RATE"]["value"] = [30.0]
    c["parameters"]["POINT"]["LABELS"]["value"] = ["A"]
    c["data"]["points"] = np.zeros((4, 1, 5))
    c.add_parameter("BIOMECH", "VERSION", [canonical.VERSION], "")
    percorso = str(tmp_path / "vecchio.c3d")
    c.write(percorso)
    with pytest.raises(ValueError, match="riconvertito"):
        canonical.read(percorso)


def test_c3d_non_canonico_viene_rifiutato(tmp_path):
    import ezc3d
    c = ezc3d.c3d()
    c["parameters"]["POINT"]["RATE"]["value"] = [30.0]
    c["parameters"]["POINT"]["LABELS"]["value"] = ["A"]
    c["data"]["points"] = np.zeros((4, 1, 5))
    percorso = str(tmp_path / "estraneo.c3d")
    c.write(percorso)
    with pytest.raises(ValueError, match="BIOMECH"):
        canonical.read(percorso)


# --- Angoli gia' presenti nella sorgente ---
# Su un C3D di laboratorio, che gli angoli portati dalla sorgente arrivino
# invariati fino al file convertito.
_LABORATORIO = "DatiGrezzi/Ref_1/level_up/Ref01_sss_le_shoes04.c3d"


@pytest.mark.skipif(not os.path.isfile(_LABORATORIO),
                    reason="serve un C3D di laboratorio con uscite di modello")
def test_gli_angoli_della_sorgente_non_vengono_mai_ricalcolati(tmp_path):
    letto = readers.read(_LABORATORIO)
    risultato = process.analizza(letto)

    misurati = [k for k, v in risultato["angle_provenance"].items()
                if v == anatomy.MEASURED]
    assert misurati, "il file ha uscite di modello e vanno conservate"
    assert any(p == "transverse" for _, p in misurati)
    for chiave in misurati:
        assert np.allclose(risultato["angles"][chiave], letto["angles"][chiave],
                           equal_nan=True), chiave

    percorso, riepilogo = process.process(_LABORATORIO, str(tmp_path),
                                          sport="walking")
    assert os.path.isfile(percorso)
    assert riepilogo["angoli_misurati"] == len(misurati)
    riletto = canonical.read(percorso)
    assert riletto["meta"]["biomech_version"] == canonical.VERSION
    for chiave in misurati:
        assert np.allclose(riletto["angles"][chiave], letto["angles"][chiave],
                           equal_nan=True, atol=1e-3), chiave
