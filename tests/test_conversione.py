"""Test della processazione: vocabolario, convenzione angolare, C3D canonico.

    python -m pytest tests/ -q
"""
import os

import numpy as np
import pytest

from src.conversion import anatomy, angles, canonical, process, readers

# Posa di riferimento: in piedi, fermo, rivolto in avanti.
# Convenzione ISB: con +x a destra e +y in alto, l'anteriore e' -z.
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
def test_i_punti_canonici_e_i_segmenti_che_ne_discendono():
    # Diciotto: ai sedici centri articolari si e' aggiunto il tallone, che
    # serve a orientare il piede come fa il modello clinico (asse tallone->
    # punta) e che entrambe le sorgenti forniscono.
    assert len(anatomy.POINTS) == 18
    assert {"heel_l", "heel_r"} <= set(anatomy.POINTS)
    punti = posa()
    _, terne = calcola(punti)
    costruiti = {s for s in terne if s != anatomy.WORLD}
    assert costruiti == set(anatomy.SEGMENTS), costruiti ^ set(anatomy.SEGMENTS)
    # ogni segmento nasce solo da punti del vocabolario o da centri costruiti
    # I marker ausiliari non sono centri articolari, ma sono un vocabolario
    # chiuso come gli altri: un segmento non puo' riferirsi a un punto che
    # nessuno ha dichiarato.
    ammessi = set(anatomy.POINTS) | set(anatomy.DERIVED_POINTS) | set(anatomy.AUX_POINTS)
    for segmento in anatomy.SEGMENTS:
        richiesti = set(anatomy.segment_inputs(segmento))
        for ricetta in anatomy.roll_recipes(segmento):
            if ricetta[0] != "borrow":
                richiesti |= set(anatomy.roll_inputs(ricetta))
        assert richiesti <= ammessi, f"{segmento}: {richiesti - ammessi}"


def test_provenienza_dei_piani():
    """Cosa nasce da geometria osservata e cosa da un'assunzione di modello."""
    D, M = anatomy.DERIVED, anatomy.MODELLED
    # anca: bacino completo contro femore a piano -> flessione e abduzione si',
    # rotazione assiale no
    assert anatomy.plane_provenance("hip_r", "sagittal") == D
    assert anatomy.plane_provenance("hip_r", "frontal") == D
    assert anatomy.plane_provenance("hip_r", "transverse") == M
    # ginocchio: cerniera fra due segmenti a piano -> solo la flessione
    assert anatomy.plane_provenance("knee_r", "sagittal") == D
    assert anatomy.plane_provenance("knee_r", "frontal") == M
    # tronco vs mondo: la terna e' piena (il rollio viene dalla linea delle
    # spalle o del bacino, non da un piano assunto) -> tutto osservato,
    # rotazione assiale compresa
    assert all(anatomy.plane_provenance("trunk", p) == D for p in anatomy.PLANES)
    # colonna (tronco vs bacino): frontale e trasverso sono osservati come per
    # il tronco. Il sagittale no: il bacino prende il suo rollio dalla stessa
    # direzione (hip_c -> shoulder_c) che definisce l'asse lungo del tronco,
    # quindi la flessione relativa risulta zero per costruzione, non per
    # assenza di movimento (vedi il commento in plane_provenance).
    assert anatomy.plane_provenance("spine", "sagittal") == M
    assert anatomy.plane_provenance("spine", "frontal") == D
    assert anatomy.plane_provenance("spine", "transverse") == D


def test_piani_non_osservabili_diventano_assenza():
    """Un piano che il modello non vede non e' un valore impreciso: non c'e'.

    Fra due segmenti il cui asse di flessione nasce da un'assunzione di piano, la
    rotazione assiale e' rumore della normale al piano: scriverla darebbe
    escursioni di decine di gradi credibili e prive di significato.
    """
    punti = posa()
    R, _ = angles.world_frame(punti)
    terne, _, _, _ = angles.segment_frames(punti, R)
    valori, prov, assenze = angles.joint_angles(terne, fps=30)

    # nessun angolo scritto puo' essere modellato
    assert all(v == anatomy.DERIVED for v in prov.values())
    # e ogni piano non osservabile deve comparire fra le assenze, col motivo
    for giunto in anatomy.JOINTS:
        for piano in anatomy.PLANES:
            osservabile = piano in anatomy.observable_planes(giunto)
            assert ((giunto, piano) in valori) == osservabile, (giunto, piano)
            if not osservabile:
                motivo, _ = assenze[(giunto, piano)]
                assert motivo == anatomy.NOT_OBSERVABLE

    assert ("knee_r", "sagittal") in valori          # flessione: si'
    assert ("knee_r", "transverse") not in valori    # rotazione assiale: no
    assert ("hip_r", "frontal") in valori            # bacino pieno: ab/adduzione si'
    assert ("trunk", "transverse") in valori         # due terne piene: tutto si'
    assert ("spine", "transverse") in valori         # idem, tronco vs bacino


# --- Convenzione angolare ---
def test_posa_neutra_a_zero():
    """In piedi e fermo ogni giunto legge zero, tranne la caviglia."""
    a, _ = calcola(posa())
    for (giunto, piano), serie in a.items():
        if giunto.startswith("ankle"):
            continue
        assert abs(float(serie[3])) < 1.0, f"{giunto} {piano} = {serie[3]:.2f}"


def test_la_caviglia_legge_zero_in_appoggio_plantare():
    """Con l'asse tallone-punta lo zero geometrico e' quello clinico.

    Finche' il piede era definito come caviglia-punta, quell'asse non era
    orizzontale in appoggio plantare e la posa neutra leggeva -12/-25 gradi: uno
    scostamento che il test dichiarava non eliminabile senza una prova statica.
    Con l'asse tallone-punta del modello clinico sparisce da solo.
    """
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
    """Il bacino ruota, le spalle no: la "spine" lo vede, il "trunk" no.

    E' la proprieta' per cui esiste il giunto: isolare il movimento della
    colonna da quello del bacino sotto di essa, non solo separarli per nome.

    Solo un fotogramma su otto ha il bacino ruotato: la terna del laboratorio
    si calibra sulla mediana di tutta la sessione (world_frame), quindi resta
    ancorata alla maggioranza neutra e non insegue quel singolo fotogramma --
    se si ruotasse ogni fotogramma allo stesso modo la calibrazione lo
    seguirebbe, e "trunk" (misurato contro il laboratorio) risulterebbe
    anch'esso spostato, mascherando la differenza che il test vuole mostrare.
    Ruotare hip_l/hip_r attorno alla verticale non sposta il loro centro (le
    due x sono opposte, il centro resta a x=0), quindi l'asse lungo del
    tronco (hip_c -> shoulder_c) non cambia: solo il bacino si e' mosso.
    """
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
def test_tratti_troppo_corti_scartati():
    serie = np.full((100, 3), np.nan)
    serie[10:14] = [1.0, 2.0, 3.0]         # tratto corto: non e' una misura
    serie[40:80] = [1.0, 2.0, 3.0]         # tratto lungo: si tiene
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
    punti["knee_l"][:4] = _IN_PIEDI["knee_l"]      # 2% dei frame
    letto = readers._finalise(punti, {"fps": 30, "n_frames": n, "source": "prova"})
    assert "knee_l" not in letto["points"]
    assert letto["absent"]["knee_l"][0] == anatomy.LOW_COVERAGE


# --- C3D canonico ---
def test_c3d_canonico_andata_e_ritorno(tmp_path):
    n = 60
    punti = {k: np.tile(np.asarray(v, dtype=float), (n, 1))
             for k, v in _IN_PIEDI.items()}
    punti["wrist_l"][10:20] = np.nan               # un buco vero
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
        # il buco deve restare un buco
        assert (np.isfinite(riletto["points"][nome]).all(axis=1)
                == np.isfinite(v).all(axis=1)).all(), nome
    for chiave, v in originale["angles"].items():
        d = np.abs(np.asarray(v) - riletto["angles"][chiave])
        assert np.nanmax(d) < 1e-3, chiave


def test_i_cicli_sono_scritti_nel_file_non_ricavati_al_caricamento(tmp_path):
    """La segmentazione appartiene al file, non a chi lo legge.

    Finche' girava all'ingest, lo stesso C3D poteva dare cicli diversi a due
    caricamenti: bastava che una soglia della segmentazione cambiasse fra i due.
    Ora i confini sono decisi una volta e restano scritti.
    """
    n = 120
    t = np.linspace(0, 4 * np.pi, n)
    punti = {k: np.tile(np.asarray(v, dtype=float), (n, 1))
             for k, v in _IN_PIEDI.items()}
    # Un passo: le punte oscillano IN AVANTI in controfase. L'asse anteriore di
    # questa posa e' -z, non x: muoverle lungo x le farebbe oscillare di lato e
    # il rilevamento del contatto non troverebbe nulla.
    punti["toe_l"][:, 2] -= 0.30 * np.sin(t)
    punti["toe_r"][:, 2] -= 0.30 * np.sin(t + np.pi)
    letto = readers._finalise(punti, {"fps": 30.0, "n_frames": n, "source": "prova",
                                      "source_format": "sintetico",
                                      "capture_system": "prova", "sport": "walking"})
    cicli = canonical.read(canonical.write(
        process.analizza(letto), str(tmp_path / "prova.c3d")))["cycles"]

    # Non basta che un metodo ci sia: `whole_trial` e' il ripiego di quando non
    # si trova niente, e accettarlo renderebbe questo test sempre verde.
    assert cicli["method"] in ("toe_coordinate", "toe_coordinate_step"), (
        f"il contatto doveva essere rilevato, invece: {cicli['method']!r}")
    assert cicli["frames"], "nessun ciclo scritto nel file"
    for inizio, fine in cicli["frames"]:
        assert 0 <= inizio < fine <= n, (inizio, fine)


def test_una_ripresa_della_sola_parte_alta_non_perde_i_cicli(tmp_path):
    """Se le gambe sono fuori campo, il ritmo lo da' il braccio.

    E' il caso reale delle sessioni MediaPipe del progetto: le caviglie cadono
    sotto il bordo inferiore del fotogramma, MediaPipe le dichiara invisibili e
    i punti del piede spariscono. Prima, esaurita la lista, la sessione finiva
    in `whole_trial` -- sessanta secondi trattati come un gesto unico -- pur
    avendo l'oscillazione delle braccia perfettamente tracciata.
    """
    n, fps = 300, 30.0
    t = np.linspace(0, 10 * 2 * np.pi, n)          # dieci oscillazioni
    punti = {k: np.tile(np.asarray(v, dtype=float), (n, 1))
             for k, v in _IN_PIEDI.items()
             # tutto cio' che sta sotto l'anca resta fuori dal fotogramma
             if not any(k.startswith(p) for p in
                        ("knee", "ankle", "heel", "toe"))}
    # L'asse anteriore di questa posa e' -z (vedi `world_frame`): le braccia
    # devono oscillare li', o l'escursione finisce nel piano frontale e la
    # flessione di spalla resta a zero.
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
    # resta un surrogato, e va detto: il braccio non e' il contatto al suolo
    assert "cycles_from_surrogate" in cicli["warnings"]


def test_un_segnale_quasi_assente_non_batte_uno_completo():
    """Il metodo migliore vince solo se ha di che lavorare.

    La lista dei metodi e' a prima-che-riesce. Senza un limite sulla copertura,
    una punta visibile in pochi frame produceva cicli veri ma presi da una
    frazione della ripresa, e vinceva su un segnale disponibile per intero:
    tre cicli da due secondi al posto di cinquantaquattro da sessanta.
    """
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

    # con le punte intere vince il contatto, che e' il metodo giusto
    intero = process.analizza(readers._finalise(
        {k: v.copy() for k, v in punti.items()}, dict(meta)))["cycles"]
    assert intero["method"] in ("toe_coordinate", "toe_coordinate_step"), intero["method"]

    # con le punte visibili solo in fondo, il contatto va scartato
    bucato = {k: v.copy() for k, v in punti.items()}
    for lato in ("l", "r"):
        bucato[f"toe_{lato}"][:int(n * 0.9)] = np.nan
    cicli = process.analizza(readers._finalise(bucato, dict(meta)))["cycles"]
    assert cicli["method"] == "shoulder_flexion", (
        f"il contatto andava scartato per copertura, invece: {cicli['method']!r}")
    assert "primary_signal_low_coverage" in cicli["warnings"], (
        "il ripiego per scarsa copertura va dichiarato")


def test_scrivere_un_c3d_senza_cicli_e_un_errore(tmp_path):
    """Dimenticare la segmentazione non deve somigliare a non averne trovata."""
    n = 20
    punti = {k: np.tile(np.asarray(v, dtype=float), (n, 1))
             for k, v in _IN_PIEDI.items()}
    letto = readers._finalise(punti, {"fps": 30.0, "n_frames": n, "source": "prova",
                                      "source_format": "sintetico",
                                      "capture_system": "prova", "sport": "walking"})
    with pytest.raises(ValueError, match="cicli"):
        canonical.write(angles.compute(letto), str(tmp_path / "monco.c3d"))


def test_un_canonico_di_prima_dei_cicli_viene_rifiutato(tmp_path):
    """Un file vecchio va riconvertito, non caricato come se fosse muto."""
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


# --- Cio' che la sorgente porta gia' ---
_LABORATORIO = "DatiGrezzi/Ref_1/level_up/Ref01_sss_le_shoes04.c3d"


@pytest.mark.skipif(not os.path.isfile(_LABORATORIO),
                    reason="serve un C3D di laboratorio con uscite di modello")
def test_gli_angoli_della_sorgente_non_vengono_mai_ricalcolati(tmp_path):
    """La regola su cui poggia tutto il resto: chi ha gia' il dato lo conserva.

    Un modello clinico e' calibrato sul soggetto -- larghezze articolari, prova
    statica, regressioni del bacino -- e i suoi angoli valgono piu' di quelli che
    sappiamo ricavare dai soli centri articolari. Ricalcolarli sarebbe una
    perdita, e la provenienza `measured` e' la promessa che non succede.

    Il confronto fra i nostri angoli e quelli del modello sta invece in
    `test_validazione_clinica.py`, che per farlo deve prima svuotare quelli della
    sorgente: qui si verifica il caso opposto, cioe' che restino intatti.
    """
    letto = readers.read(_LABORATORIO)
    risultato = process.analizza(letto)

    misurati = [k for k, v in risultato["angle_provenance"].items()
                if v == anatomy.MEASURED]
    assert misurati, "il file ha uscite di modello e vanno conservate"
    # un modello a tre gradi di liberta' li fornisce tutti e tre: si conservano
    # anche sui piani che noi non sapremmo osservare
    assert any(p == "transverse" for _, p in misurati)
    for chiave in misurati:
        assert np.allclose(risultato["angles"][chiave], letto["angles"][chiave],
                           equal_nan=True), chiave

    # e sopravvivono al giro completo fino al file canonico
    percorso, riepilogo = process.process(_LABORATORIO, str(tmp_path),
                                          sport="walking")
    assert os.path.isfile(percorso)
    assert riepilogo["angoli_misurati"] == len(misurati)
    riletto = canonical.read(percorso)
    assert riletto["meta"]["biomech_version"] == canonical.VERSION
    for chiave in misurati:
        assert np.allclose(riletto["angles"][chiave], letto["angles"][chiave],
                           equal_nan=True, atol=1e-3), chiave
