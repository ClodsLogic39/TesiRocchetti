"""Test del viewer: lettura del C3D canonico, alert e avvio dal server MCP.

La parte dati del viewer e' separata dal rendering, quindi queste prove non
richiedono Open3D: costruiscono un C3D canonico temporaneo e verificano che ossa,
busto, terne e angoli ne vengano ricostruiti. Le prove sugli alert chiamano
`_alert` direttamente su un `dati` costruito a mano, cosi' da non dipendere
nemmeno da ezc3d -- e verificano che il giunto giusto si accenda nel fotogramma
giusto, con la gravita' giusta. Le prove sull'avvio sostituiscono il processo
del viewer con uno finto.

    python -m pytest tests/test_viewer.py -q

Prerequisiti: pytest, numpy ed ezc3d. Open3D non serve.
"""
import numpy as np
import pytest

from src.conversion import anatomy, canonical, process, readers
from src.mcp import server as mcp_server
from src.visualization import viewer_open3d

# --- Posa di riferimento e C3D di prova ---
# La stessa posa in piedi dei test di conversione e la funzione che ne scrive un
# C3D canonico temporaneo, eventualmente privo di alcuni punti.
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


def _c3d(tmp_path, esclusi=(), n=40):
    punti = {k: np.tile(np.asarray(v, dtype=float), (n, 1))
             for k, v in _IN_PIEDI.items() if k not in esclusi}
    letto = readers._finalise(punti, {"fps": 30.0, "n_frames": n, "source": "prova",
                                      "source_format": "sintetico",
                                      "capture_system": "prova", "sport": "test"})
    return canonical.write(process.analizza(letto), str(tmp_path / "prova.c3d"))


# --- Lettura del C3D canonico ---
# Ricostruzione delle ossa dai punti, busto poggiato sui quattro punti reali e
# assente quando mancano, segmenti senza estremi saltati, terne come quaternioni
# per fotogramma, angoli della legenda e rifiuto di un file non canonico.
def test_carica_ricostruisce_le_ossa_dai_punti_canonici(tmp_path):
    dati = viewer_open3d.carica(_c3d(tmp_path))
    assert {o[0] for o in dati["ossa"]} == set(anatomy.SEGMENTS) - {"trunk"}
    assert dati["n_frame"] == 40
    assert dati["fps"] == 30.0
    for _, prossimale, distale, colore in dati["ossa"]:
        assert prossimale in dati["punti"] and distale in dati["punti"]
        assert len(colore) == 3 and all(0.0 <= c <= 1.0 for c in colore)


def test_il_busto_poggia_sui_quattro_punti_reali(tmp_path):
    dati = viewer_open3d.carica(_c3d(tmp_path))
    assert dati["ha_busto"]
    assert all(k in dati["punti"]
               for k in ("hip_l", "hip_r", "shoulder_l", "shoulder_r"))
    assert "trunk" not in {o[0] for o in dati["ossa"]}
    assert "trunk" in dati["angoli"] and "spine" in dati["angoli"]
    assert dati["angoli"]["trunk"][0] == "trunk"
    assert dati["angoli"]["spine"][0] == "spine"


def test_il_busto_manca_se_mancano_i_punti_delle_spalle(tmp_path):
    dati = viewer_open3d.carica(_c3d(tmp_path, esclusi=("shoulder_l", "shoulder_r")))
    assert not dati["ha_busto"]


def test_i_segmenti_senza_estremi_vengono_saltati_non_esplodono(tmp_path):
    dati = viewer_open3d.carica(_c3d(tmp_path, esclusi=("knee_l", "knee_r",
                                                        "ankle_l", "ankle_r",
                                                        "toe_l", "toe_r")))
    disegnabili = {o[0] for o in dati["ossa"]}
    assert "pelvis" in disegnabili and "humerus_l" in disegnabili
    for mancante in ("femur_l", "femur_r", "shank_l", "shank_r", "foot_l", "foot_r"):
        assert mancante not in disegnabili
    assert dati["ossa"], "il tronco deve restare disegnabile"


def test_le_terne_sono_quaternioni_per_frame(tmp_path):
    dati = viewer_open3d.carica(_c3d(tmp_path))
    assert dati["terne"], "il C3D canonico porta i quaternioni dei segmenti"
    for segmento, quat in dati["terne"].items():
        assert quat.shape == (dati["n_frame"], 4), segmento


def test_gli_angoli_della_legenda_sono_sagittali(tmp_path):
    dati = viewer_open3d.carica(_c3d(tmp_path))
    assert dati["angoli"], "la legenda mostra l'angolo sagittale del giunto"
    for chiave, (giunto, serie) in dati["angoli"].items():
        if chiave in ("trunk", "spine"):
            assert giunto == chiave
        else:
            assert anatomy.JOINTS[giunto]["distal"] == chiave
        assert len(serie) == dati["n_frame"]


def test_carica_porta_le_chiavi_dell_alert_anche_senza_alert(tmp_path):
    # La posa in piedi di _IN_PIEDI non sfonda nessun riferimento di
    # letteratura: carica() deve comunque restituire le chiavi nuove, vuote.
    dati = viewer_open3d.carica(_c3d(tmp_path))
    assert dati["alert"] == []
    assert dati["alert_per_punto"] == {}
    assert not dati["alert_busto"].any()
    assert dati["sport"] == "test"  # dichiarato in _c3d()


def test_carica_con_con_alert_falso_non_calcola_nulla(tmp_path):
    percorso = _c3d(tmp_path)
    con = viewer_open3d.carica(percorso, con_alert=True)
    senza = viewer_open3d.carica(percorso, con_alert=False)
    assert senza["alert"] == [] and senza["alert_per_punto"] == {}
    assert not senza["alert_busto"].any()
    assert con["alert"] == senza["alert"]  # qui coincidono perche' la posa e' pulita


def test_un_c3d_non_canonico_viene_rifiutato_con_chiarezza(tmp_path):
    import ezc3d
    c = ezc3d.c3d()
    c["parameters"]["POINT"]["RATE"]["value"] = [30.0]
    c["parameters"]["POINT"]["LABELS"]["value"] = ["pelvis_start"]
    c["data"]["points"] = np.zeros((4, 1, 5))
    percorso = str(tmp_path / "vecchio_formato.c3d")
    c.write(percorso)
    with pytest.raises(ValueError, match="BIOMECH"):
        viewer_open3d.carica(percorso)


# --- Alert per fotogramma ---
# `_alert` prende un `dati` con solo `meta` e `angles`, quindi queste prove non
# passano da un C3D vero: costruiscono la serie a mano, con un solo valore fuori
# dai riferimenti, e verificano che il giunto giusto si accenda nei fotogrammi
# giusti, con la gravita' giusta, e che trunk/spine finiscano sul busto invece
# che su un punto che non hanno.
def _dati_minimi(angoli, sport="walking", n_frame=None):
    n = n_frame or max(len(v) for v in angoli.values())
    return {"meta": {"sport": sport}, "angles": angoli}, n


def test_alert_localizza_il_giunto_nei_fotogrammi_giusti():
    # anca destra, sagittale: solo l'eccesso critico anatomico (aaos, peak_max
    # oltre 125) scatta -- nessun riferimento di sport perche' sport=None, e il
    # minimo resta dentro il limite. Un alert solo, sul fotogramma 2 soltanto.
    serie = [10.0, 20.0, 130.0, 40.0, 15.0]
    dati, n = _dati_minimi({("hip_r", "sagittal"): serie}, sport=None)
    voci, per_punto, busto = viewer_open3d._alert(dati, n)

    assert len(voci) == 1, voci
    a = voci[0]
    assert a["joint"] == "hip_r" and a["severity"] == "critical"
    assert a["localizzabile"]
    assert a["maschera"].tolist() == [False, False, True, False, False]

    assert per_punto["hip_r"].tolist() == [0, 0, 3, 0, 0]  # 3 = critical
    assert not busto.any()


def test_alert_di_sport_dipende_dal_gesto_dichiarato():
    # Lo stesso picco di flessione del ginocchio (80 gradi) e' dentro
    # l'intervallo atteso in corsa (70-120) ma fuori da quello in cammino
    # (50-75): stesso valore, verdetto diverso a seconda dello sport.
    serie = [10.0, 20.0, 80.0, 30.0, 15.0]

    dati_corsa, n = _dati_minimi({("knee_r", "sagittal"): serie}, sport="running")
    voci_corsa, per_punto_corsa, _ = viewer_open3d._alert(dati_corsa, n)
    assert voci_corsa == [], "80 gradi e' un picco normale in corsa"
    assert per_punto_corsa == {}

    dati_cammino, n = _dati_minimi({("knee_r", "sagittal"): serie}, sport="walking")
    voci_cammino, per_punto_cammino, _ = viewer_open3d._alert(dati_cammino, n)
    assert len(voci_cammino) == 1, voci_cammino
    a = voci_cammino[0]
    assert a["metric"] == "peak_max" and a["severity"] == "warning"
    assert a["maschera"].tolist() == [False, False, True, False, False]
    assert per_punto_cammino["knee_r"].tolist() == [0, 0, 2, 0, 0]  # 2 = warning


def test_alert_di_trunk_e_spine_tinge_il_busto_non_un_punto():
    # Inclinazione laterale del busto oltre il limite anatomico (spine, frontal,
    # critical oltre 35 gradi): spine non ha un punto proprio, quindi finisce
    # sul busto, non in alert_per_punto.
    serie = [0.0, 5.0, 45.0, 10.0, 0.0]
    dati, n = _dati_minimi({("spine", "frontal"): serie}, sport=None)
    voci, per_punto, busto = viewer_open3d._alert(dati, n)
    assert len(voci) == 1 and voci[0]["joint"] == "spine"
    assert voci[0]["punto"] is None
    assert per_punto == {}
    assert busto.tolist() == [0, 0, 3, 0, 0]  # 3 = critical


def test_alert_prende_la_gravita_peggiore_quando_due_alert_si_sovrappongono():
    # L'anca frontale ha un limite anatomico critico (aaos, peak_max oltre 50)
    # e, in cammino, un intervallo di ROM (walking, 5-20): questa serie sfonda
    # entrambi. Sul fotogramma del picco vince il critical; sul fotogramma
    # dell'altro estremo del ROM resta solo il suo avviso.
    serie = [0.0, 10.0, 140.0, 20.0, 0.0]
    dati, n = _dati_minimi({("hip_r", "frontal"): serie}, sport="walking")
    voci, per_punto, _ = viewer_open3d._alert(dati, n)
    assert {v["metric"] for v in voci} == {"peak_max", "rom"}
    assert per_punto["hip_r"].tolist() == [2, 0, 3, 0, 0]


def test_punto_del_giunto_mappa_gli_arti_e_isola_trunk_spine():
    assert viewer_open3d._punto_del_giunto("knee_l") == "knee_l"
    assert viewer_open3d._punto_del_giunto("ankle_r") == "ankle_r"
    assert viewer_open3d._punto_del_giunto("trunk") is None
    assert viewer_open3d._punto_del_giunto("spine") is None
    assert viewer_open3d._punto_del_giunto("neck") == "shoulder_c"


def test_adatta_allunga_o_accorcia_senza_inventare_valori():
    corta = np.array([1, 2], dtype=int)
    assert viewer_open3d._adatta(corta, 4).tolist() == [1, 2, 0, 0]
    lunga = np.array([1, 2, 3, 4], dtype=int)
    assert viewer_open3d._adatta(lunga, 2).tolist() == [1, 2]
    esatta = np.array([1, 2, 3], dtype=int)
    assert viewer_open3d._adatta(esatta, 3) is esatta


def test_rango_alert_legge_il_frame_giusto_e_zero_fuori_serie():
    dati = {"alert_per_punto": {"knee_l": np.array([0, 2, 3])}}
    assert viewer_open3d._rango_alert(dati, "knee_l", 1) == 2
    assert viewer_open3d._rango_alert(dati, "knee_l", 99) == 0
    assert viewer_open3d._rango_alert(dati, "hip_r", 0) == 0


def test_alert_attivi_solo_localizzabili_e_solo_nel_fotogramma():
    voci = [
        {"joint": "knee_r", "localizzabile": True,
         "maschera": np.array([False, True, False])},
        {"joint": "knee_l", "localizzabile": False,
         "maschera": np.array([False, False, False])},
    ]
    dati = {"alert": voci}
    assert [a["joint"] for a in viewer_open3d._alert_attivi(dati, 1)] == ["knee_r"]
    assert viewer_open3d._alert_attivi(dati, 0) == []


def test_riga_alert_mostra_intervallo_atteso_e_osservato():
    riga = viewer_open3d._riga_alert({
        "joint": "hip_r", "plane": "sagittal", "metric": "peak_max",
        "observed": 130.456, "expected_min": None, "expected_max": 125.0,
        "severity": "critical"})
    assert "hip_r" in riga and "130.5" in riga and "125" in riga and "critical" in riga


# --- Direzione della freccia ---
# La sola parte di `_freccia` che non tocca Open3D: la direzione radiale
# rispetto all'asse bacino-spalle, e il ripiego quando non e' calcolabile.
def test_direzione_radiale_e_perpendicolare_all_asse_del_busto():
    anche = np.array([0.0, 0.9, 0.0])
    spalle = np.array([0.0, 1.45, 0.0])  # asse verticale
    punto = np.array([0.15, 1.1, 0.0])   # spostato lateralmente
    d = viewer_open3d._direzione_radiale(punto, anche, spalle, np.array([1.0, 0.0, 0.0]))
    asse = (spalle - anche) / np.linalg.norm(spalle - anche)
    assert abs(float(d @ asse)) < 1e-6, "la direzione deve essere perpendicolare al busto"
    assert float(np.linalg.norm(d)) == pytest.approx(1.0)


def test_direzione_radiale_si_ripiega_quando_l_asse_manca():
    ripiego = np.array([0.0, 0.0, 1.0])
    assert viewer_open3d._direzione_radiale(np.zeros(3), None, np.zeros(3),
                                            ripiego) is ripiego
    coincidenti = np.array([0.0, 1.0, 0.0])
    assert viewer_open3d._direzione_radiale(coincidenti, coincidenti, coincidenti,
                                            ripiego) is ripiego


# --- Avvio dal server MCP ---
# Ricerca del file prodotto dalla conversione, segnalazione quando manca, rifiuto
# delle chiavi che contengono un percorso, e distinzione fra un viewer avviato e
# uno morto subito. Chiude l'interruttore che impedisce di aprire finestre.
def test_visualize_session_trova_il_c3d_del_convertitore(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "DATA_ROOT", tmp_path)
    (tmp_path / "sessione_x.c3d").write_bytes(b"")
    monkeypatch.setattr(mcp_server, "VIEWER_PY", "/bin/false")
    esito = mcp_server.visualize_session("sessione_x")
    assert "c3d_not_found" not in esito["warnings"]


def test_visualize_session_segnala_il_c3d_mancante(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "DATA_ROOT", tmp_path)
    esito = mcp_server.visualize_session("mai_convertita")
    assert esito["warnings"] == ["c3d_not_found"]


@pytest.mark.parametrize("chiave", ["../segreto", "a/b", "..", ""])
def test_visualize_session_rifiuta_le_chiavi_con_percorso(chiave, monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_server, "DATA_ROOT", tmp_path)
    esito = mcp_server.visualize_session(chiave)
    assert esito["warnings"] == ["invalid_session_key"]


def _finge_dipendenze_presenti(monkeypatch):
    class _Esito:
        returncode = 0
    monkeypatch.setattr(mcp_server.subprocess, "run", lambda *a, **k: _Esito())


class _ProcessoFinto:
    def __init__(self, uscita, errore=b"", log=None):
        self._uscita = uscita
        self.pid = 999
        if log is not None and errore:
            log.write(errore)
        self._log = log

    def wait(self, timeout=None):
        if self._uscita is None:
            raise mcp_server.subprocess.TimeoutExpired("viewer", timeout)
        return self._uscita


def _popen_finto(uscita, errore=b""):
    def _crea(*a, **k):
        return _ProcessoFinto(uscita, errore, log=k.get("stderr"))
    return _crea


def test_un_viewer_che_muore_subito_non_viene_dato_per_avviato(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "DATA_ROOT", tmp_path)
    (tmp_path / "sessione_x.c3d").write_bytes(b"")
    _finge_dipendenze_presenti(monkeypatch)
    monkeypatch.setattr(mcp_server.subprocess, "Popen",
                        _popen_finto(1, b"Traceback\nImportError: bum\n"))
    esito = mcp_server.visualize_session("sessione_x")
    assert esito["warnings"] == ["viewer_failed"]
    assert esito["result"] is None
    assert "ImportError: bum" in esito["explain"]


def test_un_viewer_ancora_vivo_risulta_avviato(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "DATA_ROOT", tmp_path)
    (tmp_path / "sessione_x.c3d").write_bytes(b"")
    _finge_dipendenze_presenti(monkeypatch)
    monkeypatch.setattr(mcp_server.subprocess, "Popen", _popen_finto(None))
    esito = mcp_server.visualize_session("sessione_x", show_coord_frames=True)
    assert esito["warnings"] == []
    assert esito["result"]["launched"] is True


def test_l_interruttore_impedisce_di_aprire_finestre(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "DATA_ROOT", tmp_path)
    (tmp_path / "sessione_x.c3d").write_bytes(b"")
    monkeypatch.setattr(mcp_server, "VIEWER_DISABLED", True)

    def _mai(*a, **k):
        raise AssertionError("con l'interruttore attivo non si lancia nulla")
    monkeypatch.setattr(mcp_server.subprocess, "Popen", _mai)
    monkeypatch.setattr(mcp_server.subprocess, "run", _mai)

    esito = mcp_server.visualize_session("sessione_x")
    assert esito["warnings"] == ["viewer_disabled"]
    assert esito["result"] is None


