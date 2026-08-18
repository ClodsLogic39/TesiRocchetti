"""Test del viewer: lettura del C3D canonico e percorso di lancio dall'MCP.

Non serve Open3D: il layer dati del viewer e' separato dal rendering, ed e'
proprio la parte che si era rotta quando il formato canonico e' cambiato.

    python -m pytest tests/test_viewer.py -q
"""
import numpy as np
import pytest

from src.conversion import anatomy, canonical, process, readers
from src.mcp import server as mcp_server
from src.visualization import viewer_open3d

# Stessa posa di riferimento dei test di conversione: in piedi, fermo.
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
    """Scrive un C3D canonico, opzionalmente senza alcuni punti."""
    punti = {k: np.tile(np.asarray(v, dtype=float), (n, 1))
             for k, v in _IN_PIEDI.items() if k not in esclusi}
    letto = readers._finalise(punti, {"fps": 30.0, "n_frames": n, "source": "prova",
                                      "source_format": "sintetico",
                                      "capture_system": "prova", "sport": "test"})
    return canonical.write(process.analizza(letto), str(tmp_path / "prova.c3d"))


def test_carica_ricostruisce_le_ossa_dai_punti_canonici(tmp_path):
    dati = viewer_open3d.carica(_c3d(tmp_path))
    # Il C3D non contiene piu' i canali `{segmento}_start`: le ossa nascono dai
    # punti attraverso anatomy.SEGMENTS. "trunk" fa eccezione: i suoi estremi
    # (hip_c, shoulder_c) sono centri interni mai scritti come punti, quindi
    # non puo' comparire come osso singolo -- si vede tramite il busto.
    assert {o[0] for o in dati["ossa"]} == set(anatomy.SEGMENTS) - {"trunk"}
    assert dati["n_frame"] == 40
    assert dati["fps"] == 30.0
    for _, prossimale, distale, colore in dati["ossa"]:
        assert prossimale in dati["punti"] and distale in dati["punti"]
        assert len(colore) == 3 and all(0.0 <= c <= 1.0 for c in colore)


def test_il_busto_poggia_sui_quattro_punti_reali(tmp_path):
    """Il tronco si disegna se ci sono i punti che lo delimitano, non altrimenti.

    Il volume si costruisce sui quattro centri articolari di anche e spalle:
    "trunk" non compare fra le ossa, perche' i suoi estremi -- hip_c e
    shoulder_c -- sono centri interni che la conversione non scrive mai.
    """
    dati = viewer_open3d.carica(_c3d(tmp_path))
    assert dati["ha_busto"]
    assert all(k in dati["punti"]
               for k in ("hip_l", "hip_r", "shoulder_l", "shoulder_r"))
    assert "trunk" not in {o[0] for o in dati["ossa"]}
    # gli angoli del tronco, veri, sono in legenda anche se "trunk" non e' un
    # osso disegnabile
    assert "trunk" in dati["angoli"] and "spine" in dati["angoli"]
    assert dati["angoli"]["trunk"][0] == "trunk"
    assert dati["angoli"]["spine"][0] == "spine"


def test_il_busto_manca_se_mancano_i_punti_delle_spalle(tmp_path):
    dati = viewer_open3d.carica(_c3d(tmp_path, esclusi=("shoulder_l", "shoulder_r")))
    assert not dati["ha_busto"]


def test_i_segmenti_senza_estremi_vengono_saltati_non_esplodono(tmp_path):
    """Dati parziali: una ripresa che non vede le gambe resta visualizzabile."""
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
        # "trunk" e "spine" condividono lo stesso segmento distale ("trunk"):
        # un dizionario piatto indicizzato per segmento non potrebbe portarli
        # entrambi, quindi per questi due la chiave e' gia' il nome del
        # giunto invece che del segmento.
        if chiave in ("trunk", "spine"):
            assert giunto == chiave
        else:
            assert anatomy.JOINTS[giunto]["distal"] == chiave
        assert len(serie) == dati["n_frame"]


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


# --- Lancio dall'MCP ---
def test_visualize_session_trova_il_c3d_del_convertitore(tmp_path, monkeypatch):
    """La cartella cercata deve essere quella in cui scrive il convertitore."""
    monkeypatch.setattr(mcp_server, "DATA_ROOT", tmp_path)
    (tmp_path / "sessione_x.c3d").write_bytes(b"")
    # Interprete senza open3d: ci fermiamo prima del lancio, ma dopo il
    # controllo sul percorso, che e' cio' che qui interessa.
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
    """Un viewer che esce subito, come faceva davvero prima della correzione."""

    def __init__(self, uscita, errore=b"", log=None):
        self._uscita = uscita
        self.pid = 999
        # Lo stderr non e' piu' una pipe ma un file temporaneo passato dal
        # chiamante: il finto processo ci scrive dentro, come farebbe il vero.
        if log is not None and errore:
            log.write(errore)
        self._log = log

    def wait(self, timeout=None):
        if self._uscita is None:
            raise mcp_server.subprocess.TimeoutExpired("viewer", timeout)
        return self._uscita


def _popen_finto(uscita, errore=b""):
    """Sostituto di Popen che scrive su `stderr=` come fa il processo vero."""
    def _crea(*a, **k):
        return _ProcessoFinto(uscita, errore, log=k.get("stderr"))
    return _crea


def test_un_viewer_che_muore_subito_non_viene_dato_per_avviato(tmp_path, monkeypatch):
    """Il bug peggiore: annunciare una finestra che l'utente non vedra' mai."""
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
    """Su un server condiviso il viewer non deve toccare lo schermo dell'host."""
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


# `render_frame` e il suo test sono stati rimossi per ora: scriveva il PNG
# nella cartella temporanea di sistema della macchina che ospita il server,
# un percorso che l'agente non ha modo di leggere (vedi commento in
# src/mcp/server.py). Va reintrodotto con una destinazione condivisa, o
# restituendo i byte PNG direttamente nella risposta.
