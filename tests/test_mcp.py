"""Test del server MCP: ogni risposta deve dichiarare quanto pesa.

Copertura e provenienza vengono lette dal database insieme ai valori e devono
arrivare fino alla risposta. Le prove verificano la soglia oltre la quale scatta
l'avviso, il contratto comune a tutti gli strumenti, ispezionato staticamente sul
sorgente del server, e il comportamento di alcuni strumenti su dati simulati.

    python -m pytest tests/test_mcp.py -q

Prerequisiti: pytest e numpy; il file `src/mcp/server.py`, che viene letto come
testo per l'ispezione statica.
"""
import ast
import inspect
import os

from src.alerts import references
from src.mcp import core

# --- Percorsi ---
RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(RADICE, "src", "mcp", "server.py")


# --- Copertura e provenienza ---
# Che copertura e provenienza compaiano nella risposta, che l'avviso scatti sotto
# la soglia, che la soglia sia la stessa usata da chi genera gli alert e che
# l'avviso di modello riguardi solo i piani che dipendono da un'assunzione.
def test_la_copertura_arriva_fino_alla_risposta():
    sql = inspect.getsource(core.fetch_angle)
    assert "a.coverage" in sql, "la query non la chiede"
    assert sql.count("None, None, None, None") == 1
    assert "row[3]" in sql, "la legge ma non la restituisce"


def test_avviso_quando_la_copertura_e_bassa():
    def risposta(copertura):
        return core.response(
            {}, {"joint": "knee_l", "plane": "sagittal",
                 "provenance": "derived", "coverage": copertura},
            30.0, 1943, None, [], {"value": 128.7, "metric": "rom"}, "prova")

    bassa = risposta(0.17)
    assert "bassa_copertura" in bassa["warnings"]
    assert bassa["result"]["value"] == 128.7
    assert bassa["selected_data"]["coverage"] == 0.17

    assert "bassa_copertura" not in risposta(0.95)["warnings"]
    assert "bassa_copertura" not in risposta(None)["warnings"]


def test_stessa_soglia_di_chi_giudica():
    difetto = inspect.signature(references.valuta).parameters["min_coverage"].default
    assert core.MIN_COVERAGE_AFFIDABILE == difetto


def test_avviso_solo_sui_piani_che_dipendono_da_un_modello():
    def risposta(piano, provenienza):
        return core.response(
            {}, {"joint": "knee_r", "plane": piano, "provenance": provenienza,
                 "coverage": 1.0}, 100.0, 300, None, [], {"value": 1.0}, "prova")

    assert "modelled_axis" in risposta("transverse", "modelled")["warnings"]
    assert "modelled_axis" not in risposta("transverse", "measured")["warnings"]
    assert "modelled_axis" not in risposta("sagittal", "modelled")["warnings"]


# --- Contratto comune agli strumenti ---
# Lettura statica del sorgente del server per elencare gli strumenti esposti e
# verificare che ognuno passi dalla funzione di risposta e dichiari la copertura
# quando legge una serie.
def _strumenti():
    with open(SERVER) as f:
        testo = f.read()
    albero = ast.parse(testo, SERVER)
    righe = testo.splitlines()
    out = {}
    for nodo in albero.body:
        if not isinstance(nodo, ast.FunctionDef):
            continue
        decoratori = [ast.unparse(d) for d in nodo.decorator_list]
        if not any("mcp.tool" in d for d in decoratori):
            continue
        out[nodo.name] = "\n".join(righe[nodo.lineno - 1:nodo.end_lineno])
    return out


def test_ci_sono_gli_strumenti_attesi():
    nomi = set(_strumenti())
    assert len(nomi) >= 15, nomi
    for atteso in ("metric", "list_alerts", "find_similar_cycles",
                   "compare_session_shapes", "session_repeatability",
                   "side_shape_symmetry", "describe_session"):
        assert atteso in nomi, f"manca lo strumento {atteso}"


def test_ogni_strumento_passa_dal_contratto_di_risposta():
    fuori = [nome for nome, corpo in _strumenti().items()
             if "core.response" not in corpo and '"input_spec"' not in corpo]
    assert not fuori, f"strumenti fuori dal contratto: {fuori}"


def test_gli_strumenti_che_leggono_una_serie_dichiarano_la_copertura():
    with open(SERVER) as f:
        testo = f.read()
    assert "sig[\"coverage\"] = coverage" in testo, (
        "_load non propaga la copertura al segnale")
    corpi = "\n".join(_strumenti().values())
    assert "core.fetch_angle" not in corpi, (
        "uno strumento legge la serie senza passare da _load")


# --- Comportamento dei singoli strumenti ---
# Copertura di un confronto presa dalla peggiore delle serie, estremi dei cicli
# coerenti con la lettura, risposta a una metrica inesistente, unicita' dei nomi
# di metrica e appiattimento del testo libero.
def test_la_copertura_di_un_confronto_e_la_peggiore_non_l_ultima(monkeypatch):
    import numpy as np
    from src.mcp import server

    coperture = {"piena": 1.0, "scarsa": 0.10}

    def _load_finto(key, sig, conn=None, frame_start=None, frame_end=None):
        sig["provenance"] = "measured"
        sig["coverage"] = coperture[key]
        return (30.0, np.array([10.0, 20.0, 30.0]), np.ones(3, dtype=bool),
                3, (0, 3), [])

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(server, "_load", _load_finto)
    monkeypatch.setattr(server.core, "connect", lambda *a, **k: _Conn())

    for ordine in (["piena", "scarsa"], ["scarsa", "piena"]):
        esito = server.compare_sessions(ordine, "knee_l", "peak_max")
        assert "bassa_copertura" in esito["warnings"], ordine


def test_i_cicli_hanno_la_fine_esclusiva_come_la_lettura():
    import numpy as np
    from src.database import embeddings

    serie = [float(i) for i in range(120)]
    angoli = {("knee_l", "sagittal"): serie}

    righe, _ = embeddings.atoms_for_cycle(angoli, {}, 10, 40)
    assert righe, "il ciclo doveva produrre un atomo"
    assert righe[0]["peak"] == 39.0

    letto = core.prepare_series(serie, 10, 40)[0]
    assert len(letto) == 40 - 10
    assert float(np.nanmax(letto)) == 39.0


def test_una_metrica_inesistente_lo_dice(monkeypatch):
    from src.mcp import server

    def _mai(*a, **k):
        raise AssertionError("il controllo deve precedere qualunque accesso al DB")
    monkeypatch.setattr(server.core, "connect", _mai)
    monkeypatch.setattr(server, "_load", _mai)

    for chiamata in (
        lambda: server.metric("s", "knee_l", metric="peak_max"),
        lambda: server.symmetry_index("s", "knee", metric="peak_max"),
        lambda: server.compare_sessions(["s"], "knee_l", metric="peak_max"),
        lambda: server.compare_sports("running", "walking", "knee_l", metric="peak_max"),
    ):
        esito = chiamata()
        assert esito["warnings"] == ["unknown_metric"]
        assert esito["result"] is None
        for nome in core.METRICHE:
            assert nome in esito["explain"]


def test_i_nomi_di_metrica_sono_una_fonte_sola():
    import numpy as np
    serie = np.array([1.0, 2.0, 3.0])
    mask = np.ones(3, dtype=bool)
    for nome in core.METRICHE:
        assert core.scalar_metric(serie, mask, nome) is not None, nome
    assert core.scalar_metric(serie, mask, "non_esiste") is None


def test_il_testo_libero_arriva_appiattito():
    assert core.testo_citato("una\nnota\nsu piu' righe") == "una nota su piu' righe"
    assert core.testo_citato(None) is None
    lungo = core.testo_citato("x" * (core.MAX_TESTO + 50))
    assert len(lungo) == core.MAX_TESTO and lungo.endswith("…")
