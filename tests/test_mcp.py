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
    assert len(nomi) >= 17, nomi
    for atteso in ("metric", "list_alerts", "find_similar_cycles",
                   "compare_session_shapes", "session_repeatability",
                   "side_shape_symmetry", "describe_session",
                   "list_sources", "ingest_session"):
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


# --- Scrittura dalla chat: list_sources e ingest_session ---
# La connessione resta di sola lettura: la scrittura passa da sottoprocessi
# finti, mai da un vero database. Si verifica che l'interruttore spento blocchi
# tutto, che la DSN non finisca negli argomenti, che un sottoprocesso a codice
# di uscita 0 non basti a dichiarare riuscito senza la verifica sul database, e
# che i tre stati di list_sources() non si confondano.
class _CursoreIngestFinto:
    def __init__(self, chiavi_in_db=(), ricevuta=None):
        self._chiavi = set(chiavi_in_db)
        self._ricevuta = ricevuta
        self.istruzioni = []

    def execute(self, sql, params=None):
        self.istruzioni.append((" ".join(sql.split()), params))
        self._ultimo = sql

    def fetchall(self):
        if "session_key = ANY" in self._ultimo:
            return [(k,) for k in self._chiavi]
        return []

    def fetchone(self):
        if "n_frames" in self._ultimo:
            return self._ricevuta
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _ConnessioneIngestFinta:
    def __init__(self, cur):
        self._cur = cur

    def cursor(self):
        return self._cur

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _prepara_ingest(monkeypatch, server, tmp_path, chiavi_in_db=(), ricevuta=None):
    monkeypatch.setattr(server, "INGEST_ENABLED", True)
    monkeypatch.setattr(server, "DATA_ROOT", tmp_path / "elaborati")
    (tmp_path / "elaborati").mkdir()
    grezzi = tmp_path / "grezzi"
    grezzi.mkdir()
    (grezzi / "sessione_x.c3d").write_bytes(b"")
    monkeypatch.setattr(server, "INGEST_ROOTS", [str(grezzi)])
    monkeypatch.setattr(server, "INGEST_LOG", tmp_path / "ingest.log")
    cur = _CursoreIngestFinto(chiavi_in_db, ricevuta)
    monkeypatch.setattr(server.core, "connect",
                        lambda *a, **k: _ConnessioneIngestFinta(cur))
    return grezzi, cur


def test_l_interruttore_spento_blocca_scrittura_e_lettura(monkeypatch):
    from src.mcp import server

    def _mai(*a, **k):
        raise AssertionError("con l'interruttore spento non si tocca nulla")
    monkeypatch.setattr(server.subprocess, "run", _mai)
    monkeypatch.setattr(server.core, "connect", _mai)

    assert server.list_sources()["warnings"] == ["ingest_disabled"]
    esito = server.ingest_session("qualsiasi")
    assert esito["warnings"] == ["ingest_disabled"]
    assert esito["result"] is None


def test_list_sources_distingue_i_tre_stati(tmp_path, monkeypatch):
    from src.mcp import server
    grezzi, _ = _prepara_ingest(monkeypatch, server, tmp_path,
                                chiavi_in_db=["caricata"])
    for nome in ("caricata", "convertita"):
        (grezzi / f"{nome}.c3d").write_bytes(b"")
    (tmp_path / "elaborati" / "caricata.c3d").write_bytes(b"")
    (tmp_path / "elaborati" / "convertita.c3d").write_bytes(b"")

    esito = server.list_sources()
    stati = {r["source"]: r["stato"] for r in esito["result"]["sorgenti"]}
    assert stati["caricata"] == "nel_database"
    assert stati["convertita"] == "convertita_non_caricata"
    assert stati["sessione_x"] == "da_convertire"


def test_la_dsn_non_passa_dalla_riga_di_comando(tmp_path, monkeypatch):
    from src.mcp import server
    grezzi, _ = _prepara_ingest(monkeypatch, server, tmp_path,
                                ricevuta=(40, "2026-01-01"))
    monkeypatch.setattr(server, "INGEST_DB_DSN", "dbname=segreta password=segreta")

    comandi = []

    def _run_finto(cmd, cwd=None, capture_output=None, text=None, timeout=None,
                   env=None):
        comandi.append((cmd, env))
        class _Esito:
            returncode = 0
            stdout = ""
            stderr = ""
        return _Esito()

    def _sport_presente(c3d):
        return "running"

    monkeypatch.setattr(server.subprocess, "run", _run_finto)
    monkeypatch.setattr(server, "_sport_del_canonico", _sport_presente)

    server.ingest_session("sessione_x", sport="running", subject_id="atleta01")
    assert len(comandi) == 2, "conversione e caricamento sono due sottoprocessi"
    for cmd, env in comandi:
        for pezzo in cmd:
            assert "segreta" not in pezzo, f"la DSN e' finita negli argomenti: {cmd}"
    assert any("segreta" in (env or {}).get("INGEST_DB_DSN", "")
              for _, env in comandi), "la DSN deve arrivare via ambiente"


def test_sport_mancante_ferma_prima_del_caricamento(tmp_path, monkeypatch):
    from src.mcp import server
    grezzi, cur = _prepara_ingest(monkeypatch, server, tmp_path)

    chiamate = []

    def _run_finto(cmd, cwd=None, capture_output=None, text=None, timeout=None,
                   env=None):
        chiamate.append(cmd)
        class _Esito:
            returncode = 0
            stdout, stderr = "", ""
        return _Esito()

    monkeypatch.setattr(server.subprocess, "run", _run_finto)
    monkeypatch.setattr(server, "_sport_del_canonico", lambda c3d: None)

    esito = server.ingest_session("sessione_x")
    assert esito["warnings"] == ["sport_mancante"]
    assert esito["result"] is None
    assert "running" in esito["explain"] or "running" in esito["azione_suggerita"]
    assert len(chiamate) == 1, "il caricamento non deve partire senza un gesto"
    assert not any("session_key = ANY" in i[0] or "n_frames" in i[0]
                  for i in cur.istruzioni), "non deve nemmeno interrogare il database"


def test_only_convert_salta_il_caricamento_anche_senza_sport(tmp_path, monkeypatch):
    from src.mcp import server
    grezzi, _ = _prepara_ingest(monkeypatch, server, tmp_path)

    chiamate = []

    def _run_finto(cmd, cwd=None, capture_output=None, text=None, timeout=None,
                   env=None):
        chiamate.append(cmd)
        class _Esito:
            returncode = 0
            stdout, stderr = "", ""
        return _Esito()

    monkeypatch.setattr(server.subprocess, "run", _run_finto)
    monkeypatch.setattr(server, "_sport_del_canonico", lambda c3d: None)

    esito = server.ingest_session("sessione_x", only_convert=True)
    assert esito["warnings"] == []
    assert esito["result"] == {"convertita": True, "caricata": False}
    assert len(chiamate) == 1


def test_la_conversione_fallita_si_vede_in_chat(tmp_path, monkeypatch):
    from src.mcp import server
    grezzi, _ = _prepara_ingest(monkeypatch, server, tmp_path)

    def _run_finto(cmd, cwd=None, capture_output=None, text=None, timeout=None,
                   env=None):
        class _Esito:
            returncode = 1
            stdout = ""
            stderr = "Traceback\nValueError: gesto non riconosciuto: 'jogging'\n"
        return _Esito()

    monkeypatch.setattr(server.subprocess, "run", _run_finto)

    esito = server.ingest_session("sessione_x", sport="jogging")
    assert esito["result"] is None
    assert esito["selected_data"]["esito"] == "fallito"
    assert esito["selected_data"]["passo_fallito"] == "conversione"
    assert "gesto non riconosciuto" in esito["errore"]
    assert "gesto non riconosciuto" in esito["explain"] or esito["errore"] in esito["explain"]
    assert esito["warnings"] == ["conversione_fallita"]


def test_un_sottoprocesso_a_zero_non_basta_senza_riscontro_nel_database(
        tmp_path, monkeypatch):
    # Il caso che il codice di uscita da solo non racconta: il caricamento non
    # e' esploso, ma la sessione non compare in sessions. Lo strumento deve
    # verificare, non fidarsi del codice di uscita.
    from src.mcp import server
    grezzi, _ = _prepara_ingest(monkeypatch, server, tmp_path, ricevuta=None)

    def _run_finto(cmd, cwd=None, capture_output=None, text=None, timeout=None,
                   env=None):
        class _Esito:
            returncode = 0
            stdout, stderr = "", ""
        return _Esito()

    monkeypatch.setattr(server.subprocess, "run", _run_finto)
    monkeypatch.setattr(server, "_sport_del_canonico", lambda c3d: "running")

    esito = server.ingest_session("sessione_x", sport="running")
    assert esito["result"] is None
    assert esito["selected_data"]["esito"] == "fallito"
    assert esito["selected_data"]["passo_fallito"] == "verifica"
    assert esito["warnings"] == ["verifica_fallita"]


def test_una_sessione_verificata_avvisa_sul_soggetto_non_controllato(
        tmp_path, monkeypatch):
    from src.mcp import server
    grezzi, _ = _prepara_ingest(monkeypatch, server, tmp_path,
                                ricevuta=(120, "2026-01-01T10:00:00+00:00"))

    def _run_finto(cmd, cwd=None, capture_output=None, text=None, timeout=None,
                   env=None):
        class _Esito:
            returncode = 0
            stdout, stderr = "", ""
        return _Esito()

    monkeypatch.setattr(server.subprocess, "run", _run_finto)
    monkeypatch.setattr(server, "_sport_del_canonico", lambda c3d: "running")

    con_soggetto = server.ingest_session("sessione_x", sport="running",
                                         subject_id="atleta01")
    assert con_soggetto["warnings"] == ["subject_id_non_verificato"]
    assert con_soggetto["result"] == {"convertita": True, "caricata": True}
    assert con_soggetto["selected_data"]["n_frames"] == 120

    senza_soggetto = server.ingest_session("sessione_x", sport="running")
    assert senza_soggetto["warnings"] == ["sessione_senza_soggetto"]


def test_una_sorgente_fuori_dalle_radici_ammesse_e_rifiutata(tmp_path, monkeypatch):
    from src.mcp import server
    _prepara_ingest(monkeypatch, server, tmp_path)
    altrove = tmp_path / "altrove"
    altrove.mkdir()
    (altrove / "clandestina.c3d").write_bytes(b"")

    def _mai(*a, **k):
        raise AssertionError("una sorgente fuori radice non deve avviare nulla")
    monkeypatch.setattr(server.subprocess, "run", _mai)

    esito = server.ingest_session(str(altrove / "clandestina.c3d"))
    assert esito["warnings"] == ["sorgente_non_valida"]
    assert esito["result"] is None


def test_il_log_registra_ogni_tentativo(tmp_path, monkeypatch):
    from src.mcp import server
    grezzi, _ = _prepara_ingest(monkeypatch, server, tmp_path,
                                ricevuta=(10, "2026-01-01"))

    def _run_finto(cmd, cwd=None, capture_output=None, text=None, timeout=None,
                   env=None):
        class _Esito:
            returncode = 0
            stdout, stderr = "", ""
        return _Esito()

    monkeypatch.setattr(server.subprocess, "run", _run_finto)
    monkeypatch.setattr(server, "_sport_del_canonico", lambda c3d: "running")

    server.ingest_session("sessione_x", sport="running")
    contenuto = server.INGEST_LOG.read_text(encoding="utf-8")
    righe = [r for r in contenuto.splitlines() if r]
    assert len(righe) == 2, "conversione e caricamento sono due righe"
    assert all("sessione_x" in r for r in righe)
