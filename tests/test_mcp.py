"""Il server MCP deve dire quanto pesa ogni numero che restituisce.

E' il contratto di onestà applicato allo strato di interrogazione: la copertura
e la provenienza sono lette dal database insieme ai valori, e devono arrivare
fino alla risposta. Perderle per strada significa presentare come misura del
gesto quella che è la misura di un sottoinsieme di fotogrammi.
"""
import ast
import inspect
import os

from src.alerts import references
from src.mcp import core

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(RADICE, "src", "mcp", "server.py")


def test_la_copertura_arriva_fino_alla_risposta():
    """Letta dal database, non deve fermarsi a meta' strada."""
    sql = inspect.getsource(core.fetch_angle)
    assert "a.coverage" in sql, "la query non la chiede"
    # e la funzione la restituisce: quattro valori, non tre
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
    # il valore non viene alterato: si dichiara, non si nasconde
    assert bassa["result"]["value"] == 128.7
    assert bassa["selected_data"]["coverage"] == 0.17

    assert "bassa_copertura" not in risposta(0.95)["warnings"]
    assert "bassa_copertura" not in risposta(None)["warnings"]


def test_stessa_soglia_di_chi_giudica():
    """Il server e il caricamento devono essere d'accordo su cosa e' troppo poco.

    Se la soglia dell'uno fosse piu' larga di quella dell'altro, ci sarebbe una
    fascia in cui il caricamento rifiuta di giudicare un angolo e il server lo
    riporta senza dire nulla: due comportamenti opposti sullo stesso dato.
    """
    difetto = inspect.signature(references.valuta).parameters["min_coverage"].default
    assert core.MIN_COVERAGE_AFFIDABILE == difetto


def test_avviso_solo_sui_piani_che_dipendono_da_un_modello():
    def risposta(piano, provenienza):
        return core.response(
            {}, {"joint": "knee_r", "plane": piano, "provenance": provenienza,
                 "coverage": 1.0}, 100.0, 300, None, [], {"value": 1.0}, "prova")

    # un trasverso che il modello non osserva va segnalato
    assert "modelled_axis" in risposta("transverse", "modelled")["warnings"]
    # ma se la sorgente lo forniva a tre gradi di liberta' e' una misura
    assert "modelled_axis" not in risposta("transverse", "measured")["warnings"]
    # il sagittale non si segnala mai
    assert "modelled_axis" not in risposta("sagittal", "modelled")["warnings"]


def _strumenti():
    """Nomi e corpo delle funzioni decorate con @mcp.tool, lette dal sorgente.

    Si legge il file invece di importarlo: il pacchetto `mcp` non e' installato
    ovunque, e questo controllo non ha bisogno di eseguire il server.
    """
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
    """Nessuno strumento deve costruirsi la risposta a mano.

    Il contratto -- input_spec, selected_data, result, explain, warnings -- e'
    la garanzia di esplicitezza su cui il progetto si regge: uno strumento che
    restituisce un dizionario suo la aggira in silenzio.
    """
    fuori = [nome for nome, corpo in _strumenti().items()
             if "core.response" not in corpo and '"input_spec"' not in corpo]
    assert not fuori, f"strumenti fuori dal contratto: {fuori}"


def test_gli_strumenti_che_leggono_una_serie_dichiarano_la_copertura():
    """Chi carica una serie deve passare per `_load`, che porta la copertura."""
    with open(SERVER) as f:
        testo = f.read()
    assert "sig[\"coverage\"] = coverage" in testo, (
        "_load non propaga la copertura al segnale")
    # e nessuno deve leggere gli angoli scavalcando _load
    corpi = "\n".join(_strumenti().values())
    assert "core.fetch_angle" not in corpi, (
        "uno strumento legge la serie senza passare da _load")


def test_la_copertura_di_un_confronto_e_la_peggiore_non_l_ultima(monkeypatch):
    """Un confronto pesa quanto la sessione peggiore, in qualunque ordine.

    `sig` viene riusato a ogni giro del ciclo e sovrascritto da `_load`: senza
    ripararlo alla fine, l'avviso di bassa copertura dipenderebbe da quale
    sessione e' stata letta per ultima, cioe' dall'ordine degli argomenti.
    """
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
    """Scrittura e lettura devono affettare la stessa finestra.

    L'ingest scriveva `[inizio:fine+1]` mentre il server legge `[s:e)`: gli
    stessi start/end restituiti da `list_cycles`, ridati a un altro strumento,
    producevano una finestra diversa da quella con cui erano state calcolate le
    metriche del ciclo.
    """
    import numpy as np
    from src.database import embeddings

    # Serie in cui il valore coincide con l'indice: il massimo del tratto dice
    # da solo quale sia stato l'ultimo fotogramma incluso.
    serie = [float(i) for i in range(120)]
    angoli = {("knee_l", "sagittal"): serie}

    righe, _ = embeddings.atoms_for_cycle(angoli, {}, 10, 40)
    assert righe, "il ciclo doveva produrre un atomo"
    # 39, non 40: il picco che chiude il ciclo appartiene al successivo.
    assert righe[0]["peak"] == 39.0

    # E il server legge esattamente la stessa finestra.
    letto = core.prepare_series(serie, 10, 40)[0]
    assert len(letto) == 40 - 10
    assert float(np.nanmax(letto)) == 39.0


def test_una_metrica_inesistente_lo_dice(monkeypatch):
    """Un nome inventato non deve somigliare a un'assenza di dati.

    Prima, `scalar_metric` restituiva None sia per un nome fuori elenco sia per
    un segnale senza fotogrammi validi: chi legge non poteva distinguere "quella
    metrica non esiste" da "non ci sono dati", e l'explain ripeteva il nome
    inesistente come se fosse legittimo.
    """
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
        # l'explain deve elencare i nomi validi, non ripetere quello inventato
        for nome in core.METRICHE:
            assert nome in esito["explain"]


def test_i_nomi_di_metrica_sono_una_fonte_sola():
    """L'elenco dichiarato e quello calcolato non devono poter divergere."""
    import numpy as np
    serie = np.array([1.0, 2.0, 3.0])
    mask = np.ones(3, dtype=bool)
    for nome in core.METRICHE:
        assert core.scalar_metric(serie, mask, nome) is not None, nome
    assert core.scalar_metric(serie, mask, "non_esiste") is None


def test_il_testo_libero_arriva_appiattito():
    """Note e sorgenti sono dati citati, non blocchi che sembrano istruzioni."""
    assert core.testo_citato("una\nnota\nsu piu' righe") == "una nota su piu' righe"
    assert core.testo_citato(None) is None
    lungo = core.testo_citato("x" * (core.MAX_TESTO + 50))
    assert len(lungo) == core.MAX_TESTO and lungo.endswith("…")
