"""Confronti fra andamenti articolari, calcolati da pgvector.

Tre domande, tre query. Tutte lavorano sulla stessa distanza coseno fra atomi:
su vettori a media nulla e norma unitaria vale 1 meno la correlazione, quindi
0 significa stesso andamento e 1 nessuna relazione.

  fra sessioni diverse   quali cicli somigliano a questo?  (`similar_cycles`)
                         quanto si somigliano due sessioni? (`compare_sessions`)
  dentro la stessa       quanto e' ripetibile il gesto?     (`repeatability`)
                         i due lati si somigliano?          (`symmetry`)

Il calcolo sta nel database e non in Python: e' il motivo per cui il progetto usa
PostgreSQL con pgvector invece di un archivio qualunque. Portare fuori i vettori
per confrontarli in memoria significherebbe leggerli tutti a ogni domanda.

Ogni risposta dichiara su quali atomi si e' basata e con quale provenienza: un
confronto fra una serie misurata e una modellata resta possibile, ma chi legge
deve saperlo.
"""
from . import embeddings

VERSION = embeddings.ENCODER_VERSION

# I cicli `whole_trial` sono esclusi da ogni confronto: quando il rilevamento
# non trova eventi, l'intera prova viene registrata come un ciclo unico, ma la
# sua forma dipende da dove la registrazione e' cominciata e finita, non dal
# gesto. Confrontarla darebbe una somiglianza priva di significato, e la
# somiglianza e' proprio cio' che queste funzioni promettono.

# Insiemi di giunti su cui ha senso chiedere un confronto d'insieme.
SCOPES = {
    "lower_body": ("hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r"),
    "upper_body": ("shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "neck"),
    "left_leg": ("hip_l", "knee_l", "ankle_l"),
    "right_leg": ("hip_r", "knee_r", "ankle_r"),
    "trunk": ("trunk", "spine", "neck"),
    "full_body": tuple(),          # riempito sotto: tutti i giunti noti
}


def resolve_scope(scope):
    """Nome di insieme o singolo giunto -> elenco di giunti."""
    from ..conversion import anatomy
    if scope == "full_body":
        return list(anatomy.JOINTS)
    if scope in SCOPES:
        return list(SCOPES[scope])
    return [scope] if scope in anatomy.JOINTS else []


def _filtro(giunti, piani):
    """Clausola e parametri per restringere gli atomi a giunti e piani."""
    return ("a.joint = ANY(%s) AND a.plane = ANY(%s)", [list(giunti), list(piani)])


def _righe(cur):
    colonne = [d[0] for d in cur.description]
    return [dict(zip(colonne, r)) for r in cur.fetchall()]


# --- Fra sessioni diverse ---
def similar_cycles(cur, session_key, cycle_index, scope="lower_body",
                   planes=("sagittal",), limit=10, min_atoms=2,
                   same_sport_only=True):
    """I cicli di altre sessioni con l'andamento piu' simile a quello dato.

    Il punteggio e' la media delle distanze coseno sugli atomi in comune: e'
    lecito perche' gli atomi sono unitari, e concatenarli darebbe esattamente la
    stessa media (vedi `embeddings.concat_equals_mean`).

    Il confronto guarda TUTTE le sessioni compatibili. C'e' stato un tetto di
    cinquanta candidate, scelte per data di caricamento, ed era un difetto: la
    data non ha nessuna relazione con la somiglianza, quindi su un archivio piu'
    grande la risposta conteneva le sessioni piu' recenti invece delle piu'
    simili. Misurato su 165 sessioni, i veri primi cinque vicini stavano a
    distanza 0.005 e il tetto restituiva 0.145 -- trenta volte peggio -- perche'
    le prove giuste erano fra le piu' vecchie.

    Il tetto non serviva nemmeno a risparmiare: la scansione esatta sull'intero
    archivio costa 6.8 ms (EXPLAIN ANALYZE, 165 sessioni e 11.481 atomi). E' cio'
    che la progettazione dice da sempre -- a questa scala la ricerca approssimata
    non serve, la giustificazione di pgvector non e' la velocita' ma il fatto che
    misura e rappresentazione vivano nello stesso posto.
    """
    giunti = resolve_scope(scope)
    if not giunti:
        return {"cycles": [], "warnings": ["scope_sconosciuto"], "query_atoms": 0}
    dove, params = _filtro(giunti, planes)
    # Il vincolo sullo sport si scrive o non si scrive: passarlo come parametro
    # booleano dentro `NOT %s` lascerebbe Postgres senza il tipo del parametro.
    vincolo_sport = ("AND s.sport IS NOT DISTINCT FROM rif.sport"
                     if same_sport_only else "")

    # Il metodo che ha prodotto il ciclo interrogato. Una falcata e un passo
    # coprono meta' gesto l'una dell'altro: confrontarne le forme paragona una
    # curva intera con una meta' curva, e la somiglianza che ne esce non dice
    # nulla sul gesto. Si confronta solo cio' che e' stato segmentato allo
    # stesso modo.
    cur.execute(
        """
        SELECT c.method FROM cycles c JOIN sessions s ON s.id = c.session_id
         WHERE s.session_key = %s AND c.cycle_index = %s
        """, [session_key, cycle_index])
    riga = cur.fetchone()
    if riga is None:
        return {"cycles": [], "warnings": ["ciclo_inesistente"], "query_atoms": 0}
    metodo = riga[0]
    if metodo == "whole_trial":
        return {"cycles": [], "warnings": ["ciclo_non_confrontabile"],
                "scope": giunti, "planes": list(planes), "metodo": metodo}

    # Quante sono davvero le compatibili. Lo sport da solo non basta: il
    # confronto ammette solo cicli segmentati con lo stesso metodo, quindi una
    # sessione che non ne ha nemmeno uno non e' una candidata. Contarla lo era:
    # la risposta annunciava sessioni confrontate e poi non restituiva nulla.
    cur.execute(
        f"""
        WITH rif AS (SELECT sport FROM sessions WHERE session_key = %s)
        SELECT count(*) FROM sessions s CROSS JOIN rif
         WHERE s.session_key <> %s {vincolo_sport}
           AND EXISTS (SELECT 1 FROM cycles c2
                        WHERE c2.session_id = s.id AND c2.method = %s)
        """, [session_key, session_key, metodo])
    compatibili = int(cur.fetchone()[0])

    cur.execute(
        f"""
        WITH q AS (
            SELECT a.joint, a.plane, a.shape, a.provenance
              FROM cycle_atoms a
              JOIN cycles c ON c.id = a.cycle_id AND c.method <> 'whole_trial'
              JOIN sessions s ON s.id = c.session_id
             WHERE s.session_key = %s AND c.cycle_index = %s
               AND a.encoder_version = %s AND {dove}
        ),
        rif AS (SELECT sport FROM sessions WHERE session_key = %s),
        candidate AS (
            SELECT s.id FROM sessions s CROSS JOIN rif
             WHERE s.session_key <> %s {vincolo_sport}
               AND EXISTS (SELECT 1 FROM cycles c2
                            WHERE c2.session_id = s.id AND c2.method = %s)
        )
        SELECT s.session_key, c.cycle_index, s.sport,
               avg(a.shape <=> q.shape)          AS distanza,
               count(*)                          AS n_atomi,
               min(a.coverage)                   AS copertura_minima,
               bool_and(a.provenance = q.provenance) AS stessa_provenienza
          FROM cycle_atoms a
          JOIN q ON q.joint = a.joint AND q.plane = a.plane
          JOIN cycles c ON c.id = a.cycle_id AND c.method = %s
          JOIN candidate cd ON cd.id = c.session_id
          JOIN sessions s ON s.id = c.session_id
         WHERE a.encoder_version = %s
      GROUP BY s.session_key, c.cycle_index, s.sport
        HAVING count(*) >= %s
      ORDER BY distanza
         LIMIT %s
        """,
        [session_key, cycle_index, VERSION] + params
        + [session_key, session_key, metodo,
           metodo, VERSION, min_atoms, limit])
    righe = _righe(cur)

    avvisi = []
    if any(not r["stessa_provenienza"] for r in righe):
        avvisi.append("provenienze_diverse")
    return {"cycles": righe, "warnings": avvisi, "scope": giunti,
            "planes": list(planes), "metodo": metodo,
            "sessioni_compatibili": compatibili,
            "sessioni_confrontate": compatibili}


def compare_sessions(cur, key_a, key_b, scope="lower_body", planes=("sagittal",)):
    """Quanto si somigliano due sessioni, giunto per giunto.

    Confronta ogni ciclo di A con ogni ciclo di B e ne prende la distanza
    mediana: piu' robusta della media quando un ciclo e' sporco.
    """
    giunti = resolve_scope(scope)
    if not giunti:
        return {"joints": [], "warnings": ["scope_sconosciuto"]}
    dove_a = "a.joint = ANY(%s) AND a.plane = ANY(%s)"

    cur.execute(
        f"""
        WITH aa AS (
            SELECT a.joint, a.plane, a.shape, a.rom, a.provenance
              FROM cycle_atoms a
              JOIN cycles c ON c.id = a.cycle_id AND c.method <> 'whole_trial'
              JOIN sessions s ON s.id = c.session_id
             WHERE s.session_key = %s AND a.encoder_version = %s AND {dove_a}
        ),
        bb AS (
            SELECT a.joint, a.plane, a.shape, a.rom, a.provenance
              FROM cycle_atoms a
              JOIN cycles c ON c.id = a.cycle_id AND c.method <> 'whole_trial'
              JOIN sessions s ON s.id = c.session_id
             WHERE s.session_key = %s AND a.encoder_version = %s AND {dove_a}
        )
        SELECT aa.joint, aa.plane,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY aa.shape <=> bb.shape)
                                                            AS distanza_mediana,
               count(*)                                     AS n_confronti,
               avg(aa.rom) AS rom_a, avg(bb.rom) AS rom_b,
               bool_and(aa.provenance = bb.provenance)      AS stessa_provenienza
          FROM aa JOIN bb ON bb.joint = aa.joint AND bb.plane = aa.plane
      GROUP BY aa.joint, aa.plane
      ORDER BY aa.joint, aa.plane
        """,
        [key_a, VERSION, list(giunti), list(planes),
         key_b, VERSION, list(giunti), list(planes)])
    righe = _righe(cur)

    avvisi = []
    if not righe:
        avvisi.append("nessun_atomo_in_comune")
    if any(not r["stessa_provenienza"] for r in righe):
        avvisi.append("provenienze_diverse")
    return {"joints": righe, "warnings": avvisi, "scope": giunti,
            "planes": list(planes)}


# --- Dentro la stessa sessione ---
def repeatability(cur, session_key, scope="lower_body", planes=("sagittal",)):
    """Quanto si somigliano fra loro i cicli della stessa sessione.

    Distanza mediana fra tutte le coppie di cicli distinti, giunto per giunto.
    Un valore basso dice che il gesto si ripete uguale; uno alto che cambia da
    un ciclo all'altro -- per variabilita' vera o per qualita' del tracciamento,
    e la copertura riportata accanto aiuta a distinguerle.
    """
    giunti = resolve_scope(scope)
    if not giunti:
        return {"joints": [], "warnings": ["scope_sconosciuto"]}

    # Le distanze sono grandezze di coppia, il ROM e la copertura sono grandezze
    # di ciclo. Calcolarle insieme sul join triangolare pesava il primo ciclo
    # n-1 volte e l'ultimo zero: bastava che l'anomalia fosse nell'ultimo ciclo
    # perche' la sessione risultasse "perfettamente ripetibile", cioe' l'opposto
    # di cio' che questa funzione serve a dire. Quindi due aggregazioni separate,
    # unite alla fine.
    cur.execute(
        """
        WITH atomi AS (
            SELECT c.cycle_index, a.joint, a.plane, a.shape, a.rom,
                   a.coverage, a.provenance
              FROM cycle_atoms a
              JOIN cycles c ON c.id = a.cycle_id AND c.method <> 'whole_trial'
              JOIN sessions s ON s.id = c.session_id
             WHERE s.session_key = %s AND a.encoder_version = %s
               AND a.joint = ANY(%s) AND a.plane = ANY(%s)
        ),
        coppie AS (
            SELECT x.joint, x.plane,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY x.shape <=> y.shape)
                                                    AS distanza_mediana,
                   max(x.shape <=> y.shape)         AS distanza_massima,
                   count(*)                         AS n_coppie
              FROM atomi x
              JOIN atomi y ON y.joint = x.joint AND y.plane = x.plane
                          AND y.cycle_index > x.cycle_index
          GROUP BY x.joint, x.plane
        ),
        cicli AS (
            SELECT joint, plane,
                   count(DISTINCT cycle_index)      AS n_cicli,
                   stddev_samp(rom)                 AS rom_deviazione,
                   avg(rom)                         AS rom_medio,
                   min(coverage)                    AS copertura_minima,
                   min(provenance)                  AS provenienza
              FROM atomi
          GROUP BY joint, plane
        )
        -- LEFT JOIN: con un solo ciclo non esistono coppie, ma il giunto va
        -- riportato lo stesso con n_cicli = 1 e distanze nulle, invece di
        -- sparire dal risultato come accadeva prima.
        SELECT c.joint, c.plane,
               p.distanza_mediana, p.distanza_massima,
               coalesce(p.n_coppie, 0) AS n_coppie,
               c.n_cicli, c.rom_deviazione, c.rom_medio,
               c.copertura_minima, c.provenienza
          FROM cicli c
          LEFT JOIN coppie p ON p.joint = c.joint AND p.plane = c.plane
      ORDER BY c.joint, c.plane
        """, (session_key, VERSION, list(giunti), list(planes)))
    righe = _righe(cur)
    # Con il LEFT JOIN una sessione a ciclo unico produce comunque una riga
    # (n_cicli = 1, n_coppie = 0): senza questo controllo l'avviso sparirebbe e
    # sembrerebbe che la ripetibilita' sia stata misurata. Serve almeno una
    # coppia perche' la domanda abbia risposta.
    misurabile = any((r.get("n_coppie") or 0) > 0 for r in righe)
    avvisi = [] if misurabile else ["cicli_insufficienti"]
    return {"joints": righe, "warnings": avvisi, "scope": giunti,
            "planes": list(planes)}


# Coppie di giunti omologhi: il confronto di simmetria e' fra questi.
_OMOLOGHI = (("hip_l", "hip_r"), ("knee_l", "knee_r"), ("ankle_l", "ankle_r"),
             ("shoulder_l", "shoulder_r"), ("elbow_l", "elbow_r"))


def symmetry(cur, session_key, planes=("sagittal",)):
    """Quanto si somigliano il lato destro e il sinistro, nella stessa sessione.

    Confronta gli atomi omologhi ciclo per ciclo. Attenzione a leggerlo: nel
    cammino e nella corsa i due arti sono in controfase, quindi ci si aspetta una
    distanza alta per ragioni di tempistica e non di asimmetria. Il confronto e'
    quindi utile soprattutto fra sessioni o su gesti simmetrici, e la distanza va
    letta come "quanto le due curve hanno la stessa forma", non come un indice
    clinico di asimmetria.
    """
    coppie = [c for c in _OMOLOGHI]
    cur.execute(
        """
        WITH atomi AS (
            SELECT c.cycle_index, a.joint, a.plane, a.shape, a.rom,
                   a.coverage, a.provenance
              FROM cycle_atoms a
              JOIN cycles c ON c.id = a.cycle_id AND c.method <> 'whole_trial'
              JOIN sessions s ON s.id = c.session_id
             WHERE s.session_key = %s AND a.encoder_version = %s
               AND a.plane = ANY(%s)
        ),
        coppie AS (SELECT * FROM unnest(%s::text[], %s::text[]) AS t(sx, dx))
        SELECT p.sx, p.dx, l.plane,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY l.shape <=> r.shape)
                                                    AS distanza_mediana,
               avg(l.rom) AS rom_sinistra, avg(r.rom) AS rom_destra,
               avg(abs(l.rom - r.rom))              AS differenza_rom,
               count(*)                             AS n_cicli,
               least(min(l.coverage), min(r.coverage)) AS copertura_minima
          FROM coppie p
          JOIN atomi l ON l.joint = p.sx
          JOIN atomi r ON r.joint = p.dx AND r.cycle_index = l.cycle_index
                                         AND r.plane = l.plane
      GROUP BY p.sx, p.dx, l.plane
      ORDER BY p.sx, l.plane
        """,
        (session_key, VERSION, list(planes),
         [c[0] for c in coppie], [c[1] for c in coppie]))
    righe = _righe(cur)
    avvisi = ["arti_in_controfase_nel_passo"] if righe else ["nessuna_coppia"]
    return {"pairs": righe, "warnings": avvisi, "planes": list(planes)}
