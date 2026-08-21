"""Confronti fra andamenti articolari, calcolati da pgvector.

Quattro domande, tutte basate sulla distanza coseno fra atomi di forma: su
vettori a media nulla e norma unitaria vale 1 meno la correlazione, quindi 0
significa stesso andamento e 1 nessuna relazione.

  fra sessioni diverse   quali cicli somigliano a questo   (`similar_cycles`)
                         quanto si somigliano due sessioni (`compare_sessions`)
  dentro la stessa       quanto e' ripetibile il gesto     (`repeatability`)
                         quanto si somigliano i due lati   (`symmetry`)

Il calcolo avviene nel database e non in Python. Ogni risposta dichiara su quali
atomi si e' basata e con quale provenienza.

Prerequisiti: un database PostgreSQL con l'estensione pgvector e gli atomi gia'
scritti dall'ingest; le funzioni ricevono un cursore aperto da chi le chiama.
"""
from . import embeddings

# --- Insiemi di giunti e filtri comuni ---
# Versione degli atomi su cui filtrano tutte le query, gruppi di giunti su cui ha
# senso chiedere un confronto d'insieme e le clausole condivise dalle query.
VERSION = embeddings.ENCODER_VERSION


SCOPES = {
    "lower_body": ("hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r"),
    "upper_body": ("shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "neck"),
    "left_leg": ("hip_l", "knee_l", "ankle_l"),
    "right_leg": ("hip_r", "knee_r", "ankle_r"),
    "trunk": ("trunk", "spine", "neck"),
    "full_body": tuple(),
}


def resolve_scope(scope):
    from ..conversion import anatomy
    if scope == "full_body":
        return list(anatomy.JOINTS)
    if scope in SCOPES:
        return list(SCOPES[scope])
    return [scope] if scope in anatomy.JOINTS else []


def _filtro(giunti, piani):
    return ("a.joint = ANY(%s) AND a.plane = ANY(%s)", [list(giunti), list(piani)])


def _righe(cur):
    colonne = [d[0] for d in cur.description]
    return [dict(zip(colonne, r)) for r in cur.fetchall()]


# --- Confronti fra sessioni diverse ---
# I cicli di altre sessioni piu' vicini a un ciclo dato, e la distanza media fra
# due sessioni intere.
def similar_cycles(cur, session_key, cycle_index, scope="lower_body",
                   planes=("sagittal",), limit=10, min_atoms=2,
                   same_sport_only=True):
    giunti = resolve_scope(scope)
    if not giunti:
        return {"cycles": [], "warnings": ["scope_sconosciuto"], "query_atoms": 0}
    dove, params = _filtro(giunti, planes)
    vincolo_sport = ("AND s.sport IS NOT DISTINCT FROM rif.sport"
                     if same_sport_only else "")

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


# --- Confronti dentro la stessa sessione ---
# Quanto i cicli di una sessione si somigliano fra loro, e quanto si somigliano
# le coppie di giunti omologhi dei due lati.
def repeatability(cur, session_key, scope="lower_body", planes=("sagittal",)):
    giunti = resolve_scope(scope)
    if not giunti:
        return {"joints": [], "warnings": ["scope_sconosciuto"]}

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
    misurabile = any((r.get("n_coppie") or 0) > 0 for r in righe)
    avvisi = [] if misurabile else ["cicli_insufficienti"]
    return {"joints": righe, "warnings": avvisi, "scope": giunti,
            "planes": list(planes)}


_OMOLOGHI = (("hip_l", "hip_r"), ("knee_l", "knee_r"), ("ankle_l", "ankle_r"),
             ("shoulder_l", "shoulder_r"), ("elbow_l", "elbow_r"))


def symmetry(cur, session_key, planes=("sagittal",)):
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
