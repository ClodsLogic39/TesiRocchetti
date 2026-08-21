"""Caricamento nel database, dal C3D canonico, in una sola transazione.

    python -m src.database.ingest DatiElaborati/running_20260220_121922.c3d
    python -m src.database.ingest DatiElaborati --tutte
    python -m src.database.ingest --riferimenti

La sorgente e' il C3D canonico e nient'altro: angoli, quaternioni, cicli e
assenze arrivano gia' decisi dalla conversione, insieme alla loro provenienza, e
qui non vengono ricalcolati. Durante il caricamento vengono anche costruiti gli
atomi di forma dei cicli e valutati gli alert, confrontando le metriche della
sessione con i riferimenti di letteratura presenti in tabella.

Prerequisiti: psycopg, numpy ed ezc3d; un database con lo schema di
`schema.sql` gia' applicato; la DSN del ruolo di scrittura nella variabile
d'ambiente INGEST_DB_DSN.
"""
import argparse
import os
import re
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from src.alerts import references
from src.database import embeddings
from src.conversion import anatomy, canonical

# --- Connessione e conversioni di supporto ---
# DSN del ruolo di scrittura, unita' di misura dei canali di segmento e piccole
# funzioni che portano i valori letti dal file nella forma attesa dalle colonne.
DSN_DEFAULT = os.environ.get("INGEST_DB_DSN", "dbname=biomech")

_UNITA = {"quat": "1", "omega": "rad/s", "alpha": "rad/s^2"}


def _lista(v):
    a = np.asarray(v, dtype=float)
    return [None if not np.isfinite(x) else float(x) for x in a]


def _copertura(v):
    a = np.asarray(v, dtype=float)
    valid = np.all(np.isfinite(a), axis=1) if a.ndim == 2 else np.isfinite(a)
    return float(valid.mean()), int(valid.sum())


def _tipo_assenza(entita):
    testa = entita.split(":", 1)[0]
    return testa if testa in ("point", "segment", "angle") else "kinematics"


# --- Riferimenti di letteratura ---
# Caricamento dei valori di riferimento in tabella e loro rilettura, ristretta a
# quelli che si applicano al gesto della sessione.
def carica_riferimenti(cur):
    righe = references.righe()
    for r in righe:
        cur.execute(
            """
            INSERT INTO reference_ranges
                (scope, sport, joint, plane, metric, min_value, max_value,
                 unit, severity, source, source_url, note)
            VALUES (%(scope)s, %(sport)s, %(joint)s, %(plane)s, %(metric)s,
                    %(min_value)s, %(max_value)s, %(unit)s, %(severity)s,
                    %(source)s, %(source_url)s, %(note)s)
            ON CONFLICT (scope, sport, joint, plane, metric) DO UPDATE SET
                min_value = EXCLUDED.min_value, max_value = EXCLUDED.max_value,
                severity = EXCLUDED.severity, source = EXCLUDED.source,
                source_url = EXCLUDED.source_url, note = EXCLUDED.note
            """, r)
    return len(righe)


def _riferimenti_dal_db(cur, sport):
    sql = ("SELECT id, scope, sport, joint, plane, metric, min_value, max_value,"
           "       severity, source"
           "  FROM reference_ranges WHERE scope = 'anatomical'")
    params = []
    if sport:
        sql += " OR sport = %s"
        params.append(sport)
    cur.execute(sql, params)
    colonne = [d[0] for d in cur.description]
    return [dict(zip(colonne, riga)) for riga in cur.fetchall()]


# --- Chiave di sessione ---
# Forma ammessa per il nome con cui una sessione viene registrata e letta.
CHIAVE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def chiave_valida(session_key):
    return bool(session_key and CHIAVE_RE.match(session_key))


# --- Scrittura di una sessione ---
# Una funzione per ogni parte del file: intestazione della sessione, punti,
# segmenti, angoli, assenze dichiarate, cicli con i relativi atomi di forma e
# alert calcolati dal confronto con i riferimenti.
def _sessione(cur, dati, percorso, subject_id=None):
    meta = dati["meta"]
    if not chiave_valida(meta.get("source")):
        raise ValueError(
            f"nome di sessione non ammesso: {meta.get('source')!r}. "
            "Sono ammessi lettere, cifre, trattino e trattino basso (max 64). "
            "Il nome del file diventa la session_key ed e' visibile all'agente: "
            "rinomina la cattura con un nome non parlante.")
    durata = (meta["n_frames"] / meta["fps"]) if meta.get("fps") else None
    sport = anatomy.canonical_sport(meta.get("sport"))
    cur.execute(
        """
        INSERT INTO sessions (session_key, subject_id, sport, fps, n_frames,
                              duration_s, source_format, capture_system, view,
                              convention, biomech_version, canonical_path)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (session_key) DO UPDATE SET
            -- un ricaricamento senza soggetto non cancella quello gia' associato
            subject_id = COALESCE(EXCLUDED.subject_id, sessions.subject_id),
            sport = EXCLUDED.sport, fps = EXCLUDED.fps,
            n_frames = EXCLUDED.n_frames, duration_s = EXCLUDED.duration_s,
            source_format = EXCLUDED.source_format,
            capture_system = EXCLUDED.capture_system, view = EXCLUDED.view,
            convention = EXCLUDED.convention,
            biomech_version = EXCLUDED.biomech_version,
            canonical_path = EXCLUDED.canonical_path,
            ingested_at = now()
        RETURNING id
        """,
        (meta["source"], subject_id, sport, meta.get("fps"), meta["n_frames"],
         durata, meta.get("source_format"), meta.get("capture_system"),
         meta.get("view"), meta.get("convention"), meta.get("biomech_version"),
         os.path.abspath(percorso)))
    return cur.fetchone()[0]


def _punti(cur, sid, dati):
    for nome, v in dati["points"].items():
        cop, n = _copertura(v)
        cur.execute(
            """
            INSERT INTO point_trajectories
                (session_id, point, provenance, unit, x, y, z, coverage, n_valid)
            VALUES (%s,%s,%s,'m',%s,%s,%s,%s,%s)
            """,
            (sid, nome, dati["point_provenance"].get(nome, anatomy.DERIVED),
             _lista(v[:, 0]), _lista(v[:, 1]), _lista(v[:, 2]), cop, n))
    return len(dati["points"])


def _segmenti(cur, sid, dati):
    scritti = 0
    for nome, voce in dati["segments"].items():
        prov = voce.get("provenance", anatomy.DERIVED)
        for quantita, assi in (("quat", "xyzw"), ("omega", "xyz"), ("alpha", "xyz")):
            v = voce.get(quantita)
            if v is None:
                continue
            cop, _ = _copertura(v)
            for k, asse in enumerate(assi):
                cur.execute(
                    """
                    INSERT INTO segment_series
                        (session_id, segment, quantity, axis, provenance, unit,
                         values, coverage)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (sid, nome, quantita, asse, prov, _UNITA[quantita],
                     _lista(v[:, k]), cop))
                scritti += 1
    return scritti


def _angoli(cur, sid, dati):
    for (giunto, piano), v in dati["angles"].items():
        cop, n = _copertura(v)
        cur.execute(
            """
            INSERT INTO joint_angles
                (session_id, joint, plane, provenance, unit, values, coverage,
                 n_valid, implausible_rate)
            VALUES (%s,%s,%s,%s,'deg',%s,%s,%s,%s)
            """,
            (sid, giunto, piano,
             dati["angle_provenance"].get((giunto, piano), anatomy.DERIVED),
             _lista(v), cop, n, dati["implausible"].get((giunto, piano))))
    return len(dati["angles"])


def _assenze(cur, sid, dati):
    for entita, (motivo, dettaglio) in dati["absences"].items():
        cur.execute(
            """
            INSERT INTO absences (session_id, entity, kind, reason, detail)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT (session_id, entity) DO UPDATE SET
                reason = EXCLUDED.reason, detail = EXCLUDED.detail
            """, (sid, entita, _tipo_assenza(entita), motivo, dettaglio or None))
    return len(dati["absences"])


def _cicli(cur, sid, dati):
    esito = dati.get("cycles") or {}
    metodo = esito.get("method") or "ignoto"
    avvisi = list(esito.get("warnings") or [])
    lati = esito.get("sides") or []
    identificativi = []
    for i, (inizio, fine) in enumerate(esito.get("frames") or []):
        cur.execute(
            """
            INSERT INTO cycles (session_id, cycle_index, start_frame, end_frame,
                                method, side, warnings)
            VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id
            """, (sid, i, int(inizio), int(fine), metodo,
                  lati[i] if i < len(lati) else None, avvisi))
        identificativi.append((cur.fetchone()[0], int(inizio), int(fine)))

    for cid, inizio, fine in identificativi:
        for (giunto, piano), v in dati["angles"].items():
            tratto = np.asarray(v, dtype=float)[inizio:fine]
            for metrica, valore in references.metriche(tratto).items():
                cur.execute(
                    """
                    INSERT INTO cycle_metrics (cycle_id, joint, plane, metric, value)
                    VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT DO NOTHING
                    """, (cid, giunto, piano, metrica, valore))

    n_atomi = embeddings.insert_for_cycles(cur, dati, identificativi)
    return len(identificativi), n_atomi


def _alert(cur, sid, dati, riferimenti):
    cur.execute("DELETE FROM alerts WHERE session_id = %s", (sid,))

    osservate, coperture = {}, {}
    for chiave, v in dati["angles"].items():
        m = references.metriche(v)
        if m:
            osservate[chiave] = m
            coperture[chiave] = _copertura(v)[0]

    fuori = references.valuta(osservate, riferimenti, coperture,
                              dati["angle_provenance"])
    for a in fuori:
        cur.execute(
            """
            INSERT INTO alerts (session_id, reference_id, joint, plane, metric,
                                observed, expected_min, expected_max, deviation,
                                severity, coverage, provenance)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (sid, a["reference"]["id"], a["joint"], a["plane"], a["metric"],
             a["observed"], a["expected_min"], a["expected_max"],
             a["deviation"], a["severity"], a["coverage"], a["provenance"]))
    return fuori


# --- Caricamento completo ---
# Apre la transazione, ripulisce un eventuale caricamento precedente della stessa
# sessione, scrive tutte le parti e conferma solo se ogni passo e' riuscito.
def ingest(percorso, dsn=DSN_DEFAULT, conn=None, subject_id=None):
    import psycopg

    dati = canonical.read(percorso)
    avvisi = []
    proprio = conn is None
    conn = conn or psycopg.connect(dsn)
    try:
        with conn.cursor() as cur:
            sport_letto = dati["meta"].get("sport")
            sport = anatomy.canonical_sport(sport_letto)
            if sport_letto and not sport:
                avvisi.append(f"gesto_non_riconosciuto:{sport_letto}")
            sid = _sessione(cur, dati, percorso, subject_id)
            cur.execute("DELETE FROM point_trajectories WHERE session_id = %s", (sid,))
            cur.execute("DELETE FROM segment_series WHERE session_id = %s", (sid,))
            cur.execute("DELETE FROM joint_angles WHERE session_id = %s", (sid,))
            cur.execute("DELETE FROM cycles WHERE session_id = %s", (sid,))
            cur.execute("DELETE FROM absences WHERE session_id = %s", (sid,))

            n_punti = _punti(cur, sid, dati)
            n_serie = _segmenti(cur, sid, dati)
            n_angoli = _angoli(cur, sid, dati)
            n_assenze = _assenze(cur, sid, dati)
            n_cicli, n_atomi = _cicli(cur, sid, dati)
            riferimenti = _riferimenti_dal_db(cur, sport)
            fuori = _alert(cur, sid, dati, riferimenti)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if proprio:
            conn.close()

    gravita = {}
    for a in fuori:
        gravita[a["severity"]] = gravita.get(a["severity"], 0) + 1
    return {"sessione": dati["meta"]["source"], "id": sid, "punti": n_punti,
            "serie_segmento": n_serie, "angoli": n_angoli, "assenze": n_assenze,
            "cicli": n_cicli, "atomi": n_atomi, "alert": len(fuori),
            "per_gravita": gravita, "riferimenti_applicati": len(riferimenti),
            "sport": sport, "subject_id": subject_id, "avvisi": avvisi}


# --- Riga di comando ---
# Caricamento di un singolo file, di tutti i C3D di una cartella, o dei soli
# riferimenti di letteratura.
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sorgente", nargs="?", default="DatiElaborati",
                    help="file .c3d canonico o cartella che li contiene")
    ap.add_argument("--tutte", action="store_true")
    ap.add_argument("--riferimenti", action="store_true",
                    help="carica solo i valori di letteratura ed esce")
    ap.add_argument("--dsn", default=DSN_DEFAULT)
    ap.add_argument("--subject-id", help="associa la sessione a un soggetto "
                                         "dell'anagrafica (solo nel database)")
    args = ap.parse_args(argv)

    import psycopg
    if args.riferimenti:
        with psycopg.connect(args.dsn) as conn:
            with conn.cursor() as cur:
                n = carica_riferimenti(cur)
            conn.commit()
        print(f"{n} riferimenti di letteratura caricati")
        return 0

    percorsi = ([os.path.join(args.sorgente, f)
                 for f in sorted(os.listdir(args.sorgente)) if f.endswith(".c3d")]
                if args.tutte or os.path.isdir(args.sorgente) else [args.sorgente])
    if not percorsi:
        print(f"nessun C3D canonico in {args.sorgente}", file=sys.stderr)
        return 1

    with psycopg.connect(args.dsn) as conn:
        with conn.cursor() as cur:
            carica_riferimenti(cur)
        conn.commit()

        fatte, fallite = 0, []
        for p in percorsi:
            try:
                r = ingest(p, conn=conn, subject_id=args.subject_id)
            except Exception as e:
                fallite.append((os.path.basename(p), f"{type(e).__name__}: {e}"))
                continue
            fatte += 1
            print(f"  {r['sessione']}: {r['punti']} punti, {r['angoli']} angoli, "
                  f"{r['serie_segmento']} serie, {r['cicli']} cicli, "
                  f"{r['atomi']} atomi, {r['assenze']} assenze, "
                  f"{r['alert']} alert {r['per_gravita'] or ''}")
            for avviso in r["avvisi"]:
                print(f"      avviso: {avviso}", file=sys.stderr)
    if fallite:
        print(f"\n{len(fallite)} non caricate:", file=sys.stderr)
        for nome, motivo in fallite:
            print(f"  {nome}: {motivo}", file=sys.stderr)
    print(f"\n{fatte}/{len(percorsi)} sessioni caricate")
    return 0 if fatte else 1


if __name__ == "__main__":
    raise SystemExit(main())
