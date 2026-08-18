"""Anagrafica soggetti: operazioni sulla tabella protetta subjects_identity.

Richiede una DSN amministrativa (ADMIN_DSN). Nessuna cancellazione esposta: il
diritto all'oblio resta un'operazione deliberata via psql.
Uso da CLI:  python -m src.database.subjects list
"""
import argparse
import os
import re
import secrets

ADMIN_DSN = os.environ.get("ADMIN_DSN", "host=127.0.0.1 port=5432 dbname=biomech user=admin")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,32}$")

FIELDS = ("subject_id", "full_name", "birth_date", "height_m", "contact", "notes")


def _connect(dsn=None):
    import psycopg
    return psycopg.connect(dsn or ADMIN_DSN)


def suggest_id():
    """Codice opaco casuale: non invertibile, non enumerabile."""
    return "S-" + secrets.token_hex(4).upper()


def valid_id(subject_id):
    return bool(subject_id and ID_RE.match(subject_id))


def list_subjects(dsn=None, ids_only=False):
    cols = "subject_id" if ids_only else ", ".join(FIELDS)
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT {cols} FROM subjects_identity ORDER BY subject_id")
        rows = cur.fetchall()
    if ids_only:
        return [r[0] for r in rows]
    return [dict(zip(FIELDS, r)) for r in rows]


def add_subject(subject_id, full_name=None, birth_date=None, height_m=None,
                contact=None, notes=None, dsn=None):
    """Inserisce un soggetto. L'unicita' dell'id la impone la chiave primaria."""
    if not valid_id(subject_id):
        raise ValueError("id non valido: 3-32 caratteri fra A-Z a-z 0-9 _ - (niente spazi)")
    with _connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO subjects_identity "
                "(subject_id, full_name, birth_date, height_m, contact, notes) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (subject_id, full_name or None, birth_date or None,
                 float(height_m) if height_m else None, contact or None, notes or None),
            )
        conn.commit()
    return subject_id


def update_subject(subject_id, dsn=None, **fields):
    """Aggiorna i campi indicati di un soggetto esistente."""
    cols = [k for k in fields if k in FIELDS and k != "subject_id"]
    if not cols:
        return 0
    sets = ", ".join(f"{c} = %s" for c in cols)
    values = [fields[c] if fields[c] not in ("",) else None for c in cols]
    with _connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE subjects_identity SET {sets} WHERE subject_id = %s",
                        (*values, subject_id))
            n = cur.rowcount
        conn.commit()
    return n


def delete_subject(subject_id, anonymize_sessions=True, dsn=None):
    """Rimuove l'identita'. Con anonymize_sessions, azzera anche il pseudonimo
    nelle sessioni: i dati di movimento restano ma non sono piu' riconducibili."""
    with _connect(dsn) as conn:
        with conn.cursor() as cur:
            if anonymize_sessions:
                cur.execute("UPDATE sessions SET subject_id = NULL WHERE subject_id = %s",
                            (subject_id,))
            cur.execute("DELETE FROM subjects_identity WHERE subject_id = %s", (subject_id,))
            n = cur.rowcount
        conn.commit()
    return n


def main():
    ap = argparse.ArgumentParser(description="Anagrafica soggetti (richiede ADMIN_DSN)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("suggest")
    a = sub.add_parser("add")
    a.add_argument("subject_id")
    a.add_argument("--name")
    a.add_argument("--birth")
    a.add_argument("--height", type=float)
    args = ap.parse_args()

    if args.cmd == "list":
        for s in list_subjects():
            print(f"{s['subject_id']:12} {s['full_name'] or '-':24} h={s['height_m'] or '-'}")
    elif args.cmd == "suggest":
        print(suggest_id())
    else:
        print("inserito:", add_subject(args.subject_id, args.name, args.birth, args.height))


if __name__ == "__main__":
    main()
