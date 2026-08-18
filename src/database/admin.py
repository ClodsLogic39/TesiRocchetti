"""Operazioni amministrative sulle sessioni (elenco e cancellazione).

Richiede una DSN amministrativa: i ruoli MCP e ingest non hanno DELETE.
La cancellazione di una sessione propaga in cascata a serie, cicli, embedding e
alert (ON DELETE CASCADE nello schema).
"""
import os

from .subjects import _connect


def list_sessions(dsn=None):
    """Sessioni con conteggi derivati, per la pagina di amministrazione."""
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT s.session_key, s.subject_id, s.sport, s.n_frames, s.ingested_at,"
            "  (SELECT count(*) FROM joint_angles a WHERE a.session_id = s.id),"
            "  (SELECT count(*) FROM cycles c WHERE c.session_id = s.id)"
            " FROM sessions s ORDER BY s.ingested_at DESC"
        )
        cols = ("session_key", "subject_id", "sport", "n_frames", "ingested_at",
                "n_signals", "n_cycles")
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def delete_session(session_key, dsn=None):
    """Elimina la sessione e tutto il derivato. Restituisce le righe rimosse."""
    with _connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE session_key = %s", (session_key,))
            n = cur.rowcount
        conn.commit()
    return n


def delete_sessions_of(subject_id, dsn=None):
    """Elimina tutte le sessioni di un soggetto. Restituisce le chiavi rimosse."""
    with _connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT session_key FROM sessions WHERE subject_id = %s", (subject_id,))
            keys = [r[0] for r in cur.fetchall()]
            cur.execute("DELETE FROM sessions WHERE subject_id = %s", (subject_id,))
        conn.commit()
    return keys


def remove_processed_dir(session_key, output_root):
    """Rimuove la cartella elaborata sul disco, se dentro output_root."""
    import shutil
    path = os.path.realpath(os.path.join(output_root, f"{session_key}_processed"))
    if not path.startswith(os.path.realpath(output_root) + os.sep) or not os.path.isdir(path):
        return False
    shutil.rmtree(path)
    return True
