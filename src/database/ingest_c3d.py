#!/usr/bin/env python3
"""
Ingestione delle sessioni elaborate in PostgreSQL.

La sorgente canonica e' il file per-sessione *_generic.json (scelta: il JSON e'
canonico per la cinematica derivata). Angoli articolari, velocita'/accelerazione
angolare dei segmenti e quaternioni vengono letti dal JSON e salvati in
c3d_analogs con lo stesso schema di label che si aspetta il layer MCP. Il file
gemello *_generic.c3d, se presente, viene usato solo per le traiettorie 3D dei
punti/marker (interoperabilita' + rendering) -> c3d_points.

I metadati salvati in sessions.metadata_json sono il blocco "metadata" del JSON
(compatto, ed espone "sport" al livello superiore cosi' compare_sports funziona).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import psycopg
from psycopg.types.json import Json

try:
    import ezc3d
except ImportError:
    ezc3d = None


def normalize_label(raw):
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = str(raw)
    return text.replace("\x00", "").strip()


def labels_or_default(raw_labels, count, prefix):
    labels = [normalize_label(v) for v in raw_labels] if raw_labels else []
    if len(labels) < count:
        labels.extend(f"{prefix}_{i}" for i in range(len(labels), count))
    return labels[:count]


def derive_session_key(stem):
    """Ricava la chiave di sessione dallo stem del file, togliendo il suffisso "_generic"."""
    return stem[:-8] if stem.endswith("_generic") else stem


def sanitize_json_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return {k: sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize_json_value(value.tolist())
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def to_float_or_none(value):
    """Converte null/valori non finiti in None: i gap restano NULL espliciti nei REAL[]."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def column(rows, idx):
    """Estrae un asse da una lista di righe [x, y, z(, w)], tollerando righe None."""
    out = []
    for row in rows:
        if row is None or not isinstance(row, (list, tuple)) or len(row) <= idx:
            out.append(None)
        else:
            out.append(to_float_or_none(row[idx]))
    return out


# ---------------------------------------------------------------------------
# Punti dal C3D (solo interoperabilita')
# ---------------------------------------------------------------------------

def extract_points(c3d):
    points = np.asarray(c3d["data"]["points"], dtype=float)
    if points.ndim != 3 or points.shape[0] < 3:
        raise ValueError(f"Unexpected points shape: {points.shape}")
    residual = points[3, :, :] if points.shape[0] >= 4 else None
    xyz = points[:3, :, :]
    return xyz, residual, points.shape[1], points.shape[2]


def ingest_points_from_c3d(cur, session_id, c3d_path):
    if ezc3d is None:
        return
    c3d = ezc3d.c3d(str(c3d_path))
    xyz, residual, n_points, _ = extract_points(c3d)
    raw_labels = c3d["parameters"].get("POINT", {}).get("LABELS", {}).get("value", [])
    labels = labels_or_default(raw_labels, n_points, "point")
    for i, label in enumerate(labels):
        valid_mask = (residual[i, :] >= 0).tolist() if residual is not None else None
        cur.execute(
            """
            INSERT INTO c3d_points (session_id, label, x_values, y_values, z_values, valid_mask)
            VALUES (%s,%s,%s,%s,%s,%s)
            """,
            (session_id, label, xyz[0, i, :].tolist(), xyz[1, i, :].tolist(),
             xyz[2, i, :].tolist(), valid_mask),
        )


# ---------------------------------------------------------------------------
# Canali analogici dal JSON (canonico: angoli, omega, alpha, quaternioni)
# ---------------------------------------------------------------------------

def insert_analog(cur, session_id, label, values):
    cur.execute(
        "INSERT INTO c3d_analogs (session_id, label, values) VALUES (%s,%s,%s)",
        (session_id, label, values),
    )


def ingest_analogs_from_json(cur, session_id, data):
    # Angoli articolari: f"{joint}_{axis}"  (schema di label risolto dal registry MCP)
    for joint, axes in (data.get("joints") or {}).items():
        for axis, values in (axes or {}).items():
            insert_analog(cur, session_id, f"{joint}_{axis}",
                          [to_float_or_none(v) for v in (values or [])])

    # Velocita' / accelerazione angolare / quaternioni per ogni segmento
    for seg, sd in (data.get("segments") or {}).items():
        omega = sd.get("angular_velocity") or []
        alpha = sd.get("angular_acceleration") or []
        quats = sd.get("quaternions") or []
        if omega:
            for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
                insert_analog(cur, session_id, f"{seg}_omega_{axis}", column(omega, idx))
        if alpha:
            for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
                insert_analog(cur, session_id, f"{seg}_alpha_{axis}", column(alpha, idx))
        if quats:
            for axis, idx in (("x", 0), ("y", 1), ("z", 2), ("w", 3)):
                insert_analog(cur, session_id, f"{seg}_quat_{axis}", column(quats, idx))


# ---------------------------------------------------------------------------
# Orchestrazione per sessione
# ---------------------------------------------------------------------------

def ingest_session(cur, json_path):
    """Ingerisce una sessione: upsert anagrafica, ripulisce i dati vecchi, reinserisce punti e analogici."""
    data = json.loads(json_path.read_text())
    meta = data.get("metadata", {}) or {}

    session_key = derive_session_key(json_path.stem)
    fps = to_float_or_none(meta.get("fps"))
    n_frames = meta.get("n_frames")
    duration_s = to_float_or_none(meta.get("duration_seconds"))
    if duration_s is None and fps and n_frames:
        duration_s = n_frames / fps

    c3d_path = json_path.with_suffix(".c3d")
    metadata_value = Json(sanitize_json_value(meta))

    cur.execute(
        """
        INSERT INTO sessions (session_key, fps, n_frames, duration_s, source_path, metadata_json)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (session_key) DO UPDATE
          SET fps=EXCLUDED.fps,
              n_frames=EXCLUDED.n_frames,
              duration_s=EXCLUDED.duration_s,
              source_path=EXCLUDED.source_path,
              metadata_json=EXCLUDED.metadata_json
        RETURNING id
        """,
        (session_key, fps, n_frames, duration_s, str(json_path), metadata_value),
    )
    session_id = cur.fetchone()[0]

    # Re-ingestione idempotente: rimuove i dati precedenti della sessione prima di reinserirli.
    cur.execute("DELETE FROM c3d_points WHERE session_id=%s", (session_id,))
    cur.execute("DELETE FROM c3d_analogs WHERE session_id=%s", (session_id,))
    cur.execute("DELETE FROM session_files WHERE session_id=%s", (session_id,))

    cur.execute(
        "INSERT INTO session_files (session_id, file_type, file_path) VALUES (%s,'generic_json',%s)",
        (session_id, str(json_path)),
    )
    if c3d_path.exists():
        cur.execute(
            "INSERT INTO session_files (session_id, file_type, file_path) VALUES (%s,'c3d',%s)",
            (session_id, str(c3d_path)),
        )
        ingest_points_from_c3d(cur, session_id, c3d_path)

    ingest_analogs_from_json(cur, session_id, data)
    return session_key


def main():
    parser = argparse.ArgumentParser(description="Ingest *_processed sessions into PostgreSQL (JSON canonical).")
    parser.add_argument("--db", default="dbname=biomech",
                        help="PostgreSQL connection string (default: dbname=biomech)")
    parser.add_argument("--root", default=None,
                        help="Data directory to scan for *_processed/*_generic.json "
                             "(default: <repo-root>/data/processed).")
    args = parser.parse_args()

    # Questo script vive in src/database/, quindi la radice del repo e' due livelli sopra.
    repo_root = Path(__file__).resolve().parents[2]
    data_root = Path(args.root).resolve() if args.root else repo_root / "data" / "processed"
    if not data_root.exists():
        raise SystemExit(f"Data directory not found: {data_root}")

    json_files = sorted(data_root.glob("**/*_processed/*_generic.json"))
    if not json_files:
        raise SystemExit(f"No *_generic.json files found under: {data_root}")

    conn = psycopg.connect(args.db)
    conn.autocommit = True

    ingested = []
    with conn.cursor() as cur:
        for json_path in json_files:
            ingested.append(ingest_session(cur, json_path))

    conn.close()
    print(f"Ingested {len(ingested)} sessions: {', '.join(ingested)}")


if __name__ == "__main__":
    main()
