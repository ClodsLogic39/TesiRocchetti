-- Schema del database biomeccanico.
-- Una "sessione" raccoglie i dati di una singola acquisizione motion-capture;
-- le traiettorie 3D dei punti e i canali analogici (angoli, velocita'/accel.
-- angolari, quaternioni) sono memorizzati come array REAL[] paralleli ai frame.
-- I gap nei dati restano NULL nell'array (non vengono azzerati).

BEGIN;

-- Anagrafica delle sessioni. metadata_json contiene il blocco "metadata" del
-- JSON canonico (include il campo top-level "sport", usato dalle query).
CREATE TABLE IF NOT EXISTS sessions (
  id SERIAL PRIMARY KEY,
  session_key TEXT UNIQUE NOT NULL,   -- identificativo univoco della sessione
  fps REAL,
  n_frames INT,
  duration_s REAL,
  source_path TEXT,                   -- percorso del file/cartella di origine
  created_at TIMESTAMPTZ DEFAULT now(),
  metadata_json JSONB
);

-- Traiettorie 3D dei punti (interop/rendering). Una riga per marker/label;
-- x/y/z sono array paralleli ai frame, valid_mask marca i frame validi.
CREATE TABLE IF NOT EXISTS c3d_points (
  id SERIAL PRIMARY KEY,
  session_id INT REFERENCES sessions(id) ON DELETE CASCADE,
  label TEXT NOT NULL,
  x_values REAL[] NOT NULL,
  y_values REAL[] NOT NULL,
  z_values REAL[] NOT NULL,
  valid_mask BOOLEAN[]
);

-- Cinematica derivata (sorgente delle query MCP): una riga per canale
-- (es. hip_sagittal, femur_left_omega_x, torso_quat_w), array parallelo ai frame.
CREATE TABLE IF NOT EXISTS c3d_analogs (
  id SERIAL PRIMARY KEY,
  session_id INT REFERENCES sessions(id) ON DELETE CASCADE,
  label TEXT NOT NULL,
  values REAL[] NOT NULL
);

-- Riferimenti ai file prodotti per la sessione (c3d, json, csv, ...).
CREATE TABLE IF NOT EXISTS session_files (
  id SERIAL PRIMARY KEY,
  session_id INT REFERENCES sessions(id) ON DELETE CASCADE,
  file_type TEXT NOT NULL,
  file_path TEXT NOT NULL
);

-- Indici per le ricerche piu' frequenti (per sessione + label).
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions (created_at);
CREATE INDEX IF NOT EXISTS idx_c3d_points_session_label ON c3d_points (session_id, label);
CREATE INDEX IF NOT EXISTS idx_c3d_analogs_session_label ON c3d_analogs (session_id, label);
CREATE INDEX IF NOT EXISTS idx_session_files_session_type ON session_files (session_id, file_type);

COMMIT;
