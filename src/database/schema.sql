-- Schema del database biomeccanico.
--
-- Rispecchia il contenuto del C3D canonico: i punti anatomici, le terne dei
-- segmenti, gli angoli sui piani, i cicli con la forma delle curve, cio' che
-- manca insieme al motivo, e i riferimenti di letteratura con gli alert che ne
-- derivano. Il vocabolario anatomico e' vincolato da chiavi esterne verso
-- tabelle di lookup, i cui valori ammessi sono inseriti qui.
--
-- Prerequisiti: PostgreSQL con l'estensione pgvector disponibile e un utente che
-- possa creare estensioni e tabelle. Il file si esegue per intero, in una sola
-- transazione:
--
--   psql -d biomech -f src/database/schema.sql

BEGIN;

-- Estensione su cui si appoggiano i confronti fra andamenti.
CREATE EXTENSION IF NOT EXISTS vector;

-- ===========================================================================
-- 1. Vocabolario
-- Tabelle di lookup con i valori ammessi: provenienze, motivi di assenza, piani,
-- gesti, punti anatomici, segmenti e giunti. Tutto il resto vi si aggancia con
-- chiavi esterne.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS provenance_kinds (
    provenance TEXT PRIMARY KEY,
    description TEXT NOT NULL
);
INSERT INTO provenance_kinds (provenance, description) VALUES
    ('measured', 'osservato direttamente dalla sorgente'),
    ('derived',  'calcolato in modo deterministico da grandezze misurate'),
    ('modelled', 'dipende da un''assunzione di modello dichiarata'),
    ('filled',   'interpolato per colmare un buco breve')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS absence_reasons (
    reason TEXT PRIMARY KEY,
    description TEXT NOT NULL
);
INSERT INTO absence_reasons (reason, description) VALUES
    ('not_in_source',  'la sorgente non contiene questa entita'''),
    ('low_coverage',   'presente ma vista in troppo pochi frame'),
    ('missing_inputs', 'mancano i punti da cui si costruisce'),
    ('no_time_base',   'senza frequenza non esistono derivate temporali'),
    ('not_observable', 'il modello di segmento non osserva quel piano')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS planes (
    plane TEXT PRIMARY KEY
);
INSERT INTO planes (plane) VALUES ('sagittal'), ('frontal'), ('transverse')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS sports (
    sport TEXT PRIMARY KEY
);
INSERT INTO sports (sport) VALUES ('walking'), ('running'), ('cycling'), ('swimming')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS points (
    point TEXT PRIMARY KEY,
    side  TEXT CHECK (side IN ('left', 'right', 'central'))
);
INSERT INTO points (point, side) VALUES
    ('head', 'central'), ('shoulder_c', 'central'),
    ('shoulder_l', 'left'), ('shoulder_r', 'right'),
    ('elbow_l', 'left'), ('elbow_r', 'right'),
    ('wrist_l', 'left'), ('wrist_r', 'right'),
    ('hip_l', 'left'), ('hip_r', 'right'),
    ('knee_l', 'left'), ('knee_r', 'right'),
    ('ankle_l', 'left'), ('ankle_r', 'right'),
    ('heel_l', 'left'), ('heel_r', 'right'),
    ('toe_l', 'left'), ('toe_r', 'right')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS segments (
    segment TEXT PRIMARY KEY,
    side    TEXT CHECK (side IN ('left', 'right', 'central')),
    region  TEXT CHECK (region IN ('trunk', 'upper_limb', 'lower_limb'))
);
INSERT INTO segments (segment, side, region) VALUES
    ('pelvis', 'central', 'trunk'), ('trunk', 'central', 'trunk'),
    ('neck', 'central', 'trunk'),
    ('clavicle_l', 'left', 'trunk'), ('clavicle_r', 'right', 'trunk'),
    ('humerus_l', 'left', 'upper_limb'), ('humerus_r', 'right', 'upper_limb'),
    ('forearm_l', 'left', 'upper_limb'), ('forearm_r', 'right', 'upper_limb'),
    ('femur_l', 'left', 'lower_limb'), ('femur_r', 'right', 'lower_limb'),
    ('shank_l', 'left', 'lower_limb'), ('shank_r', 'right', 'lower_limb'),
    ('foot_l', 'left', 'lower_limb'), ('foot_r', 'right', 'lower_limb')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS joints (
    joint  TEXT PRIMARY KEY,
    side   TEXT CHECK (side IN ('left', 'right', 'central')),
    region TEXT CHECK (region IN ('trunk', 'upper_limb', 'lower_limb')),
    proximal_segment TEXT,
    distal_segment   TEXT REFERENCES segments(segment)
);
INSERT INTO joints (joint, side, region, proximal_segment, distal_segment) VALUES
    ('trunk', 'central', 'trunk', 'world', 'trunk'),
    ('spine', 'central', 'trunk', 'pelvis', 'trunk'),
    ('neck', 'central', 'trunk', 'clavicle_r', 'neck'),
    ('shoulder_l', 'left', 'upper_limb', 'clavicle_l', 'humerus_l'),
    ('shoulder_r', 'right', 'upper_limb', 'clavicle_r', 'humerus_r'),
    ('elbow_l', 'left', 'upper_limb', 'humerus_l', 'forearm_l'),
    ('elbow_r', 'right', 'upper_limb', 'humerus_r', 'forearm_r'),
    ('hip_l', 'left', 'lower_limb', 'pelvis', 'femur_l'),
    ('hip_r', 'right', 'lower_limb', 'pelvis', 'femur_r'),
    ('knee_l', 'left', 'lower_limb', 'femur_l', 'shank_l'),
    ('knee_r', 'right', 'lower_limb', 'femur_r', 'shank_r'),
    ('ankle_l', 'left', 'lower_limb', 'shank_l', 'foot_l'),
    ('ankle_r', 'right', 'lower_limb', 'shank_r', 'foot_r')
ON CONFLICT DO NOTHING;

-- ===========================================================================
-- 2. Identita' dei soggetti
-- Tabella tenuta separata dai dati di movimento, che vi si riferiscono solo
-- tramite un identificativo.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS subjects_identity (
    subject_id   TEXT PRIMARY KEY,
    full_name    TEXT,
    birth_date   DATE,
    height_m     REAL,
    contact      TEXT,
    notes        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ===========================================================================
-- 3. Sessioni
-- Una riga per acquisizione, con il gesto, la base temporale, il formato di
-- origine e il riferimento al soggetto.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS sessions (
    id              SERIAL PRIMARY KEY,
    session_key     TEXT UNIQUE NOT NULL,
    subject_id      TEXT,
    sport           TEXT REFERENCES sports(sport),
    fps             REAL,
    n_frames        INTEGER NOT NULL,
    duration_s      REAL,
    source_format   TEXT,
    capture_system  TEXT,
    view            TEXT,
    convention      TEXT NOT NULL,
    biomech_version TEXT NOT NULL,
    canonical_path  TEXT,
    body_mass_kg    REAL,
    height_m        REAL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS sessions_sport_idx ON sessions (sport);

-- ===========================================================================
-- 4. Dati di movimento
-- Traiettorie dei punti, serie dei segmenti, angoli articolari e assenze
-- dichiarate, ciascuno con la propria provenienza e copertura.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS point_trajectories (
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    point       TEXT NOT NULL REFERENCES points(point),
    provenance  TEXT NOT NULL REFERENCES provenance_kinds(provenance),
    unit        TEXT NOT NULL DEFAULT 'm',
    x           REAL[] NOT NULL,
    y           REAL[] NOT NULL,
    z           REAL[] NOT NULL,
    coverage    REAL NOT NULL,
    n_valid     INTEGER NOT NULL,
    PRIMARY KEY (session_id, point)
);

CREATE TABLE IF NOT EXISTS segment_series (
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    segment     TEXT NOT NULL REFERENCES segments(segment),
    quantity    TEXT NOT NULL CHECK (quantity IN ('quat', 'omega', 'alpha')),
    axis        TEXT NOT NULL CHECK (axis IN ('x', 'y', 'z', 'w')),
    provenance  TEXT NOT NULL REFERENCES provenance_kinds(provenance),
    unit        TEXT NOT NULL,
    values      REAL[] NOT NULL,
    coverage    REAL NOT NULL,
    PRIMARY KEY (session_id, segment, quantity, axis)
);

CREATE TABLE IF NOT EXISTS joint_angles (
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    joint       TEXT NOT NULL REFERENCES joints(joint),
    plane       TEXT NOT NULL REFERENCES planes(plane),
    provenance  TEXT NOT NULL REFERENCES provenance_kinds(provenance),
    unit        TEXT NOT NULL DEFAULT 'deg',
    values      REAL[] NOT NULL,
    coverage    REAL NOT NULL,
    n_valid     INTEGER NOT NULL,
    implausible_rate REAL,
    PRIMARY KEY (session_id, joint, plane)
);

CREATE TABLE IF NOT EXISTS absences (
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    entity      TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('point', 'segment', 'angle', 'kinematics')),
    reason      TEXT NOT NULL REFERENCES absence_reasons(reason),
    detail      TEXT,
    PRIMARY KEY (session_id, entity)
);
CREATE INDEX IF NOT EXISTS absences_reason_idx ON absences (reason);

-- ===========================================================================
-- 5. Cicli
-- Intervalli individuati dalla segmentazione, con il metodo e il lato, e le
-- metriche calcolate su ciascun ciclo.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS cycles (
    id          SERIAL PRIMARY KEY,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    cycle_index INTEGER NOT NULL,
    start_frame INTEGER NOT NULL,
    end_frame   INTEGER NOT NULL,
    method      TEXT NOT NULL,
    side        TEXT,
    warnings    TEXT[] NOT NULL DEFAULT '{}',
    UNIQUE (session_id, cycle_index)
);

CREATE TABLE IF NOT EXISTS cycle_metrics (
    cycle_id    INTEGER NOT NULL REFERENCES cycles(id) ON DELETE CASCADE,
    joint       TEXT NOT NULL REFERENCES joints(joint),
    plane       TEXT NOT NULL REFERENCES planes(plane),
    metric      TEXT NOT NULL,
    value       REAL NOT NULL,
    unit        TEXT NOT NULL DEFAULT 'deg',
    PRIMARY KEY (cycle_id, joint, plane, metric)
);

-- ===========================================================================
-- 6. Atomi di ciclo
-- Un vettore per ogni terna ciclo/giunto/piano, che descrive la forma della
-- curva, con gli indici per la ricerca per somiglianza.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS cycle_atoms (
    cycle_id        INTEGER NOT NULL REFERENCES cycles(id) ON DELETE CASCADE,
    joint           TEXT NOT NULL REFERENCES joints(joint),
    plane           TEXT NOT NULL REFERENCES planes(plane),
    encoder_version TEXT NOT NULL,
    shape           vector(32) NOT NULL,
    rom             REAL NOT NULL,
    peak            REAL NOT NULL,
    trough          REAL NOT NULL,
    provenance      TEXT NOT NULL REFERENCES provenance_kinds(provenance),
    coverage        REAL NOT NULL,
    PRIMARY KEY (cycle_id, joint, plane, encoder_version)
);
CREATE INDEX IF NOT EXISTS cycle_atoms_shape_idx
    ON cycle_atoms USING hnsw (shape vector_cosine_ops);
CREATE INDEX IF NOT EXISTS cycle_atoms_lookup_idx
    ON cycle_atoms (joint, plane, encoder_version);

-- ===========================================================================
-- 7. Riferimenti per gli alert
-- Intervalli di letteratura, anatomici o legati a un gesto, e gli alert prodotti
-- dal confronto con le metriche di una sessione.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS reference_ranges (
    id          SERIAL PRIMARY KEY,
    scope       TEXT NOT NULL CHECK (scope IN ('anatomical', 'sport')),
    sport       TEXT,
    joint       TEXT NOT NULL REFERENCES joints(joint),
    plane       TEXT NOT NULL REFERENCES planes(plane),
    metric      TEXT NOT NULL CHECK (metric IN ('rom', 'peak_max', 'peak_min')),
    min_value   REAL,
    max_value   REAL,
    unit        TEXT NOT NULL DEFAULT 'deg',
    severity    TEXT NOT NULL DEFAULT 'warning'
                CHECK (severity IN ('info', 'warning', 'critical')),
    source      TEXT NOT NULL,
    source_url  TEXT,
    note        TEXT,
    UNIQUE NULLS NOT DISTINCT (scope, sport, joint, plane, metric)
);
CREATE INDEX IF NOT EXISTS reference_ranges_lookup_idx
    ON reference_ranges (joint, plane, metric);

CREATE TABLE IF NOT EXISTS alerts (
    id           SERIAL PRIMARY KEY,
    session_id   INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    cycle_id     INTEGER REFERENCES cycles(id) ON DELETE CASCADE,
    reference_id INTEGER NOT NULL REFERENCES reference_ranges(id),
    joint        TEXT NOT NULL REFERENCES joints(joint),
    plane        TEXT NOT NULL REFERENCES planes(plane),
    metric       TEXT NOT NULL,
    observed     REAL NOT NULL,
    expected_min REAL,
    expected_max REAL,
    deviation    REAL NOT NULL,
    severity     TEXT NOT NULL,
    coverage     REAL,
    provenance   TEXT REFERENCES provenance_kinds(provenance),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS alerts_session_idx ON alerts (session_id);

-- ===========================================================================
-- 8. Viste di comodo
-- Viste che riuniscono, per una sessione, i segnali disponibili, le assenze, gli
-- atomi di forma e gli alert.
-- ===========================================================================

CREATE OR REPLACE VIEW session_signals AS
SELECT s.session_key, s.sport, s.fps, s.n_frames, s.capture_system,
       a.joint, a.plane, a.provenance, a.coverage, a.n_valid,
       a.implausible_rate, a.values
  FROM joint_angles a
  JOIN sessions s ON s.id = a.session_id;

CREATE OR REPLACE VIEW session_absences AS
SELECT s.session_key, ab.entity, ab.kind, ab.reason, r.description, ab.detail
  FROM absences ab
  JOIN sessions s ON s.id = ab.session_id
  JOIN absence_reasons r ON r.reason = ab.reason;

CREATE OR REPLACE VIEW session_atoms AS
SELECT s.id AS session_id, s.session_key, s.sport, s.capture_system,
       c.id AS cycle_id, c.cycle_index, c.method AS cycle_method,
       a.joint, a.plane, a.encoder_version, a.shape, a.rom, a.peak, a.trough,
       a.provenance, a.coverage
  FROM cycle_atoms a
  JOIN cycles c ON c.id = a.cycle_id
  JOIN sessions s ON s.id = c.session_id;

CREATE OR REPLACE VIEW session_alerts AS
SELECT s.session_key, s.sport, al.joint, al.plane, al.metric, al.observed,
       al.expected_min, al.expected_max, al.deviation, al.severity,
       al.coverage, al.provenance, rr.scope, rr.source, rr.note
  FROM alerts al
  JOIN sessions s ON s.id = al.session_id
  JOIN reference_ranges rr ON rr.id = al.reference_id;

COMMIT;
