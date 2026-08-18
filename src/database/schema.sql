-- Schema del database biomeccanico.
--
-- Rispecchia uno a uno cio' che il C3D canonico contiene: sedici punti, le terne
-- dei segmenti, gli angoli sui piani osservabili, e cio' che manca con il motivo.
-- Il vocabolario anatomico e' vincolato da chiavi esterne, non da convenzioni di
-- stringhe: un giunto inesistente non entra nel database.
--
--   psql -d biomech -f src/database/schema.sql

BEGIN;

-- I confronti fra andamenti si fanno con pgvector: e' il motivo per cui il
-- progetto usa PostgreSQL e non un archivio qualunque.
CREATE EXTENSION IF NOT EXISTS vector;

-- ===========================================================================
-- 1. Vocabolario
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

-- I gesti riconosciuti. Il gesto decide come si segmenta in cicli e quali
-- riferimenti di letteratura si applicano: un nome fuori vocabolario passerebbe
-- i controlli anatomici e nessuno di quelli del gesto, e la sessione
-- risulterebbe senza alert invece che non giudicata. La chiave esterna lo rende
-- impossibile.
CREATE TABLE IF NOT EXISTS sports (
    sport TEXT PRIMARY KEY
);
INSERT INTO sports (sport) VALUES ('walking'), ('running'), ('cycling'), ('swimming')
ON CONFLICT DO NOTHING;

-- I sedici punti anatomici. Nessuno in piu': sono i centri articolari che ogni
-- sistema di cattura sa fornire, e da essi discende tutto il resto.
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

-- I segmenti che i punti danno per costruzione.
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
    -- tronco rispetto al laboratorio. Prima che il tronco avesse una terna
    -- propria questo giunto usava il bacino come surrogato (distal=pelvis),
    -- ora misura il segmento che il nome promette.
    ('trunk', 'central', 'trunk', 'world', 'trunk'),
    -- tronco rispetto al bacino: isola il movimento della colonna da quello
    -- del bacino sotto di essa (la grandezza "Spine Angles" dei modelli
    -- clinici Plug-in Gait).
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
-- 2. Identita' dei soggetti, separata dai dati di movimento
-- ===========================================================================
-- Nessun ruolo applicativo ha accesso a questa tabella: il collegamento con le
-- sessioni passa solo per subject_id, che di per se' non identifica nessuno.

CREATE TABLE IF NOT EXISTS subjects_identity (
    subject_id   TEXT PRIMARY KEY,   -- codice opaco: non parla del soggetto
    full_name    TEXT,
    birth_date   DATE,
    height_m     REAL,               -- statura di riferimento dell'anagrafica;
                                     -- quella della singola acquisizione sta in
                                     -- sessions, perche' cambia nel tempo
    contact      TEXT,
    notes        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ===========================================================================
-- 3. Sessioni
-- ===========================================================================

CREATE TABLE IF NOT EXISTS sessions (
    id              SERIAL PRIMARY KEY,
    session_key     TEXT UNIQUE NOT NULL,
    subject_id      TEXT,
    sport           TEXT REFERENCES sports(sport),
    fps             REAL,
    n_frames        INTEGER NOT NULL,
    duration_s      REAL,
    -- da dove viene e come e' stato prodotto
    source_format   TEXT,          -- mediapipe_json | c3d
    capture_system  TEXT,          -- monocular_pose_estimation | marker_based
    view            TEXT,          -- frontal | sagittal | ...
    convention      TEXT NOT NULL, -- convenzione di terne e angoli dichiarata
    biomech_version TEXT NOT NULL, -- versione del contratto del C3D canonico
    canonical_path  TEXT,          -- il file da cui questa riga e' nata
    body_mass_kg    REAL,
    height_m        REAL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS sessions_sport_idx ON sessions (sport);

-- ===========================================================================
-- 4. Dati di movimento
-- ===========================================================================
-- Le serie stanno in array REAL[]: un campione per frame, NULL dove il dato
-- manca. Un NULL non e' mai uno zero.

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
    unit        TEXT NOT NULL,          -- '1' | 'rad/s' | 'rad/s^2'
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
    -- quota di variazioni oltre il limite fisiologico: dice quanto e' gesto e
    -- quanto e' rumore, senza dover rileggere la serie
    implausible_rate REAL,
    PRIMARY KEY (session_id, joint, plane)
);

-- Cio' che manca, e perche'. E' un dato quanto gli altri: permette all'agente di
-- rispondere "non posso dartelo perche'" invece di tacere.
CREATE TABLE IF NOT EXISTS absences (
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    entity      TEXT NOT NULL,     -- 'point:knee_l' | 'angle:hip_r:transverse' | ...
    kind        TEXT NOT NULL CHECK (kind IN ('point', 'segment', 'angle', 'kinematics')),
    reason      TEXT NOT NULL REFERENCES absence_reasons(reason),
    detail      TEXT,
    PRIMARY KEY (session_id, entity)
);
CREATE INDEX IF NOT EXISTS absences_reason_idx ON absences (reason);

-- ===========================================================================
-- 5. Cicli
-- ===========================================================================

CREATE TABLE IF NOT EXISTS cycles (
    id          SERIAL PRIMARY KEY,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    cycle_index INTEGER NOT NULL,
    -- Intervallo [start_frame, end_frame): fine esclusiva, la stessa
    -- convenzione con cui il server MCP affetta le serie. Il picco che chiude
    -- un ciclo apre il successivo, e non appartiene a entrambi.
    start_frame INTEGER NOT NULL,
    end_frame   INTEGER NOT NULL,
    method      TEXT NOT NULL,     -- come e' stato individuato
    side        TEXT,
    -- Cio' che la segmentazione ha dovuto concedere, come lo scrive il C3D.
    -- Non e' deducibile dal metodo: `primary_signal_low_coverage` dipende da
    -- quanto era visibile il segnale migliore, non da quale strategia ha vinto.
    -- Senza questa colonna l'avviso restava nel file e non raggiungeva chi
    -- interroga il database.
    warnings    TEXT[] NOT NULL DEFAULT '{}',
    UNIQUE (session_id, cycle_index)
);

CREATE TABLE IF NOT EXISTS cycle_metrics (
    cycle_id    INTEGER NOT NULL REFERENCES cycles(id) ON DELETE CASCADE,
    joint       TEXT NOT NULL REFERENCES joints(joint),
    plane       TEXT NOT NULL REFERENCES planes(plane),
    metric      TEXT NOT NULL,     -- rom | peak_max | peak_min | mean
    value       REAL NOT NULL,
    unit        TEXT NOT NULL DEFAULT 'deg',
    PRIMARY KEY (cycle_id, joint, plane, metric)
);

-- ===========================================================================
-- 6. Atomi di ciclo: la forma delle curve, per i confronti
-- ===========================================================================
-- Un atomo = (ciclo, giunto, piano). Il vettore descrive la sola FORMA della
-- curva: media rimossa e norma unitaria.
--
-- Perche' solo la forma. Gli angoli che produciamo portano scostamenti propri
-- del soggetto -- la caviglia non ha una prova statica, il bacino nasce dai
-- centri articolari e non dalle spine iliache -- quindi confrontare i valori
-- assoluti fra persone diverse misurerebbe la differenza fra i due modelli, non
-- fra i due gesti. L'andamento nel tempo invece e' confrontabile. Ampiezza e
-- livello viaggiano a parte, come numeri, dove si confrontano con operatori
-- ordinari e solo quando ha senso farlo.
--
-- Su vettori a media nulla e norma unitaria la distanza coseno vale 1 meno la
-- correlazione: 0 = stesso andamento, 1 = nessuna relazione, 2 = opposto.

CREATE TABLE IF NOT EXISTS cycle_atoms (
    cycle_id        INTEGER NOT NULL REFERENCES cycles(id) ON DELETE CASCADE,
    joint           TEXT NOT NULL REFERENCES joints(joint),
    plane           TEXT NOT NULL REFERENCES planes(plane),
    encoder_version TEXT NOT NULL,
    shape           vector(32) NOT NULL,
    rom             REAL NOT NULL,     -- escursione nel ciclo, in gradi
    peak            REAL NOT NULL,
    trough          REAL NOT NULL,
    provenance      TEXT NOT NULL REFERENCES provenance_kinds(provenance),
    coverage        REAL NOT NULL,     -- quota di frame validi dentro il ciclo
    PRIMARY KEY (cycle_id, joint, plane, encoder_version)
);
-- L'indice serve la ricerca per somiglianza fra sessioni diverse.
CREATE INDEX IF NOT EXISTS cycle_atoms_shape_idx
    ON cycle_atoms USING hnsw (shape vector_cosine_ops);
CREATE INDEX IF NOT EXISTS cycle_atoms_lookup_idx
    ON cycle_atoms (joint, plane, encoder_version);

-- ===========================================================================
-- 7. Riferimenti per gli alert
-- ===========================================================================
-- Valori fissi presi da letteratura scientifica, non calcolati dai dati che
-- carichiamo: un riferimento che nasce dalle stesse sessioni che deve giudicare
-- non e' un riferimento.
--
-- Due livelli:
--   scope='anatomical'  limiti articolari validi per chiunque, indipendenti dal
--                       gesto. Superarli significa o un movimento fuori dai
--                       limiti fisiologici o un errore di misura.
--   scope='sport'       intervalli attesi per uno specifico gesto. Uscirne non
--                       e' pericoloso di per se': e' un'anomalia da guardare.

CREATE TABLE IF NOT EXISTS reference_ranges (
    id          SERIAL PRIMARY KEY,
    scope       TEXT NOT NULL CHECK (scope IN ('anatomical', 'sport')),
    sport       TEXT,              -- NULL quando scope='anatomical'
    joint       TEXT NOT NULL REFERENCES joints(joint),
    plane       TEXT NOT NULL REFERENCES planes(plane),
    metric      TEXT NOT NULL CHECK (metric IN ('rom', 'peak_max', 'peak_min')),
    min_value   REAL,
    max_value   REAL,
    unit        TEXT NOT NULL DEFAULT 'deg',
    severity    TEXT NOT NULL DEFAULT 'warning'
                CHECK (severity IN ('info', 'warning', 'critical')),
    source      TEXT NOT NULL,     -- la fonte, per esteso
    source_url  TEXT,
    note        TEXT,
    -- NULLS NOT DISTINCT non e' un dettaglio: i riferimenti anatomici hanno
    -- `sport` nullo, e per il vincolo predefinito due NULL contano come diversi.
    -- Senza questa clausola `ON CONFLICT` non riconoscerebbe mai una riga
    -- anatomica gia' presente, e ogni caricamento ne aggiungerebbe una copia:
    -- gli alert anatomici scatterebbero tante volte quante sono le copie.
    UNIQUE NULLS NOT DISTINCT (scope, sport, joint, plane, metric)
);
CREATE INDEX IF NOT EXISTS reference_ranges_lookup_idx
    ON reference_ranges (joint, plane, metric);

-- Alert prodotti confrontando una sessione con i riferimenti. Si ricalcolano
-- interamente a ogni ingest della stessa sessione.
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
    deviation    REAL NOT NULL,    -- di quanto si esce dall'intervallo
    severity     TEXT NOT NULL,
    -- un alert su un angolo poco coperto o modellato vale meno: il giudizio
    -- resta all'agente, ma il dato per formularlo sta qui
    coverage     REAL,
    provenance   TEXT REFERENCES provenance_kinds(provenance),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS alerts_session_idx ON alerts (session_id);

-- ===========================================================================
-- 8. Viste di comodo
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

-- Atomi con il contesto della sessione: e' la base di ogni confronto.
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
