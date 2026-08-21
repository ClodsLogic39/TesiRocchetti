-- Ruoli di connessione a privilegio minimo.
--
-- Crea i due ruoli usati dall'applicazione, se non esistono, e assegna a ciascuno
-- i soli permessi che gli servono. La tabella con l'identita' dei soggetti non
-- viene concessa a nessuno dei due.
--
-- Prerequisiti: eseguire dopo `src/database/schema.sql`, con un utente
-- amministrativo. Le password si impostano separatamente:
--
--   ALTER ROLE biomech_mcp    PASSWORD '...';
--   ALTER ROLE biomech_ingest PASSWORD '...';

-- --- Creazione dei ruoli e azzeramento dei permessi ---
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'biomech_mcp') THEN
    CREATE ROLE biomech_mcp LOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'biomech_ingest') THEN
    CREATE ROLE biomech_ingest LOGIN;
  END IF;
END $$;

REVOKE ALL ON ALL TABLES IN SCHEMA public FROM biomech_mcp, biomech_ingest;

-- --- Ruolo del server MCP: sola lettura ---
-- Sulle sessioni il permesso e' concesso colonna per colonna; sul resto del dato
-- analitico, sul vocabolario e sulle viste e' un SELECT pieno.
GRANT SELECT (id, session_key, subject_id, sport, fps, n_frames, duration_s,
              source_format, capture_system, view, convention, biomech_version,
              body_mass_kg, height_m, ingested_at)
    ON sessions TO biomech_mcp;

GRANT SELECT ON provenance_kinds, absence_reasons, planes, points, segments, joints,
                sports TO biomech_mcp;
GRANT SELECT ON point_trajectories, segment_series, joint_angles, absences
    TO biomech_mcp;
GRANT SELECT ON cycles, cycle_metrics, cycle_atoms TO biomech_mcp;
GRANT SELECT ON reference_ranges, alerts TO biomech_mcp;
GRANT SELECT ON session_signals, session_absences, session_alerts, session_atoms
    TO biomech_mcp;

-- --- Ruolo di caricamento: scrittura sui dati di un'acquisizione ---
-- Comprende gli alert, che vengono valutati durante il caricamento, e i
-- riferimenti di letteratura, che l'ingest aggiorna. Il vocabolario resta in
-- sola lettura.
GRANT SELECT, INSERT, UPDATE, DELETE ON
    sessions, point_trajectories, segment_series, joint_angles, absences,
    cycles, cycle_metrics, cycle_atoms, alerts TO biomech_ingest;
GRANT SELECT, INSERT, UPDATE ON reference_ranges TO biomech_ingest;
GRANT SELECT ON provenance_kinds, absence_reasons, planes, points, segments, joints,
                sports TO biomech_ingest;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO biomech_ingest;

