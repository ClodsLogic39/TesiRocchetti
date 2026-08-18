-- Ruoli a privilegio minimo. Eseguire DOPO schema.sql.
-- Le password si impostano a parte, mai nel repo:
--   ALTER ROLE biomech_mcp    PASSWORD '...';
--   ALTER ROLE biomech_ingest PASSWORD '...';

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

-- --------------------------------------------------------------------------
-- MCP: sola lettura su tutto il dato analitico. Nessuna scrittura, mai.
-- Gli alert li produce l'ingest; il server puo' solo consultarli.
-- --------------------------------------------------------------------------
-- Grant per colonna su sessions: canonical_path e' escluso perche' e' un
-- percorso del filesystem della macchina, informazione che al server non serve
-- e che non ha motivo di uscire.
GRANT SELECT (id, session_key, subject_id, sport, fps, n_frames, duration_s,
              source_format, capture_system, view, convention, biomech_version,
              body_mass_kg, height_m, ingested_at)
    ON sessions TO biomech_mcp;

-- Vocabolario
GRANT SELECT ON provenance_kinds, absence_reasons, planes, points, segments, joints,
                sports TO biomech_mcp;
-- Dati di movimento
GRANT SELECT ON point_trajectories, segment_series, joint_angles, absences
    TO biomech_mcp;
-- Cicli, forme e giudizi
GRANT SELECT ON cycles, cycle_metrics, cycle_atoms TO biomech_mcp;
GRANT SELECT ON reference_ranges, alerts TO biomech_mcp;
-- Viste di comodo
GRANT SELECT ON session_signals, session_absences, session_alerts, session_atoms
    TO biomech_mcp;
-- (Nessun GRANT su subjects_identity: il server MCP non puo' leggere l'identita'.)

-- --------------------------------------------------------------------------
-- Ingest: scrittura su tutto cio' che deriva da un file di acquisizione,
-- compresi gli alert, che vengono valutati durante il caricamento.
-- --------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON
    sessions, point_trajectories, segment_series, joint_angles, absences,
    cycles, cycle_metrics, cycle_atoms, alerts TO biomech_ingest;
-- I riferimenti di letteratura li aggiorna l'ingest all'avvio, ma non li
-- cancella: sono la base dei giudizi e non devono sparire per errore.
GRANT SELECT, INSERT, UPDATE ON reference_ranges TO biomech_ingest;
-- Il vocabolario e' fisso e viene dallo schema: all'ingest basta leggerlo,
-- tanto sono le chiavi esterne a rifiutare un nome che non esiste.
GRANT SELECT ON provenance_kinds, absence_reasons, planes, points, segments, joints,
                sports TO biomech_ingest;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO biomech_ingest;
-- L'identita' la scrive un processo amministrativo separato, non l'ingest.

-- Controprove attese (devono fallire tutte):
--   psql -U biomech_mcp    -c "UPDATE sessions SET sport='x';"
--   psql -U biomech_mcp    -c "SELECT * FROM subjects_identity;"
--   psql -U biomech_mcp    -c "SELECT canonical_path FROM sessions;"
--   psql -U biomech_mcp    -c "INSERT INTO alerts (session_id) VALUES (1);"
--   psql -U biomech_ingest -c "DELETE FROM reference_ranges;"
--   psql -U biomech_ingest -c "SELECT * FROM subjects_identity;"
