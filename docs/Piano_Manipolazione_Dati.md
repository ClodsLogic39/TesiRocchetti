# Piano d'Azione — Database, Vettori, MCP e DSL Warning

## 1) Obiettivo operativo

Costruire una pipeline completa che:

1. acquisisce i dati da `.c3d` (e dai file derivati già prodotti),
2. li salva in PostgreSQL con supporto vettoriale (`pgvector`),
3. espone query biomeccaniche complesse tramite MCP,
4. usa una DSL snella e specifica solo per warning/alert rispetto a range ottimali.

Approccio chiave: **retrieval vettoriale per confronto/similarità + metriche biomeccaniche esplicite per spiegabilità**.

---

## 2) Architettura target (alto livello)

1. **Ingestion Layer**
   - Lettura C3D (`POINT` + `ANALOG`) e JSON `_generic.json` (angoli articolari, velocità, accelerazioni, quaternioni).
   - Normalizzazione etichette (giunto, lato, asse, fase, sport, sessione).
   - Segmentazione in cicli (es. pedalata, passo) con algoritmi deterministici.

2. **Storage Layer**
   - PostgreSQL come fonte autorevole.
   - `pgvector` per embedding dei cicli e retrieval di sessioni/cicli simili.

3. **Analytics/Query Layer**
   - Libreria metrica biomeccanica (ROM, picchi, timing, fase, variabilità, simmetria, ecc.).
   - Executor query che combina SQL + calcolo numerico.

4. **MCP Layer**
   - Tool MCP per ingestion, query complesse, confronti cross-sessione/cross-sport, explain dei risultati.

5. **Alert Layer (DSL)**
   - DSL limitata ai warning: policy di qualità movimento confrontate con un profilo “ideale”.

---

## 3) Piano Database PostgreSQL + pgvector

## 3.1 Tabelle principali

1. `athletes` (opzionale se multi-soggetto)
   - `id`, `code`, `metadata`

2. `sessions`
   - `id`, `session_key`, `athlete_id`, `sport`, `protocol`, `fps`, `n_frames`, `duration_s`, `source_path`, `created_at`

3. `cycles`
   - `id`, `session_id`, `cycle_index`, `start_frame`, `end_frame`, `duration_s`, `phase_meta`

4. `joint_signals`
   - serie temporali per ciclo o sessione
   - campi: `id`, `session_id`, `cycle_id`, `joint`, `side`, `plane`, `axis`, `signal_type` (`angle|omega|alpha`), `values`, `unit`

5. `points_series`
   - per dati marker/punti C3D
   - `id`, `session_id`, `label`, `x_values`, `y_values`, `z_values`, `quality`

6. `cycle_embeddings`
   - `id`, `session_id`, `cycle_id`, `embedding vector(N)`, `encoder_name`, `encoder_version`, `input_signature`

7. `reference_profiles`
   - profili ideali (es. corsa perfetta, pedalata ottimale)
   - `id`, `sport`, `level`, `joint`, `axis`, `metric_name`, `range_min`, `range_max`, `timing_min`, `timing_max`, `notes`

8. `alerts`
   - output della DSL warning
   - `id`, `session_id`, `cycle_id`, `rule_id`, `severity`, `message`, `observed_value`, `expected_range`, `created_at`

## 3.2 Indici essenziali

- B-tree su `(session_key)`, `(sport)`, `(joint, side, axis, signal_type)`, `(session_id, cycle_index)`.
- Indice vettoriale su `cycle_embeddings.embedding` (IVFFlat o HNSW, in base al volume dati).

## 3.3 Scelte pratiche dati

- Salvare sempre sia:
  - **raw normalizzato** (traccia completa),
  - **feature derivate** (ROM, picchi, timing, variabilità) per velocizzare query frequenti.
- Versionare pipeline/encoder per riproducibilità.

---

## 4) Modello vettoriale (senza training custom)

## 4.1 Strategia consigliata

Usare un encoder pre-addestrato **solo per embedding di ciclo**:

1. segmentazione ciclo (deterministica),
2. normalizzazione temporale del ciclo (resampling a lunghezza fissa),
3. embedding per retrieval di similarità,
4. decisione finale basata su metriche biomeccaniche esplicite.

## 4.2 Requisiti per l’encoder

- robustezza a rumore e piccole variazioni temporali,
- stabilità a velocità diverse (temporal normalization),
- embedding coerenti tra sessioni diverse dello stesso gesto.

## 4.3 Uso corretto degli embedding

- embedding = **filtro/ricerca candidati**,
- metriche biomeccaniche = **spiegazione e validazione**.

Quindi il confronto tra sessioni usa:
1. nearest-neighbor vettoriale per trovare cicli comparabili,
2. calcolo di delta interpretabili (ROM, timing, picchi, simmetria, ecc.).

---

## 5) MCP per query complesse

## 5.1 Tool MCP minimi

1. `ingest_c3d(path, sport, session_key?)`
2. `ingest_directory(path, recursive=true, sport?)`
3. `list_sessions(filters?)`
4. `segment_cycles(session_key, strategy)`
5. `compute_features(session_key|cycle_id)`
6. `query_metric(spec)` — query strutturata
7. `compare_sessions(spec)`
8. `compare_sports(spec)`
9. `find_similar_cycles(spec, top_k)`
10. `run_warning_rules(session_key|cycle_id, profile_id)`

## 5.2 Copertura query di `IdeeQuery.md`

Le 20 query sono coperte con mappatura diretta:

- metriche base: ROM, picchi, timing, velocità, accelerazione, impulso,
- profili fase: appoggio/oscillazione/ciclo,
- coordinazione: fase relativa, angolo-angolo,
- robustezza: variabilità, simmetria,
- confronto: sessioni/sport,
- regole evento: stabilità catena, rilevamento gesto.

## 5.3 Formato output MCP

Ogni risposta dovrebbe includere:

1. `input_spec` normalizzato,
2. `selected_data` (sessione/ciclo/segnale usato),
3. `result` (metrica),
4. `explain` (formula e parametri),
5. `confidence` (eventuale score retrieval),
6. `warnings` (limiti dati o assunzioni).

---

## 6) DSL dedicata ai warning (non alle query generiche)

## 6.1 Obiettivo DSL

Definire policy tipo:

- “ROM ginocchio sinistro in sagittale tra 45 e 70 gradi”
- “Timing picco flessione tra 60% e 75% ciclo”
- “Simmetria sx-dx > 90%”

## 6.2 Struttura regola (esempio concettuale)

`WHEN <metrica(joint, side, plane, phase)> NOT IN <range> THEN <severity,message>`

Componenti:

1. metrica target,
2. finestra temporale/fase,
3. range o soglia,
4. severità (`INFO|WARN|CRITICAL`),
5. messaggio standardizzato.

## 6.3 Flusso warning

1. MCP recupera ciclo/sessione corrente,
2. executor calcola metrica richiesta,
3. DSL engine valuta regola,
4. salva in `alerts`,
5. restituisce alert con motivazione numerica.

---

## 7) Sequenza di implementazione (roadmap)

## Fase 1 — Fondazioni dati

1. Definizione schema PostgreSQL + `pgvector`.
2. Ingestion robusta C3D/JSON.
3. Normalizzazione metadati (sessione, sport, giunto, lato, piano, asse).

**Deliverable:** database popolato e interrogabile su metriche base.

## Fase 2 — Cicli e embedding

1. Segmentazione cicli (pedalata/corsa) deterministica.
2. Normalizzazione temporale ciclo.
3. Calcolo embedding e persistenza in `cycle_embeddings`.

**Deliverable:** retrieval per similarità tra cicli/sessioni.

## Fase 3 — MCP query engine

1. Tool MCP per query complesse (tutte le query di IdeeQuery).
2. Endpoint `compare_sessions` ibrido (vettori + metriche esplicite).
3. Output con explain strutturato.

**Deliverable:** interrogazione completa via MCP.

## Fase 4 — DSL warning

1. Definizione grammatica minima DSL (solo alert).
2. Rule engine + severità.
3. Collegamento con profili di riferimento (`reference_profiles`).

**Deliverable:** alert automatici rispetto al modello “movimento ottimale”.

## Fase 5 — Validazione e tuning

1. Verifica coerenza metriche su sessioni note.
2. Taratura soglie warning con esperto dominio.
3. Hardening (logging, audit, versionamento pipeline).

**Deliverable:** sistema affidabile e ripetibile.

---

## 8) Criteri di accettazione

Il piano è considerato completato quando:

1. i `.c3d` sono ingeriti senza perdita dei segnali utili,
2. le query di `IdeeQuery.md` sono eseguibili via MCP con output spiegabile,
3. il confronto sessioni usa retrieval vettoriale + delta metrici interpretabili,
4. la DSL produce warning coerenti con i range del profilo di riferimento,
5. ogni risultato è tracciabile (sessione, ciclo, formula, parametri).

---

## 9) Rischi e mitigazioni

1. **Dati incompleti/noisy**  
   Mitigazione: quality flag, imputazione controllata, confidence nei risultati.

2. **Embedding poco stabili cross-protocollo**  
   Mitigazione: normalizzazione ciclo, confronto ibrido con metriche esplicite.

3. **Warning troppo sensibili o troppo permissivi**  
   Mitigazione: soglie multi-livello e calibrazione iterativa con dati reali.

4. **Ambiguità terminologica nelle query**  
   Mitigazione: payload MCP strutturato e vocabolario controllato; DSL limitata ai warning.

---

## 10) Decisione architetturale finale

1. **MCP** = strato operativo principale per tutte le query biomeccaniche.
2. **Vettori (`pgvector`)** = retrieval/comparazione cicli e sessioni.
3. **DSL** = solo regole di warning/alert rispetto a profili di riferimento.
4. **Metriche esplicite** = sempre presenti per spiegabilità e validazione tecnica.
