# Dai keypoint alla conoscenza

Una pipeline che rende il movimento umano interrogabile da un agente
conversazionale **dichiarando sempre quanto ci si possa fidare della risposta**.

Legge acquisizioni di laboratorio (C3D a marker) o stime di posa da video
(MediaPipe), le porta a un vocabolario anatomico unico di diciotto centri
articolari, calcola angoli secondo la convenzione ISB, segmenta il gesto in cicli,
scrive un file C3D *canonico* autosufficiente, e da lì popola un database
PostgreSQL che un server MCP espone a un agente con diciotto strumenti.

La convenzione angolare non è asserita: su 118 prove di laboratorio contenenti sia
i marker sia le uscite di un modello clinico calibrato, ogni giunto confrontabile
ricade entro **3.2 gradi** di scarto quadratico medio. Lo strumento che lo misura è
nel progetto e si esegue con un comando.

Documentazione completa in [`docs/Tesi.md`](docs/Tesi.md) (il lavoro e i suoi
risultati) e [`docs/Progetto.md`](docs/Progetto.md) (le decisioni di progetto, con
il motivo di ognuna).

---

## Cosa serve

- **Docker** con Compose, per il database
- **Python 3.11+** per la pipeline
- **OpenSSL**, per generare il certificato del canale cifrato

Il visualizzatore 3D richiede `open3d`, che non convive con tutte le versioni di
Python: se il tuo interprete non lo accetta, va installato in un ambiente separato
(vedi [Visualizzatore](#visualizzatore-3d)). Tutto il resto funziona senza.

---

## Installazione

### 1. Dipendenze Python

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

### 2. Certificato per il canale cifrato

Il database rifiuta connessioni in chiaro. Il certificato è autofirmato e **non**
sta nel repo: va generato una volta.

```bash
mkdir -p certs && openssl req -new -x509 -days 365 -nodes -keyout certs/server.key -out certs/server.crt -subj "/CN=localhost" && chmod 600 certs/server.key
```

Postgres rifiuta una chiave leggibile da altri: il `chmod 600` non è opzionale.

### 3. Configurazione

```bash
cp .env.example .env
```

Apri `.env` e scegli `POSTGRES_PASSWORD`. È la password dell'amministratore del
database: `.env` è escluso dal versionamento e deve restarci.

### 4. Avvio del database

```bash
docker compose -f docker-compose.yml -f docker-compose.tls.yml up -d db
```

Al primo avvio il container crea da sé lo schema e i due ruoli a privilegio
minimo, montando `src/database/schema.sql` e `db/setup_roles.sql`.

### 5. Password dei ruoli operativi

I ruoli nascono **senza password**, perché una password nel repo non è una
password. Scegline due e impostale:

```bash
docker compose exec db psql -U biomech_admin -d biomech -c "ALTER ROLE biomech_ingest PASSWORD 'scegline_una'; ALTER ROLE biomech_mcp PASSWORD 'scegline_un_altra';"
```

Poi esportale per gli strumenti da riga di comando:

```bash
export INGEST_DB_DSN="host=127.0.0.1 port=5432 dbname=biomech user=biomech_ingest sslmode=require password=scegline_una"
```

`sslmode=require` fa **fallire** la connessione se il server non offre TLS, invece
di ripiegare in chiaro senza dirlo.

### 6. Valori di letteratura

Gli intervalli di riferimento contro cui si valutano gli scostamenti si caricano
una volta sola:

```bash
python -m src.database.ingest --riferimenti
```

### 7. Verifica

```bash
python -m pytest tests/ -q
```

95 test, nessuno dei quali richiede un database attivo.

Nove si saltano dichiarando il motivo: confrontano gli angoli calcolati con quelli
di un modello clinico, e servono C3D di laboratorio che contengano entrambi. Sono
registrazioni di persone e non stanno nel repo. Gli altri 86 devono passare.

---

## Usare il progetto

### Convertire un'acquisizione

La conversione produce un C3D *canonico*: un file che contiene tutto ciò che serve
a riempire il database e niente che vada cercato altrove.

```bash
python -m src.conversion.process --elenco
```

```bash
python -m src.conversion.process DatiGrezzi/una_sessione.c3d --sport walking
```

Il gesto conta e va dichiarato quando la sorgente non lo sa: decide **come** si
segmenta in cicli e **quali** riferimenti di letteratura si applicheranno. I gesti
riconosciuti sono `walking`, `running`, `cycling`, `swimming`.

Per convertire una cartella intera:

```bash
python -m src.conversion.process DatiGrezzi --tutte --sport walking
```

Se restano etichette che il lettore non sa interpretare, **si ferma**: nessun file
prodotto, nessuna riga scritta, solo l'elenco di ciò che non ha capito.

### Caricare nel database

```bash
python -m src.database.ingest DatiElaborati/una_sessione.c3d --subject-id SBJ-001
```

Il legame con il soggetto vive **solo** nel database: nel C3D canonico non c'è
nemmeno lo pseudonimo, perché quel file può essere copiato o condiviso.

Il soggetto deve esistere in anagrafica, che è una tabella separata a cui né il
ruolo di scrittura né quello di lettura hanno accesso:

```bash
docker compose exec db psql -U biomech_admin -d biomech -c "INSERT INTO subjects_identity (subject_id, full_name) VALUES ('SBJ-001', 'Nome Cognome');"
```

Per caricare una cartella: `python -m src.database.ingest DatiElaborati --tutte`

Il caricamento è **idempotente**: ricaricare lo stesso file non duplica nulla.

### Interrogare da un agente

Il server MCP si avvia così:

```bash
MCP_DB_DSN="host=127.0.0.1 port=5432 dbname=biomech user=biomech_mcp sslmode=require password=scegline_un_altra" python -m src.mcp.server
```

Per collegarlo a un client MCP, dichiaralo nella configurazione del client con lo
stesso comando e la `MCP_DB_DSN` nel blocco delle variabili d'ambiente.

I diciotto strumenti coprono:

| | |
|---|---|
| **misure** | `metric`, `angular_kinematics`, `peak_timing`, `gesture_detection` |
| **cosa c'è** | `list_sessions`, `resolve_session`, `describe_session`, `list_cycles` |
| **confronti** | `compare_sessions`, `compare_sports`, `compare_session_shapes`, `find_similar_cycles` |
| **dentro una sessione** | `session_repeatability`, `side_shape_symmetry`, `symmetry_index`, `relative_phase` |
| **giudizi** | `list_alerts` |
| **rendering** | `visualize_session` |

Ogni risposta ha la stessa forma, e il vincolo non è stilistico:

```json
{
  "input_spec":    "cosa e' stato chiesto, normalizzato",
  "selected_data": "quale segnale, su quale intervallo, con quale copertura",
  "result":        "il valore, o null",
  "explain":       "con quale formula",
  "warnings":      ["cio' di cui tenere conto"]
}
```

`result: null` con `warnings: ["signal_unresolved"]` non è una risposta mancata: è
la risposta corretta a una domanda su un segnale che non esiste, ed è distinguibile
da `result: null` con `warnings: ["no_data_or_fps"]`, che dice un'altra cosa.

### Visualizzatore 3D

Apre una finestra con la figura ricostruita dal C3D canonico:

```bash
python -m src.visualization.viewer_open3d --c3d DatiElaborati/una_sessione.c3d
```

`--coord` mostra le terne dei segmenti; `--png file.png --frame 150` rende un
fotogramma su file senza aprire nulla.

Se `open3d` sta in un altro interprete, indicalo al server MCP:

```bash
export BIOMECH_VIEWER_PYTHON=/percorso/al/python/con/open3d
```

Su una macchina condivisa il visualizzatore si disattiva, e lo strumento continua a
rispondere spiegando perché:

```bash
export BIOMECH_VIEWER_DISABLED=1
```

### Validare la convenzione angolare

Se hai C3D che contengono sia i marker sia le uscite di un modello clinico, questo
strumento **svuota** gli angoli della sorgente, li ricalcola dai soli centri
articolari e confronta i due:

```bash
python -m tools.validazione_clinica percorso/ai/dati
```

Lo svuotamento è il punto: senza, il convertitore conserverebbe i valori originali
e il confronto darebbe scarto zero — un numero paragonato con se stesso.

---

## Come leggere le risposte

Tre cose rendono il progetto diverso da un archivio interrogabile, e vale la pena
sapere che ci sono.

**La provenienza.** Ogni valore dichiara da dove viene: `measured` (letto dalla
sorgente e mai ricalcolato), `derived` (calcolato in modo deterministico),
`modelled` (dipende da un'assunzione dichiarata), `filled` (interpolato su un buco
breve). Un confronto fra una serie misurata e una modellata resta possibile, ma la
risposta lo segnala.

**La copertura.** La frazione di fotogrammi validi è una colonna del database, non
un dato da ricavare leggendo l'array. Un angolo troppo poco coperto **non viene
giudicato affatto**: il massimo di un decimo del gesto non è il massimo del gesto.

**L'assenza.** Ciò che manca è registrato con il motivo — `not_in_source`,
`low_coverage`, `missing_inputs`, `no_time_base`, `not_observable`. L'agente può
rispondere *"non posso darti l'angolo del ginocchio perché la tibia non è
tracciata"* invece di restituire silenzio.

Gli avvisi sui cicli meritano una lettura. `cycles_from_surrogate` dice che il
ciclo viene da un angolo articolare e non dal contatto al suolo;
`primary_signal_low_coverage` che il segnale migliore non era abbastanza visibile e
si è ripiegato; `single_gesture` che nessun ciclo è stato trovato e la prova intera
è trattata come un gesto unico.

---

## Struttura

```
src/conversion/     dalle sorgenti al C3D canonico
    readers.py        riconosce i punti, legge gli eventi di appoggio
    anatomy.py        il vocabolario: punti, segmenti, giunti, provenienza
    angles.py         terne, quaternioni, Eulero ZXY
    segmentation.py   cicli, con la gerarchia dei metodi per gesto
    canonical.py      scrive e rilegge il C3D canonico
    process.py        il comando di conversione
src/database/       dal C3D al database
    schema.sql        lo schema, montato dal container all'avvio
    ingest.py         il caricamento, in una sola transazione
    embeddings.py     i vettori di forma dei cicli
    similarity.py     i confronti, calcolati da pgvector
src/alerts/         i riferimenti di letteratura e la loro valutazione
src/mcp/            il server MCP e le sue query
src/visualization/  il visualizzatore
db/setup_roles.sql  i ruoli a privilegio minimo
tools/              la validazione clinica
tests/              95 test, nessun database richiesto
                      (9 saltati senza dati di laboratorio)
```

---

## Sicurezza

Tre ruoli con privilegi disgiunti: l'amministratore possiede lo schema,
`biomech_ingest` scrive solo ciò che deriva da un file di acquisizione,
`biomech_mcp` legge e basta. **Nessuno dei due ha accesso all'anagrafica.**

Il canale è cifrato in TLSv1.3. Le password stanno in `.env` o nell'ambiente, mai
nel codice.

**Cosa non va mai committato**, ed è già escluso da `.gitignore`: `.env`, la
cartella `certs/`, e soprattutto `DatiGrezzi/` — le acquisizioni sono registrazioni
di persone, e il nome del file diventa la chiave di sessione visibile all'agente.
Usa nomi non parlanti.

Il modello di sicurezza è pensato per un uso **locale**. Un'esposizione pubblica
richiederebbe autenticazione degli agenti, limitazione della frequenza delle
richieste, e il visualizzatore disattivato.

---

## Limiti dichiarati

Sono misurati, e stanno per esteso nel capitolo 6 di [`docs/Tesi.md`](docs/Tesi.md).

I riferimenti di letteratura valgono per il **cammino in piano**: applicati a prove
in pendenza segnalano l'adattamento normale alla salita come anomalia (4.5-5.5
scostamenti per sessione contro 1.7 in piano).

Una ripresa da telefono che taglia le gambe fuori dal fotogramma non produce cicli
dal contatto al suolo, e nessuno strato di elaborazione recupera un dato che non è
stato registrato. Il ripiego sull'oscillazione del braccio funziona ma resta un
surrogato con dispersione elevata, e la risposta lo dichiara.
