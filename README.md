# Pipeline per Dati Biomeccanici di Movimento

Questo progetto implementa una pipeline per l'elaborazione di dati di motion capture da sistemi di stima postura 3D.

## Workflow

1. **Trasformazione dei Dati**: I dati grezzi di postura 3D vengono elaborati utilizzando rappresentazioni quaternioniche per garantire robustezza matematica e convertiti in formati biomeccanici standard (es. C3D).

2. **Rendering 3D**: I dati di movimento elaborati vengono resi in 3D per visualizzazione e analisi.

3. **Archiviazione e Interrogazione**: I dati renderizzati vengono archiviati in un database relazionale che supporta query complesse tramite una CFG personalizzata.

## Casi d'Uso

- Analisi biomeccanica del movimento umano
- Valutazione delle prestazioni sportive
- Tracciamento del movimento riabilitativo
- Ricerca su animazione e movimento

## Struttura del repository

```
src/
  conversion/      # conversione dati grezzi -> quaternioni, angoli, C3D/JSON/CSV
  visualization/   # viewer 3D (Matplotlib, Open3D)
  database/        # schema PostgreSQL + ingestione (JSON canonico)
  mcp/             # server MCP con i tool di query biomeccanica
docs/              # documentazione di progetto (it)
data/              # sessioni motion-capture (raw/ e processed/) — NON versionate
requirements.txt   # dipendenze Python (Python 3.9)
```

I dati in `data/` sono ignorati da git. Gli script risolvono i percorsi rispetto
alla radice del repo, quindi si eseguono da qualsiasi directory:

```bash
python src/conversion/process_session_generic.py --list-sessions
python src/conversion/process_session_generic.py <nome_sessione>
python src/visualization/c3d_viewer.py
```