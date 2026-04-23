# Pipeline per Dati Biomeccanici di Movimento

Questo progetto implementa una pipeline per l'elaborazione di dati di motion capture da sistemi di stima postura 3D.

## Workflow

1. **Trasformazione dei Dati**: I dati grezzi di postura 3D vengono elaborati utilizzando rappresentazioni quaternioniche per garantire robustezza matematica e convertiti in formati biomeccanici standard (es. C3D).

2. **Rendering 3D**: I dati di movimento elaborati vengono resi in 3D per visualizzazione e analisi.

3. **Archiviazione e Interrogazione**: I dati renderizzati vengono archiviati in un database relazionale che supporta query complesse tramite una SDT personalizzata.

## Casi d'Uso

- Analisi biomeccanica del movimento umano
- Valutazione delle prestazioni sportive
- Tracciamento del movimento riabilitativo
- Ricerca su animazione e movimento