# Immagine per la conversione e il caricamento, senza la parte di rendering.
#
# Installa le sole dipendenze di cinematica e database e copia il progetto in
# /app.
#
# Prerequisiti: si usa insieme a docker-compose.yml, che la costruisce per il
# servizio `ingest`.
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy scipy "psycopg[binary]" pgvector
COPY . .
