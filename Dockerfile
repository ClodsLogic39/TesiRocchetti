# Immagine minima per conversione + ingest (no rendering/open3d).
FROM python:3.11-slim
WORKDIR /app
# Dipendenze core: cinematica (numpy/scipy) + DB (psycopg/pgvector).
RUN pip install --no-cache-dir numpy scipy "psycopg[binary]" pgvector
COPY . .
