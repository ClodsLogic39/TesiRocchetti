"""Ogni istruzione SQL del progetto deve citare cose presenti nello schema.

Il controllo e' statico: legge le stringhe SQL dai sorgenti Python, ne estrae i
nomi di tabella e di colonna e li confronta con quelli dichiarati in
`schema.sql`. Comprende una prova che il controllo stesso reagisca quando una
colonna sparisce.

    python -m pytest tests/test_sql_coerente.py -q

Prerequisiti: pytest; il file `src/database/schema.sql` e i moduli Python del
progetto. Non serve una connessione al database.
"""
import ast
import os
import re

# --- Percorsi e parole da ignorare ---
# Posizione dello schema e termini che compaiono al posto di un nome di colonna e
# non vanno verificati.
RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = os.path.join(RADICE, "src", "database", "schema.sql")

_NON_COLONNE = {"*", "count", "sum", "avg", "min", "max", "now", "coalesce",
                "distinct", "as", "case", "when", "then", "else", "end", "null",
                "true", "false", "and", "or", "not", "on", "using"}


# --- Estrazione dai sorgenti ---
# Tabelle e colonne dichiarate nello schema, stringhe SQL trovate nei moduli
# Python e nomi che ciascuna istruzione cita.
def _schema():
    with open(SCHEMA) as f:
        return f.read()


def tabelle_e_colonne(testo):
    out = {}
    for nome, corpo in re.findall(
            r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", testo, re.S):
        colonne = set()
        for riga in corpo.split("\n"):
            riga = riga.strip()
            if (not riga or riga.startswith(("--", "PRIMARY", "UNIQUE", "FOREIGN",
                                             "CHECK", ")"))):
                continue
            colonne.add(riga.split()[0])
        out[nome] = colonne
    for nome in re.findall(r"CREATE OR REPLACE VIEW (\w+)", testo):
        out[nome] = None
    return out


def stringhe_sql(percorso):
    with open(percorso) as f:
        albero = ast.parse(f.read(), percorso)
    parole = ("SELECT ", "INSERT INTO", "UPDATE ", "DELETE FROM")
    trovate, dentro_f = [], set()
    for nodo in ast.walk(albero):
        if isinstance(nodo, ast.JoinedStr):
            pezzi = []
            for parte in nodo.values:
                if isinstance(parte, ast.Constant) and isinstance(parte.value, str):
                    pezzi.append(parte.value)
                    dentro_f.add(id(parte))
                else:
                    pezzi.append(" calcolato_a_runtime ")
            testo = "".join(pezzi)
            if any(p in testo.upper() for p in parole):
                trovate.append((nodo.lineno, testo))
    for nodo in ast.walk(albero):
        if (isinstance(nodo, ast.Constant) and isinstance(nodo.value, str)
                and id(nodo) not in dentro_f):
            if any(p in nodo.value.upper() for p in parole):
                trovate.append((nodo.lineno, nodo.value))
    return trovate


def moduli_python():
    for radice, cartelle, file in os.walk(os.path.join(RADICE, "src")):
        cartelle[:] = [c for c in cartelle if c != "__pycache__"]
        for f in file:
            if f.endswith(".py"):
                yield os.path.join(radice, f)


def _normalizza(sql):
    return " ".join(sql.replace("\n", " ").split())


def _riferimenti(sql):
    s = _normalizza(sql)
    temporanee = {m.group(1).lower()
                  for m in re.finditer(r"(?:WITH|,)\s+(\w+)\s+AS\s*\(", s, re.I)}
    parole = {"lateral", "select", "unnest", "set", "calcolato_a_runtime"}
    tabelle = set()
    for m in re.finditer(r"\b(?:FROM|INTO|JOIN|UPDATE)\s+([A-Za-z_]\w*)", s, re.I):
        nome = m.group(1)
        if nome.lower() not in parole | temporanee:
            tabelle.add(nome)
    coppie = []
    for m in re.finditer(r"INSERT\s+INTO\s+(\w+)\s*\(([^)]*)\)", s, re.I):
        colonne = [c.strip().strip('"') for c in m.group(2).split(",")]
        if all(re.fullmatch(r"\w+", c or "") for c in colonne):
            coppie.append((m.group(1), colonne))
    for m in re.finditer(r"SELECT\s+(.+?)\s+FROM\s+(\w+)(?:\s|$)", s, re.I):
        campi, tabella = m.group(1), m.group(2)
        if any(x in campi for x in "()*") or tabella.lower() in temporanee:
            continue
        colonne = [c.strip().strip('"') for c in campi.split(",")]
        if all(re.fullmatch(r"[A-Za-z_]\w*", c or "") for c in colonne) \
                and not any(c.lower() in _NON_COLONNE for c in colonne):
            coppie.append((tabella, colonne))
    if len(tabelle - temporanee) == 1:
        (sola,) = tabelle - temporanee
        for m in re.finditer(r"ORDER\s+BY\s+(.+?)(?:\bLIMIT\b|$)", s, re.I):
            for pezzo in m.group(1).split(","):
                parole = pezzo.split()
                if not parole:
                    continue
                nome = parole[0].split(".")[-1]
                if (re.fullmatch(r"\w+", nome) and nome.lower() not in _NON_COLONNE
                        and nome != "calcolato_a_runtime"):
                    coppie.append((sola, [nome]))
        for m in re.finditer(r"WHERE\s+(\w+)\s*(?:=|<>|>|<|ILIKE|LIKE)", s, re.I):
            if m.group(1).lower() not in _NON_COLONNE:
                coppie.append((sola, [m.group(1)]))
    coppie = [(t, c) for t, c in coppie
              if "calcolato_a_runtime" not in " ".join(c)]
    return tabelle, coppie


# --- Verifiche ---
# Che ogni istruzione citi tabelle e colonne esistenti, e che il controllo si
# accorga davvero della sparizione di una colonna.
def test_ogni_sql_python_parla_di_tabelle_esistenti():
    schema = tabelle_e_colonne(_schema())
    problemi = []
    for percorso in moduli_python():
        for riga, sql in stringhe_sql(percorso):
            tabelle, _ = _riferimenti(sql)
            for t in tabelle:
                if t not in schema:
                    problemi.append(
                        f"{os.path.relpath(percorso, RADICE)}:{riga} tabella '{t}'")
    assert not problemi, "tabelle inesistenti:\n  " + "\n  ".join(sorted(problemi))


def test_ogni_sql_python_parla_di_colonne_esistenti():
    schema = tabelle_e_colonne(_schema())
    problemi = []
    for percorso in moduli_python():
        for riga, sql in stringhe_sql(percorso):
            _, coppie = _riferimenti(sql)
            for tabella, colonne in coppie:
                attese = schema.get(tabella)
                if attese is None:
                    continue
                for c in colonne:
                    if c not in attese:
                        problemi.append(
                            f"{os.path.relpath(percorso, RADICE)}:{riga} "
                            f"{tabella}.{c}")
    assert not problemi, "colonne inesistenti:\n  " + "\n  ".join(sorted(problemi))


def test_il_controllo_intercetta_davvero_una_colonna_sparita():
    schema = tabelle_e_colonne(_schema())
    _, coppie = _riferimenti(
        "SELECT subject_id, full_name, colonna_inventata FROM subjects_identity")
    assert coppie, "la query non e' stata riconosciuta come verificabile"
    tabella, colonne = coppie[0]
    assert "colonna_inventata" in colonne
    assert "colonna_inventata" not in schema[tabella]
