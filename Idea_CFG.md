Creazione di un motore di interrogazione basato su una Grammatica Libera da Contesto (CFG) dedicato specificamente all'analisi di dati biomeccanici (provenienti da motion capture).

L'obiettivo principale è permettere agli utenti di estrarre e calcolare metriche complesse utilizzando query espresse in una forma molto simile al linguaggio naturale (ad esempio: "ROM del ginocchio sinistro nel piano sagittale durante corsa" o "Simmetria ginocchio"), nascondendo completamente la complessità dei calcoli e della struttura dei dati sottostanti.

Architettura concettuale:

Comprensione della richiesta (Lexer & Parser): Il sistema prende in input la frase dell'utente e la scompone in concetti biomeccanici chiave (identificando qual è la Metrica, il Giunto, il Lato, il Piano di movimento, ecc.). Tramite le regole della grammatica, questi concetti vengono organizzati in una struttura logica ad albero che rappresenta l'intenzione dell'utente.

Collegamento ai dati (Analisi Semantica): La richiesta logica viene tradotta in riferimenti precisi all'interno di un database JSON. Questo database contiene tutte le informazioni pre-processate (angoli articolari, coordinate 3D, eventi del passo) ottenute tramite calcoli con quaternioni.

Calcolo ed Esecuzione (Query Executor): Una volta capito cosa misurare e su quali dati, il sistema instrada la richiesta a una specifica funzione di calcolo matematico (che calcolerà, ad esempio, le velocità angolari, i picchi di flessione, la simmetria tra arti o il Range of Motion) e restituisce il risultato finale all'utente.

## 1. Panoramica dell'Architettura


  QUERY IN LINGUAGGIO NATURALE                
  "ROM del ginocchio sinistro nel piano sagittale durante corsa"

                           │
                           ▼

  LEXER (Tokenizzazione)                                        
  [ROM] [ginocchio] [sinistro] [piano] [sagittale] [durante]   

                           │
                           ▼

  PARSER (CFG - Context Free Grammar)                           
  Genera Abstract Syntax Tree                            

                           │
                           ▼

  SEMANTIC ANALYZER                                             
  Risoluzione riferimenti → costruzione query plan              
                           │
                           ▼
  QUERY EXECUTOR                                                
  Calcola metriche dal database JSON                            


## 2. Grammatica Formale (CFG)

### Produzioni Principali


QUERY          → METRICA [LATO] [GIUNTO] [PIANO] [FASE] [FILTRO]

METRICA        → "ROM" | "PiccoFlessione" | "PiccoEstensione" | "Timing" |
                 "Velocità" | "Accelerazione" | "Impulso" | "Simmetria" |
                 "Variabilità" | "Stabilità" | "FaseRelativa" | "AngoloAngolo" |
                 "ConfrontoSport" | "ConfrontoSessioni" | "RilevamentoGesto" |
                 "ProfiloAppoggio" | "ProfiloOscillazione" | "AnalisiCiclo"

LATO           → "sinistro" | "destro" | "entrambi"

GIUNTO         → "anca" | "ginocchio" | "caviglia" | "spalla" | "gomito" |
                 "polso" | "colonna" | "tronco" | "bacino"

PIANO          → "sagittale" | "frontale" | "trasversale"

FASE           → "appoggio" | "oscillazione" | "intero_ciclo" | "stacco" | "atterraggio"

FILTRO         → "durante" SESSIONE | "tra" SESSIONE "e" SESSIONE |
                 "sport" SPORT

SPORT          → "corsa" | "ciclismo" | "nuoto" | "calcio" | "basket" | "tennis"


### Esempi di Parsing

| Input Query | AST Generato |
|-------------|--------------|
| "ROM ginocchio sinistro piano sagittale" | `QueryNode(metric="ROM", joint="knee_left", plane="sagittal")` |
| "Picco flessione anca destra durante appoggio" | `QueryNode(metric="PiccoFlessione", joint="hip_right", phase="stance")` |
| "Simmetria ginocchio tra sessione 1 e sessione 2" | `QueryNode(metric="Simmetria", joint="knee", compare=["session_1", "session_2"])` |

---

## 3. Lexer - Tokenizzazione

### Tipi di Token

class TokenType:
    # Token principali
    METRICA = "METRICA"        # ROM, Picco, Velocità...
    GIUNTO = "GIUNTO"          # ginocchio, anca, caviglia...
    LATO = "LATO"              # sinistro, destro, bilateral
    PIANO = "PIANO"            # sagittale, frontale, trasversale
    FASE = "FASE"              # appoggio, oscillazione
    
    # Token ausiliari
    PREPOSIZIONE = "PREP"      # di, nel, durante, tra
    CONGIUNZIONE = "CONJ"      # e, o, ma
    NUMERO = "NUM"             # soglie, percentuali
    
    # Token speciali
    SESSIONE = "SESSION"       # identificatore sessione
    SPORT = "SPORT"            # tipo di attività
    
    # Token di controllo
    EOF = "EOF"
    ERROR = "ERROR"


### Esempio di Tokenizzazione

```
Input: "ROM del ginocchio sinistro piano sagittale"

Tokens:
  [METRICA, "ROM"]
  [PREPOSIZIONE, "del"]
  [GIUNTO, "ginocchio"]
  [LATO, "sinistro"]
  [PIANO, "sagittale"]

