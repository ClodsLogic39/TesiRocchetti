# LaTeX essenziale per scrivere la tesi

Non ti serve toccare `report.tex` o `settings.tex`: sono gia' configurati.
Scrivi solo dentro i file dei capitoli in `include/2-chapters/`.

## Struttura del testo

- `\chapter{Titolo}` (gia' presente in ogni file), `\section{...}`,
  `\subsection{...}`, `\subsubsection{...}` per organizzare dentro un capitolo.
- Un a capo singolo nel sorgente non crea un nuovo paragrafo: serve una
  **riga vuota** fra un paragrafo e l'altro.

## Formattazione di base

- Grassetto: `\textbf{testo}` — Corsivo: `\textit{testo}` o `\emph{testo}`
- Elenco puntato:
  ```
  \begin{itemize}
    \item primo punto
    \item secondo punto
  \end{itemize}
  ```
- Elenco numerato: uguale ma con `enumerate` al posto di `itemize`.

## Caratteri da "scappare" (hanno un significato speciale in LaTeX)

| carattere | scrivi invece |
|---|---|
| `%` (inizia un commento) | `\%` |
| `&` | `\&` |
| `#` | `\#` |
| `_` | `\_` |
| `$` | `\$` |

Lettere accentate (a', e', ecc.) si scrivono normali, nessun escape richiesto.
Per le virgolette usa due apici: `` `` `` all'apertura e `''` alla chiusura,
non le dritte `"..."` (escono storte).

## Citare la bibliografia

`\cite{chiave}` produce il numero nel testo, es. `\cite{wu2002isb}` -> [1].
Le chiavi sono quelle in `include/3-back/bibliography.bib`. Piu' fonti insieme:
`\cite{wu2002isb,groodsuntay1983}`. Se una voce non e' mai citata con
`\cite{}`, non compare in bibliografia.

## Figure

```
\begin{figure}[!ht]
  \centering
  \includegraphics[width=0.7\textwidth]{figures/nome_immagine.png}
  \caption{Descrizione della figura.}
  \label{fig:nome_breve}
\end{figure}
```
Metti i file immagine dentro `figures/`. Per richiamarla nel testo:
`Come mostrato in Figura~\ref{fig:nome_breve}...`

## Tabelle

```
\begin{table}[!ht]
  \centering
  \begin{tabular}{ll}
    \hline
    Colonna 1 & Colonna 2 \\
    \hline
    valore A & valore B \\
    \hline
  \end{tabular}
  \caption{Descrizione della tabella.}
  \label{tab:nome_breve}
\end{table}
```

## Riferimenti incrociati

`\label{...}` subito dopo `\chapter`, `\section`, o dentro una figura/tabella.
`\ref{...}` per richiamarne il numero altrove nel testo. I numeri corretti
compaiono solo dopo aver ricompilato 2-3 volte (LaTeX li risolve un giro alla
volta).

## Codice sorgente (pacchetto listings, gia' configurato)

```
\begin{lstlisting}[language=python]
def somma(a, b):
    return a + b
\end{lstlisting}
```

## Matematica (se serve)

- In linea: `$x = 2$`
- Isolata e numerata:
  ```
  \begin{equation}
    E = mc^2
  \end{equation}
  ```

## Come compilare

- **Overleaf** (consigliato se non hai LaTeX installato): carichi la cartella,
  premi "Recompile", fa tutto da solo (bibliografia inclusa).
- **In locale**: `latexmk -pdf report.tex` fa in automatico la sequenza
  pdflatex -> bibtex -> pdflatex -> pdflatex necessaria per aggiornare
  indice, bibliografia e riferimenti incrociati.

## I 3 errori piu' comuni da principianti

1. Dimenticare una `}` di chiusura -> errore criptico molte righe dopo il
   punto vero. Se un errore non ha senso, controlla le parentesi graffe
   appena sopra.
2. Riga vuota mancante fra due paragrafi -> il testo si attacca in un unico
   blocco.
3. Compilare una sola volta e vedere `??` al posto di numeri di
   capitolo/figura/citazione -> serve ricompilare piu' volte (o usare
   `latexmk`, che lo fa da solo).
