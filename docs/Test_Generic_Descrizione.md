Descrizione del Test di Conversione e Rendering (codice in "src/conversion" e "src/visualization", dati in "data/"):

------------------------------------------------------------------------------------------

Il file "process_session_generic.py" implementa la pipeline di conversione e rendering dei dati grezzi di MediaPipe in un formato standardizzato, visualizzabile in 3D.

Utilizzo delle librerie:

json: Utilizzata per la lettura dei file di input (segments_frames.json, segments_summary.json) e scrittura dell'output JSON.

os: Utilizzata per la navigazione del filesystem, rilevamento automatico delle sessioni e creazione di directory di output.

sys: Utilizzata per la gestione dei codici di uscita in caso di errori.

argparse: Utilizzata per il parsing degli argomenti da riga di comando (session, --list-sessions).

numpy (np): Il motore matematico del progetto. Viene usata massicciamente per manipolare vettori e matrici 3D (calcolo del prodotto incrociato, norme, operazioni vettorializzate su frame, gradienti per le accelerazioni).

scipy.spatial.transform.Rotation: Fondamentale per la gestione delle rotazioni 3D. Permette di convertire agevolmente le matrici di rotazione create a partire dai segmenti ossei prima in Quaternioni (più stabili per evitare il gimbal lock) e poi in Angoli di Eulero per il calcolo degli angoli articolari veri e propri.

scipy.ndimage.gaussian_filter1d: Utilizzata per l'operazione di smoothing (filtraggio gaussiano) del segnale. Permette di "pulire" il rumore ad alta frequenza derivante dai marker o da MediaPipe prima di calcolare le derivate temporali, evitando picchi fittizi nelle velocità e accelerazioni.

ezc3d: (Caricata in modo opzionale con un blocco try-except). Se installata, permette di esportare i dati finali nel formato standard C3D (utilizzato nella biomeccanica).

Processo di conversione:

Vengono presi i dati da una cartella di sessione (es. running_*). Viene letta la coverage dal file segments_summary.json. La funzione auto_detect_available_segments controlla se la statistica coverage_pct di un segmento è strettamente maggiore dello 0%. Solo i segmenti rilevati vengono inclusi nella ricostruzione dello scheletro "dinamico".

Le coordinate grezze in entrata vengono corrette. MediaPipe ha l'asse Y che punta verso il basso e la Z orientata verso l'interno della camera. Il codice inverte i segni di Y e Z (convert_coords: [x, -y, -z]) per raddrizzare lo scheletro in un sistema canonico (Y verso l'alto).

Per ogni segmento, le matrici di rotazione e traslazione vengono salvate. Vengono prima convertite in quaternioni e poi in angoli di Eulero per consentire il calcolo degli angoli articolari. Segue una fase di smoothing (filtraggio gaussiano) del segnale per eliminare il rumore e calcolare le derivate temporali (velocità e accelerazione).

I file elaborati finiscono in una cartella chiamata *<nome_sessione>*_processed e sono salvati in tre formati diversi, ognuno con una precisa finalità:

Formato JSON (*<nome_sessione>*_generic.json):

Cosa contiene: Metadati della sessione, coordinate spaziali, quaternioni, velocità/accelerazioni angolari per ogni segmento, più tutti gli angoli articolari (Euler) calcolati.

A cosa serve: È un formato web-friendly, perfetto per essere re-ingerito da interfacce web, visori 3D online (rendering Three.js) o per passare i dati al sistema di backend senza vincolarsi a software specializzati.

Formato CSV (Un file per ogni segmento, es. *<nome_sessione>*_femur_left.csv):

Cosa contiene: Dati puramente tabellari e allineati per frame/tempo. Riporta coordinate di inizio, fine, centro, componenti dei quaternioni (w, x, y, z), vettori omega e alpha, e la lunghezza dell'osso.

A cosa serve: Fondamentale per il debug, per l'ispezione visiva dei dati crudi aprendoli rapidamente con software come Excel.

Formato C3D (*<nome_sessione>*_generic.c3d - se la libreria è disponibile):

Cosa contiene: Lo standard de facto in biomeccanica. Memorizza i "midpoints" dei segmenti come dati di traiettoria (POINT data) e i valori calcolati di velocità/accelerazione angolare come dati analogici (ANALOG data).

A cosa serve: Serve a garantire l'interoperabilità clinica e scientifica. Permette di importare i dati all'interno di software di simulazione biomeccanica.

------------------------------------------------------------------------------------------

Il file "c3d_viewer_open3d.py" implementa l'interfaccia grafica per la visualizzazione dei dati grezzi e convertiti.

Utilizzo delle librerie:

json: Caricamento dei dati di sessione dal formato JSON prodotto da process_session_generic.py. 

numpy: Calcolo del bounding box (centro, range), costruzione delle matrici di trasformazione 4×4, gestione array di punti 3D.

argparse: Parsing degli argomenti CLI (directory, --json, --c3d, --no-coord). 

os / pathlib.Path: Navigazione del filesystem, auto-detect delle sessioni running_* e dei file di input.   

time: Temporizzazione nella modalità legacy per rispettare il framerate reale durante la playback. 

open3d: Rendering 3D interattivo: finestra, geometrie (sfere, linee, assi), controllo camera, gestione eventi/tastiera.              

pytransform3d.rotations: Conversione quaternione → matrice di rotazione 3×3 (matrix_from_quaternion) per posizionare i reference frame.      

pytransform3d.transformations: Costruzione di trasformazioni omogenee 4×4.  

ezc3d: Lettura di file C3D binari nella funzione load_data_c3d.

Esecuzione e Funzionamento:

Vengono presi i file .c3d e .json dalla directory *_processed generata da process_session_generic.py. Il file .c3d può non essere presente. Successivamente, viene aperto un'interfaccia 3D(open3d) che permette di visualizzare i dati grezzi e convertiti in tempo reale (playback) seguendo le stesse tempistiche dell'acquisizione originale. L'utente può interagire con la scena, spostando la camera e ruotandola liberamente, o usare la tastiera per controllare la riproduzione (play/pausa, step avanti/indietro, velocità).

-----------------------------------------------------------------------------------

Il file c3d_viewer.py, come il precedente, permette di visualizzare i dati grezzi e convertiti.

Utilizzo delle librerie(json, numpy, argaprse, os/pathlib, time ed ezc3d vengono utilizzati come nel precedente):

matplotlib.pyplot: Creazione della finestra, dei subplot 3D e 2D, gestione del rendering.

matplotlib.widgets.Slider: Slider interattivo per navigare manualmente tra i frame.

matplotlib.widgets.Button: Pulsanti Play/Pause e Reset nella UI.

mpl_toolkits.mplot3d.Axes3D: Subplot con proiezione 3D per il rendering dello scheletro.

matplotlib.animation.FuncAnimation: Loop di animazione basato su timer per la playback automatica a framerate costante.

matplotlib.patches.mpatches: Rettangoli colorati nella legenda laterale.


Esecuzione e Funzionamento:

Come nel precedente si prendono i dati grezzi e convertiti dalla cartella _processed. Successivamente viene aperta una finestra che permette la visualizzazione dei dati. Si può interagire con la scena, spostando la camera e ruotandola liberamente, o usare la tastiera per controllare la riproduzione (play/pausa, step avanti/indietro, velocità). La differenza sta nel fatto che in questo caso viene usata una libreria diversa ovvero matplotlib al posto di open3d. Inoltre, in questo caso viene visualizzato anche un grafico 2D che mostra i dati grezzi e convertiti.

--------------------------------------------------------------------------------------------------------

Confronto tra i due script di rendering:

Aspetto						c3d_viewer.py (Matplotlib)	        c3d_viewer_open3d.py (Open3D)
Dipendenze					Solo matplotlib (standard)	        open3d + pytransform3d
Reference frame locali		Assenti						        Presenti (toggle con C)
Qualità rendering			Limitata (3D su canvas 2D)	        Hardware-accelerated, realistico
UI							Slider + Button nativi		        Pannello GUI o legacy keyboard
Coordinate transform		[x, -z, y]					        [x, y, z]
Uso ideale					Debug rapido	                    Analisi biomeccanica avanzata



