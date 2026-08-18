1. Gestione delle Rotazioni (Quaternioni)

L'uso dei quaternioni per gestire le rotazioni è un passaggio chiave nella conversione dei dati di cattura del movimento.
Esistono diverse strade possibili:

- pytransform3d:                   Una libreria Python per le trasformazioni geometriche 3D. È facile da usare e ben documentata, 
                                   ideale per chi ha poca esperienza in questo campo.

- qmt (Quaternion Motion Toolbox): Specifica per la biomeccanica, permette di calcolare gli angoli articolari secondo gli standard ISB. 
                                   È un'ottima scelta se la precisione scientifica è la priorità.

- scipy.spatial.transform:         Fa parte del pacchetto SciPy, quindi è già integrata nell'ecosistema Python più comune. 
                                   È molto veloce e affidabile per gestire grandi quantità di dati.

- Sviluppo di una libreria:        Opzione per creare una soluzione ad hoc. Poiché i giunti umani hanno gradi di libertà limitati, 
                                   è possibile ottimizzare i calcoli matriciali rendendo il codice più leggero e mirato.


2. Formati per l'Esportazione dei Dati

Per salvare ed esportare i dati, le opzioni principali sono:

- C3D (Coordinate 3D):            Standard di riferimento nella biomeccanica. Compatibile con quasi tutti i software professionali.
                                   Include metadati, etichette dei marker e frequenze di campionamento.

- JSON / CSV:                      Formati testuali semplici e compatibili con ogni sistema. Ideali per servizi web o strumenti 
                                   di visualizzazione moderni, ma meno efficienti su grandi volumi di dati.

- GLB (GL Binary) è il formato binario di glTF 2.0 - contiene geometrie 3D pre-costruite (mesh di cilindri per i segmenti corporei) pronte per il rendering senza elaborazione aggiuntiva.
Il vantaggio è l'interfacciamento diretto con Three.js: Three.js integra nativamente GLTFLoader, quindi il file GLB viene caricato con una singola chiamata GLTFLoader.load() e le mesh compaiono nella scena immediatamente. Il viewer HTML sfrutta questo meccanismo per visualizzare i segmenti senza ricostruire la geometria da zero.
Contemporaneamente si può creare un file JSON con tutti i dati per frame (start_points, end_points, quaternions) che alimenta l'animazione temporale: il viewer legge il JSON per iterare i frame e posizionare/ruotare le mesh GLB in tempo reale.


3. Strumenti per la Visualizzazione 3D

Per visualizzare i movimenti in 3D, si possono valutare tre soluzioni:

- pytransform3d:                   Permette di generare grafici direttamente in Python. Si integra bene con matplotlib e Jupyter, 
                                   ma non è la scelta migliore per animazioni fluide o in tempo reale.

- three.js:                        La soluzione ideale per il web. Include strumenti per gestire scheletri e animazioni. 
                                   Funziona nel browser senza installare nulla, ma richiede JavaScript o TypeScript.

- Rviz (ROS Visualization):        Lo strumento standard nel mondo della robotica. Potente per flussi di dati in tempo reale, 
                                   ma richiede l'infrastruttura ROS per funzionare correttamente.


