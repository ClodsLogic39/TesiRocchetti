"""Il C3D canonico: l'unico file da cui si fa ingest.

Contiene tutto cio' che serve a riempire il database, e niente che vada ricavato
altrove. Nei gruppi standard i dati; in un gruppo proprietario `BIOMECH` le
annotazioni che il formato non prevede -- provenienza, copertura, assenze con il
motivo, convenzione angolare. Un gruppo proprietario e' prassi consolidata nel
settore (Vicon usa `PROCESSING`, Visual3D usa `VISUAL3D`) e non impedisce ad
altri programmi di leggere il file: chi non lo conosce lo ignora.

Struttura:
  POINT   i sedici punti, poi un canale a tre componenti per giunto con gli
          angoli sui tre piani (e' la convenzione dei model output clinici)
  ANALOG  quaternione, velocita' e accelerazione angolare di ogni segmento
  BIOMECH le annotazioni
"""
import numpy as np

from . import anatomy

VERSION = "1.0"
CONVENTION = "ISB Wu2002; terne Y-lungo/Z-destra/X=YxZ; Eulero ZXY Grood-Suntay"
NOTE_CAVIGLIA = ("angolo geometrico senza prova statica: la caviglia porta uno "
                 "scostamento costante proprio del soggetto, confrontare le "
                 "escursioni piuttosto che i valori assoluti")

_QUAT_AXES = ("x", "y", "z", "w")
_VEC_AXES = ("x", "y", "z")


def _txt(value):
    """I parametri di testo del C3D non tollerano il vuoto."""
    return str(value) if value not in (None, "") else "-"


def write(risultato, path):
    """Scrive il C3D canonico. Restituisce il percorso."""
    import ezc3d

    if "cycles" not in risultato:
        # Senza questo controllo il file uscirebbe dichiarando "nessun ciclo",
        # che e' un esito legittimo: la dimenticanza si travestirebbe da misura.
        raise ValueError("il risultato non porta i cicli: passare da "
                         "`process.analizza`, non da `angles.compute`")
    punti = risultato["points"]
    segmenti = risultato["segments"]
    angoli = risultato["angles"]
    meta = risultato["meta"]
    n = int(meta.get("n_frames") or 0)
    fps = float(meta.get("fps") or 0.0)

    # --- canali POINT: punti anatomici, poi angoli articolari ---
    # Ogni voce e' (blocco n x 3, colonne che contengono davvero un dato). Un
    # canale angolare puo' avere piani assenti: quelle colonne non partecipano
    # al giudizio di validita', altrimenti l'assenza di un piano cancellerebbe
    # anche gli altri due.
    etichette, dati = [], []
    for nome in anatomy.POINTS:
        if nome not in punti:
            continue
        etichette.append(anatomy.point_label(nome))
        dati.append((np.asarray(punti[nome], dtype=float), [0, 1, 2]))

    giunti_scritti = []
    for giunto in anatomy.JOINTS:
        colonne = [angoli.get((giunto, p)) for p in anatomy.PLANES]
        if all(c is None for c in colonne):
            continue
        blocco = np.column_stack([
            np.full(n, np.nan) if c is None else np.asarray(c, dtype=float)
            for c in colonne])
        etichette.append(anatomy.angle_label(giunto, None))
        dati.append((blocco, [k for k, c in enumerate(colonne) if c is not None]))
        giunti_scritti.append(giunto)

    if dati:
        P = np.zeros((4, len(dati), n))
        P[3] = 1.0                     # coordinata omogenea, come vuole ezc3d
        residui = np.zeros((1, len(dati), n))
        for i, (blocco, presenti) in enumerate(dati):
            valido = np.all(np.isfinite(blocco[:, presenti]), axis=1)
            P[:3, i, :] = np.where(np.isfinite(blocco), blocco, 0.0).T
            # Il residuo e' il modo in cui il formato dichiara i buchi: -1 dove
            # il campione non c'e'. Sta in meta_points, non nella quarta riga
            # dei punti: quella ezc3d la usa per la coordinata omogenea.
            residui[0, i, :] = np.where(valido, 0.0, -1.0)
    else:
        P, residui = np.zeros((4, 0, n)), np.zeros((1, 0, n))

    # --- canali ANALOG: quaternioni e derivate angolari ---
    an_etichette, an_dati = [], []
    for seg in sorted(segmenti):
        s = segmenti[seg]
        for chiave, assi in (("quat", _QUAT_AXES), ("omega", _VEC_AXES),
                             ("alpha", _VEC_AXES)):
            v = np.asarray(s[chiave], dtype=float)
            for k, asse in enumerate(assi):  # noqa: B007
                an_etichette.append(f"{seg.upper()}_{chiave.upper()}_{asse.upper()}")
                an_dati.append(v[:, k])
    A = (np.asarray(an_dati, dtype=float).reshape(1, len(an_dati), n)
         if an_dati else np.zeros((1, 0, n)))

    c = ezc3d.c3d()
    c["parameters"]["POINT"]["RATE"]["value"] = [fps]
    c["parameters"]["POINT"]["LABELS"]["value"] = etichette
    c["parameters"]["POINT"]["UNITS"]["value"] = ["m"]
    c["data"]["points"] = P
    c["data"]["meta_points"]["residuals"] = residui
    if an_etichette:
        c["parameters"]["ANALOG"]["RATE"]["value"] = [fps]
        c["parameters"]["ANALOG"]["LABELS"]["value"] = an_etichette
        c["data"]["analogs"] = A

    # --- gruppo BIOMECH ---
    p = c.add_parameter
    p("BIOMECH", "VERSION", [VERSION], "versione del contratto di questo file")
    p("BIOMECH", "CONVENTION", [CONVENTION], "convenzione di terne e angoli")
    p("BIOMECH", "SOURCE", [_txt(meta.get("source"))], "sessione di origine")
    p("BIOMECH", "SOURCE_FORMAT", [_txt(meta.get("source_format"))])
    p("BIOMECH", "CAPTURE_SYSTEM", [_txt(meta.get("capture_system"))])
    for chiave in ("sport", "view", "coordinate_system"):
        if meta.get(chiave):
            p("BIOMECH", chiave.upper(), [_txt(meta[chiave])])

    presenti = [nome for nome in anatomy.POINTS if nome in punti]
    p("BIOMECH", "POINTS", presenti, "punti anatomici presenti")
    p("BIOMECH", "POINT_PROVENANCE",
      [risultato["point_provenance"].get(k, anatomy.DERIVED) for k in presenti],
      "come e' stato ottenuto ogni punto")
    p("BIOMECH", "POINT_COVERAGE",
      [float(risultato["coverage"].get(k, 0.0)) for k in presenti],
      "frazione di frame in cui il punto e' valido")

    segs = sorted(segmenti)
    p("BIOMECH", "SEGMENTS", segs, "segmenti con terna costruita")
    p("BIOMECH", "SEGMENT_PROVENANCE", [segmenti[s]["provenance"] for s in segs])

    coppie = [(g, pi) for g in giunti_scritti for pi in anatomy.PLANES
              if (g, pi) in angoli]
    p("BIOMECH", "ANGLES", [f"{g}:{pi}" for g, pi in coppie],
      "giunto e piano di ogni angolo presente")
    p("BIOMECH", "ANGLE_PROVENANCE",
      [risultato["angle_provenance"].get(k, anatomy.DERIVED) for k in coppie],
      "measured = gia' nella sorgente; derived = calcolato; "
      "modelled = dipende da un'assunzione")
    p("BIOMECH", "ANGLE_IMPLAUSIBLE_RATE",
      [float(risultato["implausible"].get(k) or 0.0) for k in coppie],
      "quota di variazioni oltre il limite fisiologico")
    p("BIOMECH", "ANGLE_COMPONENT_ORDER", list(anatomy.ANGLE_COMPONENT_ORDER),
      "a quale piano corrisponde ciascuna delle tre componenti")

    assenze = risultato["absences"]
    chiavi = sorted(assenze)
    p("BIOMECH", "ABSENT", chiavi or ["-"], "cio' che manca")
    p("BIOMECH", "ABSENT_REASON",
      [assenze[k][0] for k in chiavi] or ["-"], "perche' manca")
    p("BIOMECH", "ABSENT_DETAIL",
      [_txt(assenze[k][1]) for k in chiavi] or ["-"], "dettaglio del motivo")

    # I cicli. Stanno nel file perche' chi lo carica non deve ricavarli: la
    # segmentazione dipende dallo sport e da soglie che cambiano nel tempo, e se
    # girasse al caricamento lo stesso file potrebbe dare cicli diversi a due
    # letture. Qui e' deciso una volta, e resta scritto.
    cicli = risultato.get("cycles") or {}
    intervalli = cicli.get("frames") or []
    p("BIOMECH", "CYCLE_METHOD", [_txt(cicli.get("method"))],
      "strategia con cui i cicli sono stati individuati")
    p("BIOMECH", "CYCLE_SIDE", [_txt(s) for s in cicli.get("sides") or []] or ["-"],
      "lato di ogni ciclo: le falcate dei due piedi convivono nella stessa prova")
    p("BIOMECH", "CYCLE_START", [int(s) for s, _ in intervalli] or [-1],
      "primo frame di ogni ciclo")
    p("BIOMECH", "CYCLE_END", [int(e) for _, e in intervalli] or [-1],
      "frame che chiude ogni ciclo, escluso: apre il successivo")
    p("BIOMECH", "CYCLE_WARNINGS", cicli.get("warnings") or ["-"],
      "cio' che la segmentazione ha dovuto concedere")

    p("BIOMECH", "WARNINGS", risultato.get("warnings") or ["-"])
    p("BIOMECH", "NOTE_ANKLE", [NOTE_CAVIGLIA])
    p("BIOMECH", "WORLD_FRAME",
      np.asarray(risultato["world_frame"], dtype=float).ravel().tolist(),
      "terna del laboratorio, per righe")

    c.write(path)
    return path


def read(path):
    """Rilegge il C3D canonico. Inverso esatto di `write`, per l'ingest."""
    import ezc3d

    c = ezc3d.c3d(path)
    par = c["parameters"]
    b = par.get("BIOMECH")
    if not b:
        raise ValueError(f"{path} non e' un C3D canonico: manca il gruppo BIOMECH")
    if "CYCLE_METHOD" not in b:
        # Distinzione che conta: "nessun ciclo trovato" e' un esito, "cicli mai
        # cercati" e' un file incompleto. Senza questo controllo il secondo si
        # travestirebbe da primo e il database si riempirebbe di sessioni mute.
        raise ValueError(
            f"{path} e' stato convertito prima che i cicli entrassero nel file. "
            "Va riconvertito: python -m src.conversion.process <sorgente>")

    def val(nome, default=None):
        if nome not in b:
            return default
        v = b[nome]["value"]
        if len(v) == 0:                  # parametro dichiarato ma senza valori
            return default if default is not None else []
        return [str(x).strip() for x in v] if isinstance(v[0], str) else list(v)

    etichette = [str(x).strip() for x in par["POINT"]["LABELS"]["value"]]
    P = c["data"]["points"]
    residui = c["data"].get("meta_points", {}).get("residuals")
    valido = (residui[0] >= 0 if residui is not None
              else np.ones(P.shape[1:], dtype=bool))
    canali = {}
    for i, lb in enumerate(etichette):
        xyz = P[:3, i, :].T.copy()
        xyz[~valido[i]] = np.nan
        canali[lb] = xyz

    punti = {nome: canali[anatomy.point_label(nome)]
             for nome in val("POINTS", []) if anatomy.point_label(nome) in canali}
    angoli = {}
    for chiave in val("ANGLES", []):
        giunto, piano = chiave.split(":")
        etichetta = anatomy.angle_label(giunto, piano)
        if etichetta in canali:
            angoli[(giunto, piano)] = canali[etichetta][:, anatomy.PLANES.index(piano)]

    segmenti = {}
    if "ANALOG" in par and par["ANALOG"]["LABELS"]["value"]:
        an = [str(x).strip() for x in par["ANALOG"]["LABELS"]["value"]]
        A = c["data"]["analogs"][0]
        indice = {lb: A[i] for i, lb in enumerate(an) if i < A.shape[0]}
        for seg in val("SEGMENTS", []):
            voce = {}
            for chiave, assi in (("quat", _QUAT_AXES), ("omega", _VEC_AXES),
                                 ("alpha", _VEC_AXES)):
                colonne = [indice.get(f"{seg.upper()}_{chiave.upper()}_{a.upper()}")
                           for a in assi]
                if all(col is not None for col in colonne):
                    voce[chiave] = np.column_stack(colonne)
            if voce:
                segmenti[seg] = voce

    chiavi = val("ABSENT", [])
    motivi, dettagli = val("ABSENT_REASON", []), val("ABSENT_DETAIL", [])
    assenze = {k: (motivi[i] if i < len(motivi) else "",
                   dettagli[i] if i < len(dettagli) else "")
               for i, k in enumerate(chiavi) if k != "-"}

    prov_seg = val("SEGMENT_PROVENANCE", [])
    for i, seg in enumerate(val("SEGMENTS", [])):
        if seg in segmenti and i < len(prov_seg):
            segmenti[seg]["provenance"] = prov_seg[i]

    coppie = [tuple(k.split(":")) for k in val("ANGLES", [])]
    prov_ang = val("ANGLE_PROVENANCE", [])
    impl = val("ANGLE_IMPLAUSIBLE_RATE", [])
    presenti = val("POINTS", [])
    prov_pt = val("POINT_PROVENANCE", [])
    cop = val("POINT_COVERAGE", [])

    metodo = (val("CYCLE_METHOD", ["-"]) or ["-"])[0]
    lati = val("CYCLE_SIDE", [])
    inizi, fini = val("CYCLE_START", []), val("CYCLE_END", [])
    # Il segnaposto -1 delle sessioni senza cicli cade da solo: nessun intervallo
    # valido finisce prima di dove comincia. I lati seguono gli stessi indici.
    tenuti = [i for i, (s, e) in enumerate(zip(inizi, fini)) if int(e) > int(s)]
    cicli = {
        "method": None if metodo == "-" else metodo,
        "frames": [[int(inizi[i]), int(fini[i])] for i in tenuti],
        "sides": [None if i >= len(lati) or lati[i] == "-" else lati[i]
                  for i in tenuti],
        "warnings": [w for w in val("CYCLE_WARNINGS", []) if w != "-"],
    }

    return {
        "points": punti,
        "cycles": cicli,
        "point_provenance": {k: prov_pt[i] for i, k in enumerate(presenti)
                             if i < len(prov_pt)},
        "coverage": {k: float(cop[i]) for i, k in enumerate(presenti) if i < len(cop)},
        "segments": segmenti,
        "angles": angoli,
        "angle_provenance": {c2: prov_ang[i] for i, c2 in enumerate(coppie)
                             if i < len(prov_ang)},
        "implausible": {c2: float(impl[i]) for i, c2 in enumerate(coppie)
                        if i < len(impl)},
        "absences": assenze,
        "warnings": [w for w in val("WARNINGS", []) if w != "-"],
        "world_frame": np.asarray(val("WORLD_FRAME", [1, 0, 0, 0, 1, 0, 0, 0, 1]),
                                  dtype=float).reshape(3, 3).tolist(),
        "meta": {
            "source": (val("SOURCE", ["-"]) or ["-"])[0],
            "source_format": (val("SOURCE_FORMAT", ["-"]) or ["-"])[0],
            "capture_system": (val("CAPTURE_SYSTEM", ["-"]) or ["-"])[0],
            "sport": (val("SPORT", []) or [None])[0],
            "view": (val("VIEW", []) or [None])[0],
            "convention": (val("CONVENTION", ["-"]) or ["-"])[0],
            "biomech_version": (val("VERSION", ["-"]) or ["-"])[0],
            "fps": float(par["POINT"]["RATE"]["value"][0]) or None,
            "n_frames": int(P.shape[2]),
        },
    }
