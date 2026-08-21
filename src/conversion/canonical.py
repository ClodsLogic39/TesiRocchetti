"""Scrittura e lettura del C3D canonico, l'unico file da cui si fa ingest.

`write` salva in un solo C3D i punti anatomici, gli angoli articolari, i
quaternioni con le derivate angolari e, in un gruppo proprietario BIOMECH, le
annotazioni che il formato non prevede: provenienza, copertura, assenze con il
motivo, cicli e convenzione adottata. `read` rilegge lo stesso file e ricostruisce
la struttura di partenza.

Struttura del file:
  POINT    i punti anatomici, poi un canale a tre componenti per giunto con gli
           angoli sui piani sagittale, frontale e trasverso
  ANALOG   quaternione, velocita' e accelerazione angolare di ogni segmento
  BIOMECH  le annotazioni

Prerequisiti: numpy ed ezc3d. `write` accetta il risultato di `process.analizza`,
che porta con se' i cicli; `read` accetta solo file che contengono il gruppo
BIOMECH con la voce CYCLE_METHOD.
"""
import numpy as np

from . import anatomy

# --- Costanti del contratto ---
# Versione del formato, descrizione della convenzione di terne e angoli, nota
# allegata agli angoli di caviglia e nomi delle componenti dei canali analogici.
VERSION = "1.0"
CONVENTION = "ISB Wu2002; terne Y-lungo/Z-destra/X=YxZ; Eulero ZXY Grood-Suntay"
NOTE_CAVIGLIA = ("angolo geometrico senza prova statica: la caviglia porta uno "
                 "scostamento costante proprio del soggetto, confrontare le "
                 "escursioni piuttosto che i valori assoluti")

_QUAT_AXES = ("x", "y", "z", "w")
_VEC_AXES = ("x", "y", "z")


def _txt(value):
    return str(value) if value not in (None, "") else "-"


# --- Scrittura ---
# Compone i canali del file e vi allega le annotazioni, poi lo salva su disco.
def write(risultato, path):
    import ezc3d

    if "cycles" not in risultato:
        raise ValueError("il risultato non porta i cicli: passare da "
                         "`process.analizza`, non da `angles.compute`")
    punti = risultato["points"]
    segmenti = risultato["segments"]
    angoli = risultato["angles"]
    meta = risultato["meta"]
    n = int(meta.get("n_frames") or 0)
    fps = float(meta.get("fps") or 0.0)

    # --- Canali POINT: prima i punti anatomici, poi gli angoli articolari ---
    # I fotogrammi non validi vengono marcati nel canale dei residui invece che
    # scritti come coordinate.
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
        P[3] = 1.0
        residui = np.zeros((1, len(dati), n))
        for i, (blocco, presenti) in enumerate(dati):
            valido = np.all(np.isfinite(blocco[:, presenti]), axis=1)
            P[:3, i, :] = np.where(np.isfinite(blocco), blocco, 0.0).T
            residui[0, i, :] = np.where(valido, 0.0, -1.0)
    else:
        P, residui = np.zeros((4, 0, n)), np.zeros((1, 0, n))

    # --- Canali ANALOG: quaternioni e derivate angolari di ogni segmento ---
    an_etichette, an_dati = [], []
    for seg in sorted(segmenti):
        s = segmenti[seg]
        for chiave, assi in (("quat", _QUAT_AXES), ("omega", _VEC_AXES),
                             ("alpha", _VEC_AXES)):
            v = np.asarray(s[chiave], dtype=float)
            for k, asse in enumerate(assi):
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

    # --- Gruppo BIOMECH: le annotazioni ---
    # Versione e convenzione, origine della sessione, elenco di punti, segmenti e
    # angoli presenti con la loro provenienza e copertura, assenze con il motivo,
    # cicli individuati e terna del laboratorio.
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


# --- Lettura ---
# Ricostruisce punti, angoli, segmenti, annotazioni e cicli da un file scritto da
# `write`. Un file privo del gruppo BIOMECH, o privo dei cicli, viene rifiutato.
def read(path):
    import ezc3d

    c = ezc3d.c3d(path)
    par = c["parameters"]
    b = par.get("BIOMECH")
    if not b:
        raise ValueError(f"{path} non e' un C3D canonico: manca il gruppo BIOMECH")
    if "CYCLE_METHOD" not in b:
        raise ValueError(
            f"{path} e' stato convertito prima che i cicli entrassero nel file. "
            "Va riconvertito: python -m src.conversion.process <sorgente>")

    def val(nome, default=None):
        if nome not in b:
            return default
        v = b[nome]["value"]
        if len(v) == 0:
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
