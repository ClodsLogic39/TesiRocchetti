"""Viewer Open3D: figura umana a volumi dal C3D canonico.

Ogni segmento e' un tronco di cono a sezione ellittica fra due punti canonici,
con il raggio preso al PUNTO e non al segmento: la coscia e' grossa all'anca e
sottile al ginocchio, e un cilindro di raggio unico non e' una gamba. Il busto e'
un volume unico dal bacino alle spalle, la testa un ellissoide orientato lungo il
collo. Gli spessori vengono dalle circonferenze antropometriche medie, riferite
alla lunghezza del tronco del soggetto: la figura resta proporzionata a chi e'
stato ripreso, non a una costante.

Nulla e' inventato: i volumi poggiano sugli stessi punti che la conversione ha
scritto. La composizione dei segmenti e i colori vengono da `anatomy`; punti,
angoli e quaternioni da `canonical.read`, che e' l'inverso esatto di quanto
scrive il convertitore. Il viewer non interpreta il C3D per conto suo.

Avvio: python -m src.visualization.viewer_open3d --c3d <file.c3d> [--coord]
"""
import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import open3d as o3d
    from open3d.visualization import gui, rendering
except ImportError:  # il rendering vive in un ambiente separato
    o3d = None

from ..conversion import anatomy, canonical

GRIGIO = (0.5, 0.5, 0.5)

# Il busto non e' un segmento di anatomy.SEGMENTS -- gli estremi del segmento
# "trunk" sono hip_c e shoulder_c, centri interni mai scritti come punti --
# quindi non ha un colore in anatomy.colours(). Sta qui, vicino a dove si disegna.
COLORE_BUSTO = (0.72, 0.68, 0.42)
COLORE_TESTA = (0.86, 0.72, 0.60)

# Semispessore del corpo a ogni giunto, in frazioni della lunghezza del tronco.
# Il raggio appartiene al PUNTO, non al segmento: la coscia e' grossa all'anca e
# sottile al ginocchio, e un cilindro di raggio unico non e' una gamba. Cosi' la
# rastremazione viene da se' e non dipende dall'ordine in cui `anatomy` elenca
# gli estremi, che per gli arti mette il distale per primo.
# I valori vengono dalle circonferenze medie dell'adulto riportate in
# antropometria (coscia ~55 cm, polpaccio ~36, caviglia ~22, braccio ~30,
# polso ~17), divise per un tronco di 47 cm: e' cosi' che una figura risulta
# proporzionata invece che gonfia.
_RAGGI = {"hip": 0.19, "knee": 0.13, "ankle": 0.075, "heel": 0.075, "toe": 0.055,
          "shoulder": 0.11, "elbow": 0.085, "wrist": 0.058}
# Il segmento del bacino sta dentro il busto, che lo disegna gia': tenerlo allo
# spessore dell'anca lo farebbe sporgere ai lati come una palla.
_RAGGIO_BACINO = 0.09

# Segmenti che il ciclo delle ossa non deve disegnare.
#
# `neck` va da shoulder_c al punto `head`, che NON e' il centro del capo ma la
# fronte: disegnarlo come un osso produceva un tubo che sbucava davanti alla
# faccia. Il collo lo costruisce `costruisci`, che sa dove finisce la testa.
#
# `clavicle_l` e `clavicle_r` hanno gli stessi due estremi -- esistono separati
# perche' ogni spalla vuole la propria terna, non perche' siano due oggetti --
# quindi disegnarli entrambi sovrappone due volumi identici, che nel rendering
# lampeggiano l'uno sull'altro.
_NON_DISEGNATI = frozenset({"neck", "clavicle_l"})
_RAGGIO_DEFAULT = 0.08

# Il corpo umano e' piu' largo che profondo: schiacciamento delle sezioni.
_PROFONDITA_ARTO = 0.85
_PROFONDITA_BUSTO = 0.62
# I punti del bacino e delle spalle sono CENTRI ARTICOLARI, che stanno dentro il
# corpo: la loro distanza non e' la larghezza del busto. I due fattori la portano
# alla misura esterna -- molto per il bacino, dove le anche sono profonde, poco
# per le spalle, dove l'articolazione e' quasi sul bordo.
_LARGHEZZA_BACINO = 1.60
_LARGHEZZA_SPALLE = 1.05
# Semiassi della testa (largo, profondo, alto), sempre in frazioni del tronco.
_TESTA = (0.15, 0.19, 0.22)


# --- Dati (nessuna dipendenza da Open3D: e' la parte verificabile) ---
def carica(percorso):
    """Legge il C3D canonico e prepara cio' che serve a disegnare.

    Un segmento entra in `ossa` solo se entrambi i suoi estremi sono fra i punti
    del file: i dati sono spesso parziali (una ripresa che non inquadra le gambe
    non ha le caviglie) e un segmento incompleto va saltato, non e' un errore.
    """
    dati = canonical.read(percorso)
    punti = {nome: np.asarray(v, dtype=float) for nome, v in dati["points"].items()}
    colori = anatomy.colours()

    ossa = []
    for segmento in anatomy.SEGMENTS:
        # Gli estremi possono avere alternative (il piede usa il tallone se
        # c'e', altrimenti la caviglia): si disegna la prima disponibile.
        coppia = next((c for c in anatomy.segment_ends_options(segmento)
                       if all(k in punti for k in c)), None)
        if coppia:
            prossimale, distale = coppia
            ossa.append((segmento, prossimale, distale,
                         tuple(colori.get(segmento, GRIGIO))))

    # Il busto si disegna come volume unico dai quattro punti che lo delimitano
    # ai lati -- nessuna ricostruzione locale, solo punti che la conversione ha
    # gia' scritto. Qui basta sapere se quei punti ci sono: una ripresa senza
    # spalle non ha un tronco da mostrare, e la legenda non deve nominarlo.
    ha_busto = all(k in punti for k in ("hip_l", "hip_r", "shoulder_l", "shoulder_r"))

    # Il giunto che ogni segmento distale rappresenta, per la legenda.
    giunto_di = {spec["distal"]: giunto for giunto, spec in anatomy.JOINTS.items()}
    angoli = {}
    for segmento, _, _, _ in ossa:
        giunto = giunto_di.get(segmento)
        serie = dati["angles"].get((giunto, "sagittal")) if giunto else None
        if serie is not None:
            angoli[segmento] = (giunto, np.asarray(serie, dtype=float))
    # Il tronco ha due giunti con lo stesso segmento distale, "trunk" (vs
    # mondo) e "spine" (vs bacino): la mappa sopra, indicizzata per segmento,
    # ne terrebbe uno solo. Si aggiungono qui entrambi, con chiavi separate
    # che non collidono con nessun nome di segmento. "spine" mostra il piano
    # trasverso e non il sagittale come gli altri: il bacino prende il suo
    # rollio dalla stessa direzione che definisce l'asse del tronco, quindi
    # la flessione relativa e' zero per costruzione (vedi
    # anatomy.plane_provenance) e il piano davvero osservato -- e quello che
    # mostra la torsione -- e' il trasverso.
    for giunto, piano in (("trunk", "sagittal"), ("spine", "transverse")):
        serie = dati["angles"].get((giunto, piano))
        if serie is not None:
            angoli[giunto] = (giunto, np.asarray(serie, dtype=float))

    terne = {segmento: np.asarray(v["quat"], dtype=float)
             for segmento, v in dati["segments"].items() if "quat" in v}

    return {
        "punti": punti,
        "ossa": ossa,
        "ha_busto": ha_busto,
        # Grandezza del soggetto: tutti gli spessori del manichino vi si
        # riferiscono, cosi' la figura resta proporzionata a chi e' stato ripreso.
        "scala": _scala_corporea(punti),
        "angoli": angoli,
        "terne": terne,
        # righe: 0 anteriore, 1 alto, 2 destra. Serve a orientare la camera.
        "terna_mondo": dati["world_frame"],
        "fps": float(dati["meta"]["fps"] or 30.0),
        "n_frame": int(dati["meta"]["n_frames"]),
        "sorgente": dati["meta"]["source"],
    }


# --- Misure del corpo ---
def _centro(punti, sinistra, destra):
    """Punto medio fra due punti simmetrici, o None se ne manca uno."""
    if sinistra not in punti or destra not in punti:
        return None
    return (punti[sinistra] + punti[destra]) / 2.0


def _scala_corporea(punti):
    """Lunghezza del tronco: la misura a cui si riferiscono tutti gli spessori.

    Serve una grandezza del soggetto, non una costante: lo stesso manichino deve
    andare bene per un adulto e per un bambino. Se il tronco non c'e' -- una
    ripresa che taglia sotto le spalle -- si ripiega sulla larghezza delle
    spalle, che nell'adulto vale circa due terzi del tronco.
    """
    anche, spalle = _centro(punti, "hip_l", "hip_r"), _centro(punti, "shoulder_l", "shoulder_r")
    if anche is not None and spalle is not None:
        d = np.linalg.norm(spalle - anche, axis=1)
        d = d[np.isfinite(d)]
        if d.size and np.median(d) > 1e-3:
            return float(np.median(d))
    if "shoulder_l" in punti and "shoulder_r" in punti:
        d = np.linalg.norm(punti["shoulder_r"] - punti["shoulder_l"], axis=1)
        d = d[np.isfinite(d)]
        if d.size and np.median(d) > 1e-3:
            return float(np.median(d)) / 0.66
    return 0.5


def _raggio(punto, scala):
    """Semispessore del corpo in quel punto. Il lato non cambia lo spessore."""
    nome = punto.rsplit("_", 1)[0] if punto[-2:] in ("_l", "_r") else punto
    return _RAGGI.get(nome, _RAGGIO_DEFAULT) * scala


# --- Primitive ---
def _base_ortonormale(asse, destra=None):
    """Due assi perpendicolari all'asse dato. `destra` orienta la sezione."""
    u = asse / np.linalg.norm(asse)
    if destra is not None:
        e1 = destra - float(destra @ u) * u
        if np.linalg.norm(e1) < 1e-6:
            e1 = None
        else:
            e1 = e1 / np.linalg.norm(e1)
    else:
        e1 = None
    if e1 is None:
        appoggio = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(u, appoggio)
        e1 = e1 / np.linalg.norm(e1)
    return u, e1, np.cross(u, e1)


def _tronco_di_cono(a, b, ra, rb, colore, destra=None, profondita=1.0, lati=20):
    """Volume fra due punti, con raggio proprio a ciascuno e sezione ellittica.

    E' la primitiva che sostituisce il cilindro: un arto ha due spessori, e la
    sezione di un corpo umano non e' un cerchio.
    """
    asse = b - a
    lunghezza = float(np.linalg.norm(asse))
    if lunghezza < 1e-6:
        return None
    u, e1, e2 = _base_ortonormale(asse, destra)
    ang = np.linspace(0.0, 2.0 * np.pi, lati, endpoint=False)
    anello = np.cos(ang)[:, None] * e1 + profondita * np.sin(ang)[:, None] * e2
    vertici = np.vstack([a + ra * anello, b + rb * anello, a, b])
    facce = []
    for i in range(lati):
        j = (i + 1) % lati
        facce += [[i, j, lati + i], [j, lati + j, lati + i],
                  [2 * lati, j, i], [2 * lati + 1, lati + i, lati + j]]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertici),
                                  o3d.utility.Vector3iVector(np.asarray(facce, dtype=np.int32)))
    m.paint_uniform_color(colore)
    m.compute_vertex_normals()
    return m


def _ellissoide(centro, semiassi, colore, alto=None, destra=None, risoluzione=20):
    """Sfera schiacciata su tre assi. Serve alla testa, che ovale lo e' davvero."""
    m = o3d.geometry.TriangleMesh.create_sphere(1.0, resolution=risoluzione)
    v = np.asarray(m.vertices) * np.asarray(semiassi, dtype=float)
    if alto is not None:
        u, e1, e2 = _base_ortonormale(alto, destra)
        # colonne: largo (destra), profondo (avanti), alto (asse del collo)
        v = v @ np.vstack([e1, e2, u])
    m.vertices = o3d.utility.Vector3dVector(v + centro)
    m.paint_uniform_color(colore)
    m.compute_vertex_normals()
    return m


def _sfera(centro, raggio, colore):
    m = o3d.geometry.TriangleMesh.create_sphere(raggio, resolution=12)
    m.translate(centro)
    m.paint_uniform_color(colore)
    m.compute_vertex_normals()
    return m


def _terna(quaternione, centro, dimensione):
    """Assi del segmento. Il quaternione e' (x, y, z, w), come vuole scipy."""
    if not np.all(np.isfinite(quaternione)):
        return None
    m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=dimensione)
    m.rotate(Rotation.from_quat(quaternione).as_matrix(), center=(0, 0, 0))
    m.translate(centro)
    return m


def costruisci(dati, frame, con_terne=False):
    """Geometrie di un frame: arti rastremati, busto pieno, testa, terne."""
    scala = dati["scala"]
    punti = dati["punti"]

    def posa(nome):
        v = punti[nome][frame] if nome in punti else None
        return v if v is not None and np.all(np.isfinite(v)) else None

    def centro(sinistra, destra):
        a, b = posa(sinistra), posa(destra)
        return None if a is None or b is None else (a + b) / 2.0

    anche, spalle = centro("hip_l", "hip_r"), centro("shoulder_l", "shoulder_r")
    # Asse destro del soggetto: orienta la sezione ellittica di tutto il corpo.
    # Senza, un busto schiacciato sarebbe schiacciato in una direzione a caso.
    destra_rif = None
    for a, b in (("hip_l", "hip_r"), ("shoulder_l", "shoulder_r")):
        pa, pb = posa(a), posa(b)
        if pa is not None and pb is not None and np.linalg.norm(pb - pa) > 1e-6:
            destra_rif = (pb - pa) / np.linalg.norm(pb - pa)
            break

    geometrie = []
    for segmento, primo, secondo, colore in dati["ossa"]:
        if segmento in _NON_DISEGNATI:
            continue
        a, b = posa(primo), posa(secondo)
        if a is None or b is None:
            continue
        ra, rb = _raggio(primo, scala), _raggio(secondo, scala)
        if segmento == "pelvis":
            ra = rb = _RAGGIO_BACINO * scala
        corpo = _tronco_di_cono(a, b, ra, rb, colore, destra_rif, _PROFONDITA_ARTO)
        if corpo is None:
            continue
        # Le sfere ai giunti raccordano due tronchi di raggio diverso: senza,
        # fra coscia e gamba resterebbe uno scalino visibile.
        geometrie += [corpo, _sfera(a, ra * 0.95, colore), _sfera(b, rb * 0.95, colore)]
        if con_terne and segmento in dati["terne"]:
            assi = _terna(dati["terne"][segmento][frame], (a + b) / 2,
                          max(scala * 0.25, 0.08))
            if assi is not None:
                geometrie.append(assi)

    # Busto: un volume unico dal bacino alle spalle, non due linee laterali.
    # Si appoggia agli stessi quattro punti reali di prima -- niente e' inventato
    # -- ma la larghezza di ogni sezione e' quella misurata a quell'altezza, e la
    # profondita' segue la proporzione umana.
    if anche is not None and spalle is not None:
        larghezza_anche = float(np.linalg.norm(
            posa("hip_r") - posa("hip_l"))) / 2.0 * _LARGHEZZA_BACINO
        larghezza_spalle = float(np.linalg.norm(
            posa("shoulder_r") - posa("shoulder_l"))) / 2.0 * _LARGHEZZA_SPALLE
        busto = _tronco_di_cono(anche, spalle,
                                max(larghezza_anche, 0.05 * scala),
                                max(larghezza_spalle, 0.05 * scala),
                                COLORE_BUSTO, destra_rif, _PROFONDITA_BUSTO)
        if busto is not None:
            geometrie.append(busto)
        if "trunk" in dati["terne"]:
            assi = _terna(dati["terne"]["trunk"][frame], (anche + spalle) / 2.0,
                          max(scala * 0.35, 0.15))
            if assi is not None:
                geometrie.append(assi)

    # Testa e collo. La testa e' un ellissoide orientato lungo il collo: una
    # sfera darebbe una palla, e il piano su cui il soggetto guarda non si
    # vedrebbe. Se le spalle mancano, l'asse ripiega sulla verticale del
    # laboratorio, che e' comunque dichiarata nel file.
    riferimento = posa("head")
    if riferimento is not None:
        base = spalle if spalle is not None else posa("shoulder_c")
        alto = (testa_asse := riferimento - base) if (
            base is not None and np.linalg.norm(riferimento - base) > 1e-6) else None
        if alto is None:
            alto = np.asarray(dati["terna_mondo"], dtype=float)[1]
        alto = alto / np.linalg.norm(alto)
        semiassi = np.asarray([s * scala for s in _TESTA])

        # Il punto `head` non e' il centro del capo: viene dai marker frontali
        # (LFHD/RFHD, o il naso in MediaPipe), che stanno sulla superficie
        # davanti. Centrarci l'ellissoide sposta tutta la testa in avanti di
        # mezzo capo. Si arretra lungo l'asse anteriore, cosi' e' la faccia a
        # passare per quel punto -- che e' dove il marker era davvero.
        avanti = np.asarray(dati["terna_mondo"], dtype=float)[0]
        avanti = avanti - float(avanti @ alto) * alto
        norma = np.linalg.norm(avanti)
        centro_testa = (riferimento - avanti / norma * semiassi[1] if norma > 1e-6
                        else riferimento)

        # Il collo si ferma alla base del capo. Portandolo fino al centro
        # entrerebbe nella testa e ne spunterebbe intorno, appiattendola.
        if base is not None:
            attacco = centro_testa - alto * semiassi[2] * 0.85
            if float((attacco - base) @ alto) > 1e-3:
                collo = _tronco_di_cono(base, attacco, 0.13 * scala, 0.10 * scala,
                                        COLORE_TESTA, destra_rif, 0.9)
                if collo is not None:
                    geometrie.append(collo)
        geometrie.append(
            _ellissoide(centro_testa, semiassi, COLORE_TESTA, alto, destra_rif))
    return geometrie


def inquadratura(dati):
    """(centro, occhio, verticale) per una vista sagittale, piu' il raggio.

    Guardare dal lato destro del soggetto e' la vista che serve in biomeccanica:
    flessione ed estensione stanno tutte sul piano sagittale. Lasciare
    l'orientamento di default significa ricevere un angolo arbitrario, che nella
    finestra si corregge ruotando ma in un'immagine ferma no.

    I canali angolari non stanno in `punti`, quindi non entrano nel calcolo:
    e' cio' che sballava l'inquadratura quando il viewer leggeva tutti i canali
    POINT del C3D.
    """
    finiti = [p[np.all(np.isfinite(p), axis=1)] for p in dati["punti"].values()]
    finiti = [p for p in finiti if len(p)]
    if not finiti:
        raise ValueError("nessun punto valido: non c'e' niente da inquadrare")
    nuvola = np.concatenate(finiti)
    centro = nuvola.mean(0)
    raggio = float(np.max(nuvola.max(0) - nuvola.min(0))) / 2 or 1.0
    terna = np.asarray(dati["terna_mondo"], dtype=float)
    destra, su = terna[2], terna[1]
    return centro, centro + destra * raggio * 3.0, su, raggio


# --- Finestra ---
def esegui(dati, con_terne=False):
    n = dati["n_frame"]
    fps = dati["fps"]

    gui.Application.instance.initialize()
    finestra = gui.Application.instance.create_window(
        f"Biomech Viewer - {dati['sorgente']}", 1400, 820)
    vista = gui.SceneWidget()
    vista.scene = rendering.Open3DScene(finestra.renderer)
    vista.scene.set_background([0.09, 0.10, 0.12, 1.0])
    materiale = rendering.MaterialRecord()
    materiale.shader = "defaultLit"

    centro, occhio, su, raggio = inquadratura(dati)
    vista.setup_camera(60.0, o3d.geometry.AxisAlignedBoundingBox(
        centro - raggio * 1.6, centro + raggio * 1.6), centro)
    vista.look_at(centro, occhio, su)

    pannello = gui.ScrollableVert(4, gui.Margins(8, 8, 8, 8))
    pannello.add_child(gui.Label("Legenda / Angoli (sagittale; colonna: trasverso)"))
    righe = []  # (chiave in dati["angoli"], giunto, suffisso, etichetta)
    for segmento, _, _, colore in dati["ossa"]:
        quadratino = gui.Label("■")
        quadratino.text_color = gui.Color(*colore)
        riga = gui.Horiz(6)
        riga.add_child(quadratino)
        riga.add_child(gui.Label(segmento))
        pannello.add_child(riga)
        if segmento in dati["angoli"]:
            giunto, _ = dati["angoli"][segmento]
            etichetta = gui.Label(f"   {giunto}: --")
            pannello.add_child(etichetta)
            righe.append((segmento, giunto, "", etichetta))

    # Il busto non e' in `ossa`, ma i suoi due angoli reali (vs mondo, vs
    # bacino) vanno comunque in legenda. "spine" e' in trasverso, non
    # sagittale come tutte le altre righe (vedi il commento in `carica`).
    if dati["ha_busto"]:
        quadratino = gui.Label("■")
        quadratino.text_color = gui.Color(*COLORE_BUSTO)
        riga = gui.Horiz(6)
        riga.add_child(quadratino)
        riga.add_child(gui.Label("tronco"))
        pannello.add_child(riga)
        for chiave, suffisso in (("trunk", ""), ("spine", " (trasv.)")):
            if chiave in dati["angoli"]:
                giunto, _ = dati["angoli"][chiave]
                etichetta = gui.Label(f"   {giunto}{suffisso}: --")
                pannello.add_child(etichetta)
                righe.append((chiave, giunto, suffisso, etichetta))

    info = gui.Label("Frame 0")
    pannello.add_child(info)
    finestra.add_child(vista)
    finestra.add_child(pannello)

    def disponi(ctx):
        r = finestra.content_rect
        larghezza = 320
        vista.frame = gui.Rect(r.x, r.y, r.width - larghezza, r.height)
        pannello.frame = gui.Rect(r.x + r.width - larghezza, r.y, larghezza, r.height)
    finestra.set_on_layout(disponi)

    stato = {"frame": 0, "play": False, "nomi": [], "t": time.time()}

    def aggiorna(frame):
        for nome in stato["nomi"]:
            vista.scene.remove_geometry(nome)
        stato["nomi"] = []
        for i, g in enumerate(costruisci(dati, frame, con_terne)):
            nome = f"g{i}"
            vista.scene.add_geometry(nome, g, materiale)
            stato["nomi"].append(nome)
        info.text = f"Frame {frame}/{n - 1}  t={frame / fps:.2f}s"
        for segmento, giunto, suffisso, etichetta in righe:
            serie = dati["angoli"][segmento][1]
            valore = serie[frame] if frame < len(serie) else np.nan
            testo = "--" if not np.isfinite(valore) else f"{valore:.1f}"
            etichetta.text = f"   {giunto}{suffisso}: {testo}"

    aggiorna(0)

    def al_tasto(e):
        if e.type == gui.KeyEvent.UP:
            if e.key == ord(" "):
                stato["play"] = not stato["play"]
            elif e.key in (ord("q"), ord("Q")):
                gui.Application.instance.quit()
            elif e.key == gui.KeyName.RIGHT:
                stato["frame"] = min(n - 1, stato["frame"] + 1)
                aggiorna(stato["frame"])
            elif e.key == gui.KeyName.LEFT:
                stato["frame"] = max(0, stato["frame"] - 1)
                aggiorna(stato["frame"])
        return gui.Widget.EventCallbackResult.CONSUMED
    vista.set_on_key(al_tasto)

    def al_tick():
        if stato["play"] and time.time() - stato["t"] >= 1.0 / max(fps, 1.0):
            stato["t"] = time.time()
            stato["frame"] = (stato["frame"] + 1) % n
            aggiorna(stato["frame"])
        return True
    finestra.set_on_tick_event(al_tick)
    gui.Application.instance.run()


def istantanea(dati, frame, destinazione, con_terne=False, larghezza=900, altezza=700):
    """Rende un fotogramma su file PNG, senza aprire finestre.

    Serve a restituire l'immagine come dato: un server che non deve (o non puo')
    aprire una finestra sulla macchina che lo ospita puo' comunque mostrare la
    posa. Stessa geometria della finestra interattiva, stesse funzioni.
    """
    if not 0 <= frame < dati["n_frame"]:
        raise ValueError(f"frame {frame} fuori intervallo (0..{dati['n_frame'] - 1})")
    geometrie = costruisci(dati, frame, con_terne)
    if not geometrie:
        raise ValueError(f"frame {frame}: nessun segmento con entrambi gli estremi noti")

    # Due strade, perche' nessuna copre entrambi i sistemi. OffscreenRenderer
    # vuole EGL: e' la via giusta su un server Linux senza schermo, ma su macOS
    # solleva "EGL Headless is not supported". Il Visualizer classico con
    # finestra nascosta funziona su macOS e non richiede EGL.
    vista = inquadratura(dati)
    try:
        render = rendering.OffscreenRenderer(larghezza, altezza)
    except RuntimeError:
        return _istantanea_finestra_nascosta(geometrie, destinazione,
                                             larghezza, altezza, vista)

    render.scene.set_background([0.09, 0.10, 0.12, 1.0])
    materiale = rendering.MaterialRecord()
    materiale.shader = "defaultLit"
    for i, g in enumerate(geometrie):
        render.scene.add_geometry(f"g{i}", g, materiale)
    centro, occhio, su, raggio = vista
    render.setup_camera(60.0, o3d.geometry.AxisAlignedBoundingBox(
        centro - raggio * 1.6, centro + raggio * 1.6), centro)
    render.scene.camera.look_at(centro, occhio, su)
    o3d.io.write_image(destinazione, render.render_to_image())
    return destinazione


def _istantanea_finestra_nascosta(geometrie, destinazione, larghezza, altezza, vista):
    """Ripiego senza EGL: finestra mai mostrata, catturata e chiusa subito.

    Questa strada usa il visualizzatore classico, che ha un'altra API per la
    camera: `front` e' la direzione da cui si guarda, non il punto in cui sta
    l'occhio. Senza impostarla, la figura esce da un angolo arbitrario e
    lontana, e in un'immagine ferma non c'e' modo di correggerla ruotando.
    """
    centro, occhio, su, _ = vista
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(width=larghezza, height=altezza, visible=False):
        raise RuntimeError("impossibile creare il contesto di rendering")
    try:
        opt = vis.get_render_option()
        opt.background_color = np.array([0.09, 0.10, 0.12])
        for g in geometrie:
            vis.add_geometry(g)
        ctrl = vis.get_view_control()
        fronte = np.asarray(occhio, float) - np.asarray(centro, float)
        norma = float(np.linalg.norm(fronte)) or 1.0
        ctrl.set_lookat(np.asarray(centro, float))
        ctrl.set_front(fronte / norma)
        ctrl.set_up(np.asarray(su, float))
        ctrl.set_zoom(0.75)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(destinazione, do_render=True)
    finally:
        vis.destroy_window()
    return destinazione


def main():
    ap = argparse.ArgumentParser(description="Viewer Open3D (manichino)")
    ap.add_argument("--c3d", required=True, help="C3D canonico da mostrare")
    ap.add_argument("--coord", action="store_true",
                    help="mostra le terne degli assi dei segmenti")
    ap.add_argument("--png", help="rende un fotogramma su file e esce, senza finestra")
    ap.add_argument("--frame", type=int, default=0, help="fotogramma da rendere con --png")
    args = ap.parse_args()
    if o3d is None:
        raise SystemExit("Open3D non installato in questo interprete "
                         "(vedi BIOMECH_VIEWER_PYTHON)")
    dati = carica(args.c3d)
    if args.png:
        istantanea(dati, args.frame, args.png, con_terne=args.coord)
        return
    esegui(dati, con_terne=args.coord)


if __name__ == "__main__":
    main()
