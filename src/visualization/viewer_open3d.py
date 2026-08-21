"""Viewer Open3D: figura umana a volumi ricostruita dal C3D canonico.

Ogni segmento e' un tronco di cono a sezione ellittica fra due punti canonici,
con il raggio preso al punto anziche' al segmento; il busto e' un volume unico
dal bacino alle spalle e la testa un ellissoide orientato lungo il collo. Gli
spessori sono espressi in frazioni della lunghezza del tronco del soggetto.

La composizione dei segmenti e i colori vengono da `anatomy`; punti, angoli e
quaternioni da `canonical.read`. Il modulo offre una finestra interattiva con
cursore dei fotogrammi e la scrittura di un'istantanea su file PNG.

Avvio: python -m src.visualization.viewer_open3d --c3d <file.c3d> [--coord]

Prerequisiti: numpy, scipy ed ezc3d per la parte dati; open3d per il rendering,
importato in modo opzionale, cosi' che lettura e composizione restino disponibili
anche senza. In ingresso un C3D canonico prodotto dalla conversione.
"""
import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import open3d as o3d
    from open3d.visualization import gui, rendering
except ImportError:
    o3d = None

from ..conversion import anatomy, canonical

# --- Colori e proporzioni della figura ---
# Colori di busto e testa, semispessore del corpo a ogni giunto in frazioni della
# lunghezza del tronco, segmenti che non vengono disegnati, schiacciamento delle
# sezioni e semiassi della testa.
GRIGIO = (0.5, 0.5, 0.5)

COLORE_BUSTO = (0.72, 0.68, 0.42)
COLORE_TESTA = (0.86, 0.72, 0.60)

_RAGGI = {"hip": 0.19, "knee": 0.13, "ankle": 0.075, "heel": 0.075, "toe": 0.055,
          "shoulder": 0.11, "elbow": 0.085, "wrist": 0.058}
_RAGGIO_BACINO = 0.09

_NON_DISEGNATI = frozenset({"neck", "clavicle_l"})
_RAGGIO_DEFAULT = 0.08

_PROFONDITA_ARTO = 0.85
_PROFONDITA_BUSTO = 0.62
_LARGHEZZA_BACINO = 1.60
_LARGHEZZA_SPALLE = 1.05
_TESTA = (0.15, 0.19, 0.22)


# --- Dati ---
# Lettura del C3D canonico e preparazione di punti, ossa, quaternioni e angoli da
# mostrare. Questa parte non dipende da Open3D.
def carica(percorso):
    dati = canonical.read(percorso)
    punti = {nome: np.asarray(v, dtype=float) for nome, v in dati["points"].items()}
    colori = anatomy.colours()

    ossa = []
    for segmento in anatomy.SEGMENTS:
        coppia = next((c for c in anatomy.segment_ends_options(segmento)
                       if all(k in punti for k in c)), None)
        if coppia:
            prossimale, distale = coppia
            ossa.append((segmento, prossimale, distale,
                         tuple(colori.get(segmento, GRIGIO))))

    ha_busto = all(k in punti for k in ("hip_l", "hip_r", "shoulder_l", "shoulder_r"))

    giunto_di = {spec["distal"]: giunto for giunto, spec in anatomy.JOINTS.items()}
    angoli = {}
    for segmento, _, _, _ in ossa:
        giunto = giunto_di.get(segmento)
        serie = dati["angles"].get((giunto, "sagittal")) if giunto else None
        if serie is not None:
            angoli[segmento] = (giunto, np.asarray(serie, dtype=float))
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
        "scala": _scala_corporea(punti),
        "angoli": angoli,
        "terne": terne,
        "terna_mondo": dati["world_frame"],
        "fps": float(dati["meta"]["fps"] or 30.0),
        "n_frame": int(dati["meta"]["n_frames"]),
        "sorgente": dati["meta"]["source"],
    }


# --- Misure del corpo ---
# Centri di bacino e spalle, lunghezza del tronco a cui tutte le proporzioni sono
# riferite e raggio associato a ciascun punto.
def _centro(punti, sinistra, destra):
    if sinistra not in punti or destra not in punti:
        return None
    return (punti[sinistra] + punti[destra]) / 2.0


def _scala_corporea(punti):
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
    nome = punto.rsplit("_", 1)[0] if punto[-2:] in ("_l", "_r") else punto
    return _RAGGI.get(nome, _RAGGIO_DEFAULT) * scala


# --- Primitive geometriche ---
# Costruzione delle forme elementari con cui la figura viene composta: tronco di
# cono a sezione ellittica, ellissoide, sfera e terna di riferimento.
def _base_ortonormale(asse, destra=None):
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
    m = o3d.geometry.TriangleMesh.create_sphere(1.0, resolution=risoluzione)
    v = np.asarray(m.vertices) * np.asarray(semiassi, dtype=float)
    if alto is not None:
        u, e1, e2 = _base_ortonormale(alto, destra)
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
    if not np.all(np.isfinite(quaternione)):
        return None
    m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=dimensione)
    m.rotate(Rotation.from_quat(quaternione).as_matrix(), center=(0, 0, 0))
    m.translate(centro)
    return m


# --- Composizione della figura ---
# Assembla le geometrie di un singolo fotogramma -- ossa, busto, testa ed
# eventuali terne -- e calcola l'inquadratura che contiene tutta la scena.
def costruisci(dati, frame, con_terne=False):
    scala = dati["scala"]
    punti = dati["punti"]

    def posa(nome):
        v = punti[nome][frame] if nome in punti else None
        return v if v is not None and np.all(np.isfinite(v)) else None

    def centro(sinistra, destra):
        a, b = posa(sinistra), posa(destra)
        return None if a is None or b is None else (a + b) / 2.0

    anche, spalle = centro("hip_l", "hip_r"), centro("shoulder_l", "shoulder_r")
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
        geometrie += [corpo, _sfera(a, ra * 0.95, colore), _sfera(b, rb * 0.95, colore)]
        if con_terne and segmento in dati["terne"]:
            assi = _terna(dati["terne"][segmento][frame], (a + b) / 2,
                          max(scala * 0.25, 0.08))
            if assi is not None:
                geometrie.append(assi)

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

    riferimento = posa("head")
    if riferimento is not None:
        base = spalle if spalle is not None else posa("shoulder_c")
        alto = (testa_asse := riferimento - base) if (
            base is not None and np.linalg.norm(riferimento - base) > 1e-6) else None
        if alto is None:
            alto = np.asarray(dati["terna_mondo"], dtype=float)[1]
        alto = alto / np.linalg.norm(alto)
        semiassi = np.asarray([s * scala for s in _TESTA])

        avanti = np.asarray(dati["terna_mondo"], dtype=float)[0]
        avanti = avanti - float(avanti @ alto) * alto
        norma = np.linalg.norm(avanti)
        centro_testa = (riferimento - avanti / norma * semiassi[1] if norma > 1e-6
                        else riferimento)

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


# --- Finestra e istantanee ---
# Finestra interattiva con cursore dei fotogrammi e legenda, e scrittura di un
# singolo fotogramma su file PNG tramite una finestra nascosta.
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
    righe = []
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
    if not 0 <= frame < dati["n_frame"]:
        raise ValueError(f"frame {frame} fuori intervallo (0..{dati['n_frame'] - 1})")
    geometrie = costruisci(dati, frame, con_terne)
    if not geometrie:
        raise ValueError(f"frame {frame}: nessun segmento con entrambi gli estremi noti")

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


# --- Riga di comando ---
# Apre il file indicato e avvia la finestra, o ne salva un'istantanea.
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
