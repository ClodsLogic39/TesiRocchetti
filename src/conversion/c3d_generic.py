#!/usr/bin/env python3
"""
Visualizzatore C3D generico (Open3D).

Legge file C3D senza assumere label anatomiche o nomi di canale fissi.
Mostra tutti i canali dei punti come animazione frame-per-frame di una nuvola di punti.
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import ezc3d
except ImportError:
    ezc3d = None

try:
    import open3d as o3d
except ImportError:
    o3d = None


def normalize_label(raw_label):
    """Normalizza una label C3D: decodifica i byte e rimuove NUL e spazi ai bordi."""
    if isinstance(raw_label, bytes):
        text = raw_label.decode("utf-8", errors="ignore")
    else:
        text = str(raw_label)
    return text.replace("\x00", "").strip()


def _collect_c3d_files(directory, recursive):
    c3d_files = []
    if recursive:
        for root, _, files in os.walk(directory):
            for name in files:
                if name.lower().endswith(".c3d"):
                    c3d_files.append(os.path.join(root, name))
    else:
        for name in os.listdir(directory):
            full_path = os.path.join(directory, name)
            if os.path.isfile(full_path) and name.lower().endswith(".c3d"):
                c3d_files.append(full_path)
    return sorted(c3d_files)


def _find_candidate_dirs(root_dir, recursive):
    candidates = []
    if recursive:
        for current_root, _, files in os.walk(root_dir):
            if any(name.lower().endswith(".c3d") for name in files):
                candidates.append(current_root)
    else:
        dirs_to_check = [root_dir]
        for name in sorted(os.listdir(root_dir)):
            full_path = os.path.join(root_dir, name)
            if os.path.isdir(full_path):
                dirs_to_check.append(full_path)
        for d in dirs_to_check:
            files = _collect_c3d_files(d, recursive=False)
            if files:
                candidates.append(d)
    return sorted(set(candidates))


def _choose_directory_interactively(candidates, root_dir, dir_index=None):
    if not candidates:
        raise FileNotFoundError(f"Nessuna cartella con file .c3d trovata sotto: {root_dir}")

    if dir_index is not None:
        if dir_index < 0 or dir_index >= len(candidates):
            raise IndexError(f"Indice cartella {dir_index} fuori intervallo. Trovate {len(candidates)} cartelle.")
        return candidates[dir_index]

    if not sys.stdin.isatty():
        raise RuntimeError(
            "La selezione della cartella richiede input interattivo. Usa --dir-index per sceglierla."
        )

    print("\nSeleziona la cartella contenente il file C3D:")
    for i, d in enumerate(candidates):
        rel = os.path.relpath(d, root_dir)
        label = "." if rel == "." else rel
        print(f"  {i + 1}. {label}")

    while True:
        raw = input("Numero della cartella: ").strip()
        try:
            choice = int(raw) - 1
        except ValueError:
            print("Input non valido. Inserisci un numero.")
            continue
        if 0 <= choice < len(candidates):
            return candidates[choice]
        print("Selezione fuori intervallo.")


def pick_c3d_path(
    input_path,
    explicit_c3d=None,
    recursive=False,
    index=0,
    select_directory=True,
    dir_index=None,
):
    """Risolve quale file .c3d aprire da un percorso esplicito, un file o una cartella (con eventuale scelta interattiva)."""
    if explicit_c3d:
        if not os.path.isfile(explicit_c3d):
            raise FileNotFoundError(f"File C3D non trovato: {explicit_c3d}")
        return explicit_c3d

    if os.path.isfile(input_path):
        if not input_path.lower().endswith(".c3d"):
            raise ValueError(f"Il file di input non e' un .c3d: {input_path}")
        return input_path

    if not os.path.isdir(input_path):
        raise NotADirectoryError(f"Il percorso di input non e' un file ne' una cartella: {input_path}")

    search_dir = input_path
    if select_directory:
        candidates = _find_candidate_dirs(input_path, recursive=recursive)
        search_dir = _choose_directory_interactively(candidates, root_dir=input_path, dir_index=dir_index)

    c3d_files = _collect_c3d_files(search_dir, recursive=recursive)

    c3d_files = sorted(c3d_files)
    if not c3d_files:
        scope = "ricorsivamente" if recursive else "nella cartella"
        raise FileNotFoundError(f"Nessun file .c3d trovato {scope}: {search_dir}")

    if index < 0 or index >= len(c3d_files):
        raise IndexError(f"Indice {index} fuori intervallo. Trovati {len(c3d_files)} file.")
    return c3d_files[index]


def load_c3d_generic(c3d_path):
    """Legge un C3D in modo generico e restituisce punti, residui, label, fps e dimensioni."""
    if ezc3d is None:
        raise RuntimeError("ezc3d non installato. Installa con: pip install ezc3d")

    c3d = ezc3d.c3d(c3d_path)

    points = np.asarray(c3d["data"]["points"], dtype=float)
    if points.ndim != 3 or points.shape[0] < 3:
        raise ValueError(f"Forma dei dati dei punti inattesa: {points.shape}")

    n_points = points.shape[1]
    n_frames = points.shape[2]

    point_labels = c3d["parameters"].get("POINT", {}).get("LABELS", {}).get("value", [])
    labels = [normalize_label(v) for v in point_labels]
    if len(labels) < n_points:
        labels.extend([f"point_{i}" for i in range(len(labels), n_points)])
    elif len(labels) > n_points:
        labels = labels[:n_points]

    rate_values = c3d["parameters"].get("POINT", {}).get("RATE", {}).get("value", [60.0])
    fps = float(rate_values[0]) if len(rate_values) else 60.0
    if not np.isfinite(fps) or fps <= 0:
        fps = 60.0

    residual = None
    if points.shape[0] >= 4:
        residual = points[3, :, :]

    return {
        "path": c3d_path,
        "labels": labels,
        "xyz": points[:3, :, :],  # [3, n_punti, n_frame]
        "residual": residual,      # [n_punti, n_frame] oppure None
        "n_points": n_points,
        "n_frames": n_frames,
        "fps": fps,
    }


def summary_text(data):
    xyz = data["xyz"]
    finite_xyz = int(np.isfinite(xyz).sum())
    total_xyz = int(np.prod(xyz.shape))
    residual = data["residual"]
    residual_info = "non disponibile"
    if residual is not None:
        min_res = np.nanmin(residual)
        max_res = np.nanmax(residual)
        residual_info = f"disponibile (min={min_res:.3f}, max={max_res:.3f})"

    lines = [
        f"File: {data['path']}",
        f"Frame: {data['n_frames']}",
        f"Punti: {data['n_points']}",
        f"FPS: {data['fps']:.3f}",
        f"Valori XYZ finiti: {finite_xyz}/{total_xyz}",
        f"Residuo: {residual_info}",
        f"Prime label: {data['labels'][:10]}",
    ]
    return "\n".join(lines)


def _compute_valid_mask(frame_xyz, frame_residual):
    valid = np.all(np.isfinite(frame_xyz), axis=1)
    if frame_residual is None:
        return valid

    if np.any(frame_residual < 0) and np.any(frame_residual >= 0):
        valid = valid & (frame_residual >= 0)
    return valid


def run_open3d_viewer(data, point_size=6.0, show_coord=True, playback_fps=None, zoom=4.0):
    """Anima la nuvola di punti del C3D in Open3D (play/pausa, navigazione frame)."""
    if o3d is None:
        raise RuntimeError("open3d non installato. Installa con: pip install open3d")

    n_frames = data["n_frames"]
    fps = float(playback_fps) if playback_fps else float(data["fps"])
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    xyz = data["xyz"]  # [3, n_points, n_frames]
    residual = data["residual"]

    all_points = xyz.reshape(3, -1).T
    finite_rows = np.all(np.isfinite(all_points), axis=1)
    if np.any(finite_rows):
        min_bound = np.min(all_points[finite_rows], axis=0)
        max_bound = np.max(all_points[finite_rows], axis=0)
        center = (min_bound + max_bound) * 0.5
        diag = float(np.linalg.norm(max_bound - min_bound))
        if not np.isfinite(diag) or diag < 1e-6:
            diag = 1.0
    else:
        center = np.zeros(3)
        diag = 1.0

    # Normalizza la scala della scena cosi' il pinch del trackpad ha piu' margine su macOS.
    scene_scale = 2.0 / diag if diag > 1e-9 else 1.0
    xyz_view = xyz * scene_scale
    center_view = center * scene_scale
    diag_view = diag * scene_scale

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("C3D Generic Viewer (Open3D)", width=1280, height=720)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    if show_coord:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=diag_view * 0.08)
        vis.add_geometry(axes)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.08, 0.08], dtype=float)
    opt.point_size = float(point_size)

    ctr = vis.get_view_control()
    ctr.set_lookat(center_view.tolist())
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_zoom(float(zoom))
    if hasattr(ctr, "set_constant_z_near"):
        ctr.set_constant_z_near(max(diag_view * 1e-4, 1e-6))
    if hasattr(ctr, "set_constant_z_far"):
        ctr.set_constant_z_far(max(diag_view * 5000.0, 10.0))

    state = {
        "frame": 0,
        "playing": True,
        "quit": False,
        "zoom": float(zoom),
    }

    def update_frame(frame_idx):
        frame_xyz = xyz_view[:, :, frame_idx].T
        frame_residual = residual[:, frame_idx] if residual is not None else None
        valid = _compute_valid_mask(frame_xyz, frame_residual)
        pts = frame_xyz[valid]

        pcd.points = o3d.utility.Vector3dVector(pts)
        if pts.shape[0] > 0:
            color = np.tile(np.array([[0.2, 0.8, 1.0]], dtype=float), (pts.shape[0], 1))
            pcd.colors = o3d.utility.Vector3dVector(color)
        else:
            pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

        vis.update_geometry(pcd)
        print(
            f"Frame {frame_idx + 1}/{n_frames} | punti visibili: {pts.shape[0]}",
            end="\r",
            flush=True,
        )

    def on_space(_):
        state["playing"] = not state["playing"]
        return False

    def on_left(_):
        state["frame"] = max(0, state["frame"] - 1)
        update_frame(state["frame"])
        return False

    def on_right(_):
        state["frame"] = min(n_frames - 1, state["frame"] + 1)
        update_frame(state["frame"])
        return False

    def on_quit(_):
        state["quit"] = True
        return False

    vis.register_key_callback(ord(" "), on_space)
    vis.register_key_callback(ord("Q"), on_quit)
    vis.register_key_callback(ord("q"), on_quit)
    vis.register_key_callback(65361, on_left)   # Freccia sinistra (Linux)
    vis.register_key_callback(65363, on_right)  # Freccia destra (Linux)
    vis.register_key_callback(63234, on_left)   # Freccia sinistra (macOS)
    vis.register_key_callback(63235, on_right)  # Freccia destra (macOS)

    update_frame(0)
    last_time = time.time()

    print("\nControlli: SPAZIO play/pausa, Sinistra/Destra cambia frame, Q esci")
    while not state["quit"]:
        if not vis.poll_events():
            break

        now = time.time()
        if state["playing"] and (now - last_time) >= (1.0 / fps):
            state["frame"] = (state["frame"] + 1) % n_frames
            update_frame(state["frame"])
            last_time = now

        vis.update_renderer()

    vis.destroy_window()
    print("\nViewer chiuso.")


# Questo strumento vive in src/conversion/; per default i .c3d elaborati stanno
# in <radice-repo>/data/processed.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_PROCESSED = os.path.join(REPO_ROOT, "data", "processed")


def main():
    parser = argparse.ArgumentParser(
        description="Lettore C3D generico e viewer Open3D (indipendente dalle label)"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_DATA_PROCESSED,
        help="Percorso di un file C3D o cartella contenente file .c3d "
             "(default: <radice-repo>/data/processed)",
    )
    parser.add_argument(
        "--c3d",
        default=None,
        help="Percorso esplicito del file C3D (sovrascrive l'input posizionale)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Cerca i .c3d ricorsivamente quando l'input e' una cartella",
    )
    parser.add_argument(
        "--no-dir-select",
        action="store_true",
        help="Disabilita la selezione interattiva della cartella quando l'input e' una cartella",
    )
    parser.add_argument(
        "--dir-index",
        type=int,
        default=None,
        help="Indice (0-based) per scegliere tra le cartelle che contengono .c3d",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Quando ci sono piu' .c3d, sceglie questo indice nell'ordine (default: 0)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Stampa solo il riepilogo dei metadati del C3D, senza aprire la finestra Open3D",
    )
    parser.add_argument(
        "--no-coord",
        action="store_true",
        help="Disabilita la terna di riferimento nella vista Open3D",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=6.0,
        help="Dimensione dei punti nel rendering Open3D",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Forza gli FPS di riproduzione (default: POINT.RATE)",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=4.0,
        help="Zoom iniziale della camera (valori piu' bassi = vista piu' lontana)",
    )

    args = parser.parse_args()

    try:
        c3d_path = pick_c3d_path(
            input_path=args.input,
            explicit_c3d=args.c3d,
            recursive=args.recursive,
            index=args.index,
            select_directory=not args.no_dir_select,
            dir_index=args.dir_index,
        )
        data = load_c3d_generic(c3d_path)
    except Exception as exc:
        print(f"Errore: {exc}")
        sys.exit(1)

    print(summary_text(data))

    if args.summary_only:
        return

    try:
        run_open3d_viewer(
            data,
            point_size=args.point_size,
            show_coord=not args.no_coord,
            playback_fps=args.fps,
            zoom=args.zoom,
        )
    except Exception as exc:
        print(f"Errore: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
