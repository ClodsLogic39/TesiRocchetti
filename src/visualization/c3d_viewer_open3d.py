#!/usr/bin/env python3
"""
Visualizzatore di movimento C3D - versione Open3D.

Rendering 3D avanzato (con terne di riferimento) dei dati motion-capture tramite
Open3D + pytransform3d. Ricostruisce i segmenti anatomici dai punti del C3D e,
quando presenti, sovrappone i canali analogici (velocita'/accelerazione angolare,
angoli articolari).
"""

import numpy as np
import argparse
import os
import time

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Errore: Open3D non installato. Installare con: pip install open3d")

try:
    from pytransform3d import transformations as pt
    from pytransform3d import rotations as pr
    PYTRANSFORM3D_AVAILABLE = True
except ImportError:
    PYTRANSFORM3D_AVAILABLE = False
    print("Errore: pytransform3d non installato. Installare con: pip install pytransform3d")


SEGMENT_COLORS = {
    'pelvis_center': '#ec4899',
    'pelvis_width': '#f43f5e',
    'femur_left': '#22c55e',
    'femur_right': '#14b8a6',
    'tibia_left': '#10b981',
    'tibia_right': '#0d9488',
    'foot_left': '#0ea5e9',
    'foot_right': '#6366f1',
    'heel_left': '#f97316',
    'heel_right': '#ea580c',
    'torso': '#eab308',
    'shoulder_width': '#f59e0b',
    'humerus_left': '#22c55e',
    'humerus_right': '#3b82f6',
    'forearm_left': '#14b8a6',
    'forearm_right': '#8b5cf6',
    'head_neck': '#ef4444',
    'stride_left': '#84cc16',
    'stride_right': '#65a30d',
    'contact_point_left': '#a855f7',
    'contact_point_right': '#d946ef',
}

SEGMENT_LABELS = {
    'pelvis_center': 'Pelvis',
    'pelvis_width': 'Pelvis Width',
    'femur_left': 'Femur L',
    'femur_right': 'Femur R',
    'tibia_left': 'Tibia L',
    'tibia_right': 'Tibia R',
    'foot_left': 'Foot L',
    'foot_right': 'Foot R',
    'heel_left': 'Heel L',
    'heel_right': 'Heel R',
    'torso': 'Torso',
    'shoulder_width': 'Shoulders',
    'humerus_left': 'Humerus L',
    'humerus_right': 'Humerus R',
    'forearm_left': 'Forearm L',
    'forearm_right': 'Forearm R',
    'head_neck': 'Head/Neck',
    'stride_left': 'Stride L',
    'stride_right': 'Stride R',
    'contact_point_left': 'Contact L',
    'contact_point_right': 'Contact R',
}


# Connessioni dello scheletro: (segmento_A, punto_A, segmento_B, punto_B), dove
# punto e' 'start' (prossimale), 'end' (distale) o 'mid' (punto medio).
SKELETON_CONNECTIONS = [
    # Arti inferiori
    ('pelvis_center', 'start', 'femur_left', 'start'),
    ('femur_left', 'end', 'tibia_left', 'start'),
    ('tibia_left', 'end', 'foot_left', 'start'),
    ('tibia_left', 'end', 'heel_left', 'start'),
    ('pelvis_center', 'end', 'femur_right', 'start'),
    ('femur_right', 'end', 'tibia_right', 'start'),
    ('tibia_right', 'end', 'foot_right', 'start'),
    ('tibia_right', 'end', 'heel_right', 'start'),

    # Tronco / parte superiore
    ('pelvis_center', 'mid', 'torso', 'start'),
    ('torso', 'end', 'shoulder_width', 'mid'),
    ('shoulder_width', 'mid', 'head_neck', 'start'),

    # Arti superiori
    ('shoulder_width', 'start', 'humerus_left', 'start'),
    ('humerus_left', 'end', 'forearm_left', 'start'),
    ('shoulder_width', 'end', 'humerus_right', 'start'),
    ('humerus_right', 'end', 'forearm_right', 'start'),
]

JOINT_AXIS_SUFFIXES = [
    "flexion_extension",
    "lateral_bending",
    "axial_rotation",
    "abduction_adduction",
    "internal_external_rotation",
    "varus_valgus",
    "plantar_dorsiflexion",
    "inversion_eversion",
    "pronation_supination",
]


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4)]


def is_valid_vec3(v):
    if v is None:
        return False
    arr = np.asarray(v, dtype=float).reshape(-1)
    return arr.size == 3 and np.all(np.isfinite(arr))


def is_valid_quaternion_xyzw(q):
    if q is None:
        return False
    arr = np.asarray(q, dtype=float).reshape(-1)
    if arr.size != 4 or not np.all(np.isfinite(arr)):
        return False
    return np.linalg.norm(arr) > 1e-9


def normalize_c3d_label(raw_label):
    if isinstance(raw_label, bytes):
        s = raw_label.decode('utf-8', errors='ignore')
    else:
        s = str(raw_label)
    return s.replace('\x00', '').strip()


def quaternion_to_transform(q, position):
    if not is_valid_quaternion_xyzw(q):
        return np.eye(4)

    q_xyzw = np.asarray(q, dtype=float)
    if np.allclose(q_xyzw, [0, 0, 0, 1]) or np.allclose(q_xyzw, [0, 0, 0, 0]):
        transform = np.eye(4)
        transform[:3, 3] = position if is_valid_vec3(position) else [0, 0, 0]
        return transform

    # pytransform3d vuole l'ordine [w, x, y, z], mentre i dati esportati sono [x, y, z, w]
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
    q_wxyz = q_wxyz / np.linalg.norm(q_wxyz)
    R = pr.matrix_from_quaternion(q_wxyz)

    transform = np.eye(4)
    transform[:3, :3] = R
    if is_valid_vec3(position):
        transform[:3, 3] = np.asarray(position, dtype=float)

    return transform


def load_data_c3d(c3d_path):
    try:
        import ezc3d
    except ImportError:
        print("Errore: ezc3d non installato")
        return None

    c3d = ezc3d.c3d(c3d_path)

    points = c3d['data']['points']
    point_labels = c3d['parameters']['POINT']['LABELS']['value']

    analogs = c3d['data']['analogs']
    analog_labels = c3d['parameters']['ANALOG']['LABELS']['value']

    fps = c3d['parameters']['POINT']['RATE']['value'][0]
    n_frames = points.shape[2]

    data = {
        'metadata': {
            'fps': fps,
            'n_frames': n_frames,
            'duration_seconds': n_frames / fps
        },
        'segments': {},
        'joints': {}
    }

    analog_map = {}
    analog_data = None
    if analogs is not None:
        if analogs.ndim == 3 and analogs.shape[0] > 0:
            analog_data = analogs[0]  # (n_analogs, n_frames)
        elif analogs.ndim == 2:
            analog_data = analogs

    if analog_data is not None:
        for i, raw_label in enumerate(analog_labels):
            if i >= analog_data.shape[0]:
                break
            label = normalize_c3d_label(raw_label)
            analog_map[label] = analog_data[i]

        # Interpreta i canali analogici degli angoli articolari, esportati come "<joint>_<asse>".
        for label, values in analog_map.items():
            if label.endswith("_omega_x") or label.endswith("_omega_y") or label.endswith("_omega_z"):
                continue
            if label.endswith("_alpha_x") or label.endswith("_alpha_y") or label.endswith("_alpha_z"):
                continue

            for axis_name in JOINT_AXIS_SUFFIXES:
                suffix = f"_{axis_name}"
                if not label.endswith(suffix):
                    continue
                joint_name = label[:-len(suffix)]
                if not joint_name:
                    break
                if len(values) != n_frames:
                    break
                data['joints'].setdefault(joint_name, {})[axis_name] = np.asarray(values, dtype=float).tolist()
                break

    # Costruisce una mappa label -> (n_frames, 3) per le traiettorie dei punti.
    point_map = {}
    for i, raw_point_label in enumerate(point_labels):
        label = normalize_c3d_label(raw_point_label)
        point_map[label] = points[:3, i, :].T

    # Ricostruisce i segmenti anatomici:
    # - <segmento>_start -> punto prossimale
    # - <segmento>_end   -> punto distale
    # - <segmento>       -> punto medio (se disponibile)
    # I landmark *_center non sono ossa: vengono ignorati nel rendering dei segmenti.
    base_segment_names = sorted({
        name for name in SEGMENT_COLORS
        if f"{name}_start" in point_map or f"{name}_end" in point_map or name in point_map
    })

    # Ripiego per set di label non standard: deduce i nomi dei segmenti da *_start/*_end.
    if not base_segment_names:
        inferred = set()
        for label in point_map.keys():
            if label.endswith('_start'):
                inferred.add(label[:-6])
            elif label.endswith('_end'):
                inferred.add(label[:-4])
        base_segment_names = sorted(inferred)
    for seg_name in base_segment_names:
        start = point_map.get(f"{seg_name}_start")
        end = point_map.get(f"{seg_name}_end")
        mid = point_map.get(seg_name)

        if start is None and end is None and mid is None:
            continue
        if start is None:
            start = mid if mid is not None else end
        if end is None:
            end = mid if mid is not None else start
        if mid is None:
            mid = 0.5 * (start + end)

        omega = np.zeros((n_frames, 3))
        alpha = np.zeros((n_frames, 3))
        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            omega_key = f"{seg_name}_omega_{axis_name}"
            alpha_key = f"{seg_name}_alpha_{axis_name}"
            if omega_key in analog_map and len(analog_map[omega_key]) == n_frames:
                omega[:, axis_idx] = analog_map[omega_key]
            if alpha_key in analog_map and len(analog_map[alpha_key]) == n_frames:
                alpha[:, axis_idx] = analog_map[alpha_key]

        data['segments'][seg_name] = {
            'start_points': start.tolist(),
            'end_points': end.tolist(),
            'midpoints': mid.tolist(),
            'quaternions': [[0, 0, 0, 1]] * n_frames,
            'angular_velocity': omega.tolist(),
            'angular_acceleration': alpha.tolist()
        }

    return data


def find_c3d_in_directory(directory):
    if not os.path.isdir(directory):
        return None

    c3d_files = sorted([f for f in os.listdir(directory) if f.endswith('.c3d')])
    if not c3d_files:
        return None

    generic_files = [f for f in c3d_files if f.endswith('_generic.c3d')]
    chosen = sorted(generic_files)[0] if generic_files else c3d_files[0]
    return os.path.join(directory, chosen)


def resolve_c3d_for_session(session_dir):
    """Trova il C3D della sessione: prima nella cartella scelta, poi nella *_processed corrispondente."""
    # 1) Se e' stata scelta una cartella gia' elaborata, cerca direttamente li'.
    c3d_path = find_c3d_in_directory(session_dir)
    if c3d_path:
        return c3d_path

    # 2) Se e' stata scelta una cartella grezza running_*, cerca nella *_processed gemella.
    if not session_dir.endswith('_processed'):
        processed_dir = f"{session_dir}_processed"
        c3d_path = find_c3d_in_directory(processed_dir)
        if c3d_path:
            return c3d_path

    return None


class MotionDatabase:
    """Accesso ai dati di una sessione: segmenti (punti, quaternioni, omega/alpha) e angoli articolari."""

    def __init__(self, data):
        self.data = data
        self.metadata = data['metadata']
        self.segments = data['segments']
        self.joints = data.get('joints', {})

    def filter_frames(self, start=0, end=None):
        if end is None:
            end = self.metadata['n_frames']
        return range(max(0, start), min(end, self.metadata['n_frames']))

    def get_segment(self, name, frame):
        if name not in self.segments:
            return None
        seg = self.segments[name]
        return {
            'start_point': seg['start_points'][frame],
            'end_point': seg['end_points'][frame],
            'midpoint': seg['midpoints'][frame],
            'quaternion': seg['quaternions'][frame],
            'angular_velocity': seg['angular_velocity'][frame] if 'angular_velocity' in seg else [],
            'angular_acceleration': seg['angular_acceleration'][frame] if 'angular_acceleration' in seg else []
        }

    def _frame_value(self, name, frame, *keys):
        """Valore al frame dato dalla prima serie disponibile tra `keys`, o None se il segmento non esiste."""
        seg = self.segments.get(name)
        if seg is None:
            return None
        for key in keys:
            series = seg.get(key)
            if series is not None:
                return series[frame]
        return None

    def get_start_point(self, name, frame):
        return self._frame_value(name, frame, 'start_points', 'positions')

    def get_end_point(self, name, frame):
        return self._frame_value(name, frame, 'end_points', 'positions')

    def get_midpoint(self, name, frame):
        return self._frame_value(name, frame, 'midpoints', 'positions')

    def get_quaternion(self, name, frame):
        return self._frame_value(name, frame, 'quaternions')

    def get_all_segment_names(self):
        return list(self.segments.keys())

    def get_all_joint_names(self):
        return list(self.joints.keys())

    def get_joint_axes(self, joint_name):
        if joint_name not in self.joints:
            return []
        return list(self.joints[joint_name].keys())

    def get_joint_angle(self, joint_name, axis_name, frame):
        if joint_name not in self.joints:
            return None
        axis_values = self.joints[joint_name].get(axis_name)
        if axis_values is None or frame >= len(axis_values):
            return None
        return axis_values[frame]


def create_joint_sphere(position, radius=0.015, color=None):
    if not is_valid_vec3(position):
        return None

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
    sphere.translate(np.asarray(position, dtype=float))

    if color is not None:
        sphere.paint_uniform_color(color)

    sphere.compute_vertex_normals()
    return sphere


def create_bone_line(start, end, color):
    if not is_valid_vec3(start) or not is_valid_vec3(end):
        return None

    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    if np.linalg.norm(end_arr - start_arr) < 1e-9:
        return None

    points = [start_arr.tolist(), end_arr.tolist()]
    lines = [[0, 1]]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([color])

    return line_set


def create_coordinate_frame(position, quaternion, size=0.04):
    if not is_valid_vec3(position) or not is_valid_quaternion_xyzw(quaternion):
        return None

    q_xyzw = np.asarray(quaternion, dtype=float)
    if np.allclose(q_xyzw, [0, 0, 0, 1]) or np.allclose(q_xyzw, [0, 0, 0, 0]):
        return None

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

    transform = quaternion_to_transform(quaternion, position)
    if not np.all(np.isfinite(transform)):
        return None

    coord_frame.transform(transform)

    return coord_frame


def get_anchor_point(db, segment_name, frame, anchor):
    if anchor == 'start':
        return db.get_start_point(segment_name, frame)
    if anchor == 'end':
        return db.get_end_point(segment_name, frame)
    if anchor == 'mid':
        return db.get_midpoint(segment_name, frame)
    return None


def build_frame_geometries(db, frame, show_coord_frames=True, transform_point=None):
    """Costruisce tutte le geometrie 3D per un dato frame.

    Args:
        db: istanza di MotionDatabase
        frame: indice del frame
        show_coord_frames: se mostrare le terne di riferimento
        transform_point: funzione opzionale di trasformazione dei punti (es. riorientamento assi)
    """
    def transform(p):
        if p is None:
            return None
        return p if transform_point is None else transform_point(p)
    
    geometries = []
    segment_names = db.get_all_segment_names()

    for name in segment_names:
        start = transform(db.get_start_point(name, frame))
        end = transform(db.get_end_point(name, frame))
        color = hex_to_rgb(SEGMENT_COLORS.get(name, '#888888'))

        if start is not None:
            sphere = create_joint_sphere(start, radius=0.015, color=color)
            if sphere:
                geometries.append(sphere)

        if end is not None:
            sphere = create_joint_sphere(end, radius=0.015, color=color)
            if sphere:
                geometries.append(sphere)

    for (parent, parent_anchor, child, child_anchor) in SKELETON_CONNECTIONS:
        start = get_anchor_point(db, parent, frame, parent_anchor)
        end = get_anchor_point(db, child, frame, child_anchor)
        if start is None or end is None:
            continue

        color = hex_to_rgb(SEGMENT_COLORS.get(child, '#888888'))
        line = create_bone_line(transform(start), transform(end), color)
        if line:
            geometries.append(line)

    if show_coord_frames:
        for name in segment_names:
            mid = transform(db.get_midpoint(name, frame))
            q = db.get_quaternion(name, frame)

            if mid is not None and q is not None:
                coord = create_coordinate_frame(mid, q, size=0.04)
                if coord:
                    geometries.append(coord)

    return geometries


def compute_bounding_box(db):
    segment_names = db.get_all_segment_names()
    n_frames = db.metadata['n_frames']

    all_points = []

    sample_frames = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]

    for frame in sample_frames:
        for name in segment_names:
            start = db.get_start_point(name, frame)
            end = db.get_end_point(name, frame)

            if start is not None:
                all_points.append(start)
            if end is not None:
                all_points.append(end)

    if not all_points:
        return np.array([0, 0, 0]), 1.0

    arr = np.array(all_points)
    center = arr.mean(axis=0)
    max_range = np.max(arr.max(axis=0) - arr.min(axis=0)) / 2

    return center, max_range


def visualize_open3d_gui(data, show_coord_frames=True):
    try:
        from open3d.visualization import gui
        from open3d.visualization import rendering
    except ImportError:
        print("Open3D GUI non disponibile (versione vecchia). Uso Visualizer standard.")
        visualize_open3d_legacy(data, show_coord_frames)
        return

    db = MotionDatabase(data)
    fps = db.metadata['fps']
    n_frames = db.metadata['n_frames']
    segment_names = db.get_all_segment_names()

    state = {
        'current_frame': 0,
        'is_playing': False,
        'show_coord': show_coord_frames,
        'geometry_names': [],
        'db': db,
        'n_frames': n_frames,
        'fps': fps,
        'last_update_time': time.time()
    }

    center, max_range = compute_bounding_box(db)

    def transform_point(p):
        if p is None:
            return None
        return [p[0], p[1], p[2]]

    app = gui.Application.instance
    app.initialize()

    window = app.create_window("Open3D Motion Viewer - C3D", 1400, 800)

    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)

    if hasattr(scene_widget.scene, "set_background"):
        scene_widget.scene.set_background(np.array([0.1, 0.1, 0.1, 1.0]))

    coord_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    scene_widget.scene.add_geometry("coord_axes", coord_axes, rendering.MaterialRecord())

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=center - max_range * 1.5,
        max_bound=center + max_range * 1.5
    )
    center_np = np.asarray(center, dtype=np.float32)
    try:
        scene_widget.setup_camera(60.0, bbox, center_np)
    except TypeError:
        scene_widget.setup_camera(60.0, bbox, center_np.reshape((3, 1)))
    eye = center_np + np.array([0.0, 0.0, max_range * 2.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    scene_widget.look_at(center_np, eye, up)

    legend_panel = gui.ScrollableVert()
    legend_content = gui.Vert(4, gui.Margins(8, 8, 8, 8))
    legend_panel.add_child(legend_content)

    title = gui.Label("Legenda")
    legend_content.add_child(title)

    for name in sorted(segment_names):
        color_hex = SEGMENT_COLORS.get(name, '#888888')
        rgb = hex_to_rgb(color_hex)
        label = SEGMENT_LABELS.get(name, name)

        color_indicator = gui.Label("■")
        color_indicator.text_color = gui.Color(rgb[0], rgb[1], rgb[2])

        text_label = gui.Label(label)

        row = gui.Horiz()
        row.add_child(color_indicator)
        row.add_child(text_label)
        legend_content.add_child(row)

    info_panel = gui.ScrollableVert()
    info_content = gui.Vert(4, gui.Margins(8, 8, 8, 8))
    info_panel.add_child(info_content)

    info_title = gui.Label("Info")
    info_content.add_child(info_title)

    frame_label = gui.Label(f"Frame: 0 / {n_frames-1}")
    info_content.add_child(frame_label)

    time_label = gui.Label(f"Tempo: 0.00s")
    info_content.add_child(time_label)

    fps_label = gui.Label(f"FPS: {fps}")
    info_content.add_child(fps_label)

    controls_title = gui.Label("Controlli")
    info_content.add_child(controls_title)

    controls = [
        "←/→ : Frame prec./succ.",
        "SPAZIO : Play/Pausa",
        "R : Reset vista",
        "C : Mostra/nascondi terne",
        "Q : Esci"
    ]
    for ctrl in controls:
        ctrl_label = gui.Label(ctrl)
        info_content.add_child(ctrl_label)

    angles_title = gui.Label("Angoli articolari (gradi)")
    info_content.add_child(angles_title)

    angle_labels = []
    joint_pairs = []
    for joint_name in sorted(db.get_all_joint_names()):
        for axis_name in db.get_joint_axes(joint_name):
            label = gui.Label(f"{joint_name}.{axis_name}: --")
            info_content.add_child(label)
            angle_labels.append(label)
            joint_pairs.append((joint_name, axis_name))

    if not angle_labels:
        no_angles_label = gui.Label("Nessun dato di angolo articolare in questa sorgente")
        info_content.add_child(no_angles_label)

    window.add_child(scene_widget)
    window.add_child(legend_panel)
    window.add_child(info_panel)

    def on_layout(layout_context):
        r = window.content_rect
        panel_w = min(320, max(220, int(r.width * 0.22)))
        gap = 6
        scene_w = max(1, r.width - panel_w * 2 - gap * 2)
        scene_widget.frame = gui.Rect(r.x, r.y, scene_w, r.height)
        legend_panel.frame = gui.Rect(r.x + scene_w + gap, r.y, panel_w, r.height)
        info_panel.frame = gui.Rect(r.x + scene_w + gap + panel_w + gap, r.y, panel_w, r.height)

    window.set_on_layout(on_layout)

    mesh_material = rendering.MaterialRecord()
    mesh_material.shader = "defaultLit"
    line_material = rendering.MaterialRecord()
    line_material.shader = "unlitLine"
    if hasattr(line_material, "line_width"):
        line_material.line_width = 2.0

    def update_frame(frame):
        for geom_name in state['geometry_names']:
            scene_widget.scene.remove_geometry(geom_name)
        state['geometry_names'] = []

        geometries = build_frame_geometries(state['db'], frame, show_coord_frames=state['show_coord'], transform_point=transform_point)

        for i, geom in enumerate(geometries):
            geom_name = f"skeleton_{i}"
            try:
                material = line_material if isinstance(geom, o3d.geometry.LineSet) else mesh_material
                scene_widget.scene.add_geometry(geom_name, geom, material)
                state['geometry_names'].append(geom_name)
            except RuntimeError:
                # Salta le geometrie non valide del frame invece di far crashare il viewer.
                continue

        t = frame / state['fps']
        frame_label.text = f"Frame: {frame} / {state['n_frames']-1}"
        time_label.text = f"Tempo: {t:.2f}s"
        for idx, (joint_name, axis_name) in enumerate(joint_pairs):
            angle_value = db.get_joint_angle(joint_name, axis_name, frame)
            angle_text = "--" if angle_value is None else f"{angle_value:.2f}"
            angle_labels[idx].text = f"{joint_name}.{axis_name}: {angle_text}"

    update_frame(0)

    def on_key(event):
        if event.type == gui.KeyEvent.UP:
            if event.key == ord(' '):
                state['is_playing'] = not state['is_playing']
                state['last_update_time'] = time.time()
            elif event.key == ord('Q') or event.key == ord('q'):
                app.quit()
            elif event.key == ord('R') or event.key == ord('r'):
                scene_widget.look_at(center_np, eye, up)
            elif event.key == ord('C') or event.key == ord('c'):
                state['show_coord'] = not state['show_coord']
                update_frame(state['current_frame'])
            elif event.key == gui.KeyName.LEFT:
                state['current_frame'] = max(0, state['current_frame'] - 1)
                update_frame(state['current_frame'])
            elif event.key == gui.KeyName.RIGHT:
                state['current_frame'] = min(state['n_frames'] - 1, state['current_frame'] + 1)
                update_frame(state['current_frame'])

        return gui.Widget.EventCallbackResult.CONSUMED

    scene_widget.set_on_key(on_key)

    def on_tick():
        if state['is_playing']:
            now = time.time()
            if now - state['last_update_time'] < (1.0 / max(state['fps'], 1.0)):
                return True
            state['last_update_time'] = now
            state['current_frame'] = (state['current_frame'] + 1) % state['n_frames']
            update_frame(state['current_frame'])
        return True

    if hasattr(window, "set_on_tick_event"):
        window.set_on_tick_event(on_tick)

    app.run()


def visualize_open3d_legacy(data, show_coord_frames=True):
    """Visualizzatore Open3D legacy (Visualizer con callback da tastiera), usato sulle versioni senza GUI."""
    if not OPEN3D_AVAILABLE:
        print("Open3D non disponibile")
        return

    if not PYTRANSFORM3D_AVAILABLE:
        print("pytransform3d non disponibile")
        return

    db = MotionDatabase(data)
    fps = db.metadata['fps']
    n_frames = db.metadata['n_frames']
    segment_names = db.get_all_segment_names()

    print(f"\nVisualizzatore di movimento Open3D")
    print(f"Frame: {n_frames}, FPS: {fps}")
    print(f"Segmenti: {len(segment_names)}")
    print(f"Terne di riferimento: {'ON' if show_coord_frames else 'OFF'}")
    print("\nControlli:")
    print("  [←/→] Frame precedente/successivo")
    print("  [SPAZIO] Play/Pausa")
    print("  [R] Reset vista")
    print("  [C] Mostra/nascondi terne di riferimento")
    print("  [Q] Esci")
    print("\nLegenda segmenti:")
    for name in sorted(segment_names):
        print(f"  - {SEGMENT_LABELS.get(name, name)} ({name})")

    center, max_range = compute_bounding_box(db)

    def transform_point(p):
        if p is None:
            return None
        return [p[0], p[1], p[2]]

    center_t = transform_point(center.tolist()) if center is not None else [0, 0, 0]

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Open3D Motion Viewer - C3D", width=1280, height=720)

    ctr = vis.get_view_control()
    ctr.set_lookat(center_t)
    ctr.set_front([0.5, -0.5, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(max_range * 2)

    current_frame = [0]
    is_playing = [False]
    show_coord = [show_coord_frames]
    current_geometries = [None]
    joint_names = sorted(db.get_all_joint_names())

    def format_joint_angles(frame, max_items=6):
        if not joint_names:
            return "Angoli articolari: n/d"
        parts = []
        for joint_name in joint_names:
            for axis_name in db.get_joint_axes(joint_name):
                angle = db.get_joint_angle(joint_name, axis_name, frame)
                if angle is None:
                    continue
                parts.append(f"{joint_name}.{axis_name}={angle:.1f}°")
                if len(parts) >= max_items:
                    return "Angoli: " + " | ".join(parts) + " ..."
        return "Angoli: " + (" | ".join(parts) if parts else "n/d")

    def create_geometry_for_frame(frame):
        return build_frame_geometries(db, frame, show_coord[0], transform_point)

    current_geometries[0] = create_geometry_for_frame(current_frame[0])

    for geom in current_geometries[0]:
        vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.line_width = 2.0
    opt.point_size = 5.0

    def update_visualization():
        vis.clear_geometries()
        current_geometries[0] = create_geometry_for_frame(current_frame[0])
        for geom in current_geometries[0]:
            vis.add_geometry(geom)
        print(
            f"Frame: {current_frame[0]}/{n_frames-1} | Tempo: {current_frame[0]/fps:.2f}s | {format_joint_angles(current_frame[0])}",
            end='\r'
        )

    def on_next_frame(vis):
        current_frame[0] = min(n_frames - 1, current_frame[0] + 1)
        update_visualization()

    def on_prev_frame(vis):
        current_frame[0] = max(0, current_frame[0] - 1)
        update_visualization()

    def on_toggle_play(vis):
        is_playing[0] = not is_playing[0]
        print(f"Play: {is_playing[0]}", end='\r')

    def on_reset_view(vis):
        ctr.set_lookat(center_t)
        ctr.set_front([0.5, -0.5, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(max_range * 2)

    def on_toggle_coord(vis):
        show_coord[0] = not show_coord[0]
        update_visualization()

    def on_quit(vis):
        vis.close()

    vis.register_key_callback(ord(' '), on_toggle_play)
    vis.register_key_callback(ord('R'), on_reset_view)
    vis.register_key_callback(ord('C'), on_toggle_coord)
    vis.register_key_callback(ord('Q'), on_quit)
    vis.register_key_callback(65361, on_prev_frame)
    vis.register_key_callback(65363, on_next_frame)
    vis.register_key_callback(63234, on_prev_frame)
    vis.register_key_callback(63235, on_next_frame)

    last_time = time.time()
    while True:
        running = vis.poll_events()
        if not running:
            break
        if is_playing[0]:
            current_time = time.time()
            if current_time - last_time >= 1.0 / fps:
                current_frame[0] = (current_frame[0] + 1) % n_frames
                last_time = current_time
                update_visualization()
        vis.update_renderer()

    vis.destroy_window()
    print("\nChiuso.")


# Questo viewer vive in src/visualization/; per default le sessioni elaborate
# stanno in <radice-repo>/data/processed.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_PROCESSED = os.path.join(REPO_ROOT, "data", "processed")


def main():
    parser = argparse.ArgumentParser(description='Visualizzatore di movimento C3D - Open3D (solo C3D)')
    parser.add_argument('directory', nargs='?', default=DEFAULT_DATA_PROCESSED,
                        help='Cartella di lavoro con le sessioni running_*_processed')
    parser.add_argument('--c3d', default=None, help='File C3D esplicito (sovrascrive il rilevamento automatico)')
    parser.add_argument('--no-coord', action='store_true', help='Disabilita le terne di riferimento')

    args = parser.parse_args()

    working_dir = args.directory

    if not os.path.isdir(working_dir):
        print(f"Errore: {working_dir} non e' una cartella")
        return

    if args.c3d:
        c3d_file = args.c3d
        if not os.path.isfile(c3d_file):
            print(f"Errore: file C3D non trovato: {c3d_file}")
            return
    else:
        session_dirs = sorted([
            d for d in os.listdir(working_dir)
            if os.path.isdir(os.path.join(working_dir, d)) and d.startswith('running_')
        ])

        if not session_dirs:
            print("Nessuna cartella di sessione trovata nella cartella di lavoro")
            return

        print("Sessioni disponibili:")
        for i, session in enumerate(session_dirs):
            print(f"  {i+1}. {session}")

        if len(session_dirs) == 1:
            chosen = session_dirs[0]
            print(f"Selezione automatica: {chosen}")
        else:
            try:
                choice = int(input("Numero della sessione: ")) - 1
                if choice < 0 or choice >= len(session_dirs):
                    print("Selezione non valida")
                    return
                chosen = session_dirs[choice]
            except ValueError:
                print("Input non valido")
                return

        session_dir = os.path.join(working_dir, chosen)
        c3d_file = resolve_c3d_for_session(session_dir)
        if c3d_file is None:
            print(f"Nessun .c3d trovato per la sessione {chosen}.")
            print("Esegui prima process_session_generic.py per generare <sessione>_processed/<sessione>_generic.c3d")
            return

    print(f"Caricamento da {c3d_file}...")
    data = load_data_c3d(c3d_file)
    if data is None:
        return

    print(f"Frame: {data['metadata']['n_frames']}, FPS: {data['metadata']['fps']}")
    print(f"Segmenti: {len(data['segments'])}")

    show_coord = not args.no_coord
    visualize_open3d_gui(data, show_coord_frames=show_coord)


if __name__ == "__main__":
    main()
