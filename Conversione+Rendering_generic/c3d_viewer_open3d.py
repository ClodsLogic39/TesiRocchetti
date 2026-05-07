#!/usr/bin/env python3
"""
C3D Motion Viewer - Open3D Version
Visualizza dati motion capture con Open3D + pytransform3d
Versione con rendering 3D avanzato e coordinate frames
"""

import json
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


SKELETON_CONNECTIONS = [
    ('pelvis_center', 'femur_left'),
    ('femur_left', 'tibia_left'),
    ('tibia_left', 'foot_left'),
    ('tibia_left', 'heel_left'),
    ('pelvis_center', 'femur_right'),
    ('femur_right', 'tibia_right'),
    ('tibia_right', 'foot_right'),
    ('tibia_right', 'heel_right'),
    ('pelvis_center', 'torso'),
    ('torso', 'head_neck'),
    ('torso', 'shoulder_width'),
    ('shoulder_width', 'humerus_left'),
    ('humerus_left', 'forearm_left'),
    ('shoulder_width', 'humerus_right'),
    ('humerus_right', 'forearm_right'),
    ('pelvis_center', 'pelvis_width'),
]


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4)]


def quaternion_to_transform(q, position):
    if q is None or len(q) != 4:
        return np.eye(4)

    if np.allclose(q, [1, 0, 0, 0]) or np.allclose(q, [0, 0, 0, 0]):
        transform = np.eye(4)
        transform[:3, 3] = position if position is not None else [0, 0, 0]
        return transform

    R = pr.matrix_from_quaternion(np.array(q))

    transform = np.eye(4)
    transform[:3, :3] = R
    if position is not None:
        transform[:3, 3] = position

    return transform


def load_data_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


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
        'segments': {}
    }

    for i, label in enumerate(point_labels):
        positions = points[:3, i, :].T
        data['segments'][label] = {
            'positions': positions.tolist(),
            'quaternions': [[0, 0, 0, 1]] * n_frames,
            'angular_velocity': [],
            'angular_acceleration': []
        }

    return data


class MotionDatabase:
    def __init__(self, data):
        self.data = data
        self.metadata = data['metadata']
        self.segments = data['segments']

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

    def get_start_point(self, name, frame):
        if name not in self.segments:
            return None
        pts = self.segments[name]['start_points']
        if isinstance(pts, list):
            return pts[frame]
        return pts[frame]

    def get_end_point(self, name, frame):
        if name not in self.segments:
            return None
        pts = self.segments[name]['end_points']
        if isinstance(pts, list):
            return pts[frame]
        return pts[frame]

    def get_quaternion(self, name, frame):
        if name not in self.segments:
            return None
        q = self.segments[name]['quaternions']
        if isinstance(q, list):
            return q[frame]
        return q[frame]

    def get_midpoint(self, name, frame):
        if name not in self.segments:
            return None
        mp = self.segments[name]['midpoints']
        if isinstance(mp, list):
            return mp[frame]
        return mp[frame]

    def get_all_segment_names(self):
        return list(self.segments.keys())


def create_joint_sphere(position, radius=0.015, color=None):
    if position is None:
        return None

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
    sphere.translate(position)

    if color is not None:
        sphere.paint_uniform_color(color)

    sphere.compute_vertex_normals()
    return sphere


def create_bone_line(start, end, color):
    if start is None or end is None:
        return None

    points = [start, end]
    lines = [[0, 1]]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([color])

    return line_set


def create_coordinate_frame(position, quaternion, size=0.04):
    if position is None or quaternion is None:
        return None

    if np.allclose(quaternion, [1, 0, 0, 0]) or np.allclose(quaternion, [0, 0, 0, 0]):
        return None

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

    transform = quaternion_to_transform(quaternion, position)

    coord_frame.transform(transform)

    return coord_frame


def build_frame_geometries(db, frame, show_coord_frames=True, transform_point=None):
    """Build all 3D geometries for a given frame.
    
    Args:
        db: MotionDatabase instance
        frame: frame index
        show_coord_frames: whether to show coordinate frames
        transform_point: optional point transformation function (e.g., for axis reorientation)
    """
    def transform(p):
        if p is None:
            return None
        return p if transform_point is None else transform_point(p)
    
    geometries = []
    segment_names = db.get_all_segment_names()

    for name in segment_names:
        start = db.get_start_point(name, frame)
        end = db.get_end_point(name, frame)
        color = hex_to_rgb(SEGMENT_COLORS.get(name, '#888888'))

        if start is not None:
            sphere = create_joint_sphere(start, radius=0.015, color=color)
            if sphere:
                geometries.append(sphere)

        if end is not None:
            sphere = create_joint_sphere(end, radius=0.015, color=color)
            if sphere:
                geometries.append(sphere)

    for (parent, child) in SKELETON_CONNECTIONS:
        parent_start = db.get_start_point(parent, frame)
        parent_end = db.get_end_point(parent, frame)
        child_start = db.get_start_point(child, frame)
        child_end = db.get_end_point(child, frame)

        connections = [
            (parent_start, child_start),
            (parent_start, child_end),
            (parent_end, child_start),
            (parent_end, child_end),
        ]

        color = hex_to_rgb(SEGMENT_COLORS.get(child, '#888888'))

        for start, end in connections:
            if start is not None and end is not None:
                line = create_bone_line(start, end, color)
                if line:
                    geometries.append(line)
                break

    if show_coord_frames:
        for name in segment_names:
            mid = db.get_midpoint(name, frame)
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
        'geometries': None,
        'db': db,
        'n_frames': n_frames,
        'fps': fps
    }

    center, max_range = compute_bounding_box(db)

    def transform_point(p):
        if p is None:
            return None
        return [p[0], p[1], p[2]]

    app = gui.Application.instance
    app.initialize()

    window = app.create_window("Open3D Motion Viewer - C3D", 1400, 800)
    w = window.layout

    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)

    scene_widget.scene.set_background(np.array([0.1, 0.1, 0.1, 1.0]))

    coord_axes = rendering.Scene.create_coordinate_frame(size=0.1)
    scene_widget.scene.add_geometry("coord_axes", coord_axes, rendering.MaterialRecord())

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=center - max_range * 1.5,
        max_bound=center + max_range * 1.5
    )
    scene_widget.setup_camera(max_range * 2, center, [0, 0, 1], [0, -1, 0])
    scene_widget.look_at(center, [0, 0, 1], 2.0)

    legend_panel = gui.VGrid(1, 0.02)
    legend_panel.set_background_color(gui.Color(0.15, 0.15, 0.15))

    title = gui.Label("Legend")
    title.font_size = 14
    title.font_weight = gui.Label.Weight.BOLD
    legend_panel.add_child(title)

    for name in sorted(segment_names):
        color_hex = SEGMENT_COLORS.get(name, '#888888')
        rgb = hex_to_rgb(color_hex)
        label = SEGMENT_LABELS.get(name, name)

        color_indicator = gui.Label("■")
        color_indicator.text_color = gui.Color(rgb[0], rgb[1], rgb[2])
        color_indicator.font_size = 12

        text_label = gui.Label(label)
        text_label.font_size = 11

        row = gui.HorizontalFlow()
        row.add_child(color_indicator)
        row.add_child(text_label)
        legend_panel.add_child(row)

    info_panel = gui.VGrid(1, 0.02)
    info_panel.set_background_color(gui.Color(0.15, 0.15, 0.15))

    info_title = gui.Label("Info")
    info_title.font_size = 14
    info_title.font_weight = gui.Label.Weight.BOLD
    info_panel.add_child(info_title)

    frame_label = gui.Label(f"Frame: 0 / {n_frames-1}")
    frame_label.font_size = 12
    info_panel.add_child(frame_label)

    time_label = gui.Label(f"Time: 0.00s")
    time_label.font_size = 12
    info_panel.add_child(time_label)

    fps_label = gui.Label(f"FPS: {fps}")
    fps_label.font_size = 12
    info_panel.add_child(fps_label)

    controls_title = gui.Label("Controls")
    controls_title.font_size = 14
    controls_title.font_weight = gui.Label.Weight.BOLD
    info_panel.add_child(controls_title)

    controls = [
        "←/→ : Prev/Next Frame",
        "SPACE : Play/Pause",
        "R : Reset View",
        "C : Toggle Coord",
        "Q : Quit"
    ]
    for ctrl in controls:
        ctrl_label = gui.Label(ctrl)
        ctrl_label.font_size = 10
        info_panel.add_child(ctrl_label)

    main_layout = gui.HorizontalFlow()
    main_layout.set_weight([0.8, 0.1, 0.1])

    main_layout.add_child(scene_widget)

    main_layout.add_child(legend_panel)

    main_layout.add_child(info_panel)

    w.add_child(main_layout)

    def update_frame(frame):
        scene_widget.scene.remove_geometry("skeleton")

        geometries = build_frame_geometries(state['db'], frame, show_coord_frames=state['show_coord'], transform_point=transform_point)

        for geom in geometries:
            scene_widget.scene.add_geometry(f"skeleton_{id(geom)}", geom, rendering.MaterialRecord())

        t = frame / state['fps']
        frame_label.text = f"Frame: {frame} / {state['n_frames']-1}"
        time_label.text = f"Time: {t:.2f}s"

    update_frame(0)

    def on_key(event):
        if event.type == gui.KeyEvent.UP:
            if event.key == ord(' '):
                state['is_playing'] = not state['is_playing']
            elif event.key == ord('Q') or event.key == ord('q'):
                app.quit()
            elif event.key == ord('R') or event.key == ord('r'):
                scene_widget.look_at(center, [0, 0, 1], 2.0)
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
            state['current_frame'] = (state['current_frame'] + 1) % state['n_frames']
            update_frame(state['current_frame'])

    app.run()


def visualize_open3d_legacy(data, show_coord_frames=True):
    """Legacy Open3D visualizer with key callbacks."""
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

    print(f"\nOpen3D Motion Viewer")
    print(f"Frames: {n_frames}, FPS: {fps}")
    print(f"Segments: {len(segment_names)}")
    print(f"Coordinate frames: {'ON' if show_coord_frames else 'OFF'}")
    print("\nControlli:")
    print("  [←/→] Frame precedente/successivo")
    print("  [SPACE] Play/Pause")
    print("  [R] Reset vista")
    print("  [C] Toggle coordinate frames")
    print("  [Q] Esci")

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
        print(f"Frame: {current_frame[0]}/{n_frames-1} | Time: {current_frame[0]/fps:.2f}s", end='\r')

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


def main():
    parser = argparse.ArgumentParser(description='C3D Motion Viewer - Open3D')
    parser.add_argument('directory', nargs='?', default='.', help='Working directory containing session files')
    parser.add_argument('--json', default=None, help='JSON input file (overrides auto-detect)')
    parser.add_argument('--c3d', default=None, help='C3D input file')
    parser.add_argument('--no-coord', action='store_true', help='Disable coordinate frames')

    args = parser.parse_args()

    working_dir = args.directory if args.directory != '.' else '.'

    if not os.path.isdir(working_dir):
        print(f"Error: {working_dir} is not a directory")
        return

    session_dirs = [d for d in os.listdir(working_dir)
                  if os.path.isdir(os.path.join(working_dir, d)) and d.startswith('running_')]

    if not session_dirs:
        print("No session directories found in working directory")
        return

    print("Available sessions:")
    for i, session in enumerate(sorted(session_dirs)):
        print(f"  {i+1}. {session}")

    if len(session_dirs) == 1:
        chosen = session_dirs[0]
        print(f"Auto-selected: {chosen}")
    else:
        try:
            choice = int(input("Select session number: ")) - 1
            if choice < 0 or choice >= len(session_dirs):
                print("Invalid selection")
                return
            chosen = sorted(session_dirs)[choice]
        except ValueError:
            print("Invalid input")
            return

    session_dir = os.path.join(working_dir, chosen)
    json_file = None
    c3d_file = None

    for f in os.listdir(session_dir):
        if f.endswith('_generic.json'):
            json_file = os.path.join(session_dir, f)
        elif f.endswith('.c3d'):
            c3d_file = os.path.join(session_dir, f)

    data = None

    if args.json:
        print(f"Loading from {args.json}...")
        data = load_data_json(args.json)
    elif args.c3d:
        print(f"Loading from {args.c3d}...")
        data = load_data_c3d(args.c3d)
    elif json_file:
        print(f"Loading from {json_file}...")
        data = load_data_json(json_file)
    elif c3d_file:
        print(f"Loading from {c3d_file}...")
        data = load_data_c3d(c3d_file)
    else:
        print(f"No _generic.json or .c3d file found in {chosen}")
        return

    print(f"Frames: {data['metadata']['n_frames']}, FPS: {data['metadata']['fps']}")
    print(f"Segments: {len(data['segments'])}")

    show_coord = not args.no_coord
    visualize_open3d_legacy(data, show_coord_frames=show_coord)


if __name__ == "__main__":
    main()