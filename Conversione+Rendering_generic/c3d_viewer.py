#!/usr/bin/env python3
"""
C3D Motion Viewer - Matplotlib Version
Visualizza dati motion capture con matplotlib
Versione con rendering 3D e navigazione interattiva
"""

import json
import numpy as np
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

try:
    import ezc3d
    EZC3D_AVAILABLE = True
except ImportError:
    EZC3D_AVAILABLE = False


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


def load_data_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def load_data_c3d(c3d_path):
    if not EZC3D_AVAILABLE:
        print("Error: ezc3d not installed")
        return None

    c3d = ezc3d.c3d(c3d_path)

    points = c3d['data']['points']
    point_labels = c3d['parameters']['POINT']['LABELS']['value']

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
            'start_points': positions.tolist(),
            'end_points': positions.tolist(),
            'midpoints': positions.tolist(),
            'quaternions': [[0, 0, 0, 1]] * n_frames
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

    def get_start_point(self, name, frame):
        if name not in self.segments:
            return None
        pts = self.segments[name].get('start_points')
        if pts is None:
            return None
        if frame < len(pts) and pts[frame] is not None:
            return pts[frame]
        return None

    def get_end_point(self, name, frame):
        if name not in self.segments:
            return None
        pts = self.segments[name].get('end_points')
        if pts is None:
            return None
        if frame < len(pts) and pts[frame] is not None:
            return pts[frame]
        return None

    def get_midpoint(self, name, frame):
        if name not in self.segments:
            return None
        pts = self.segments[name].get('midpoints')
        if pts is None:
            return None
        if frame < len(pts) and pts[frame] is not None:
            return pts[frame]
        return None

    def get_quaternion(self, name, frame):
        if name not in self.segments:
            return None
        q = self.segments[name].get('quaternions')
        if q is None:
            return None
        if frame < len(q):
            return q[frame]
        return None

    def get_all_segment_names(self):
        return list(self.segments.keys())


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


def visualize_matplotlib(data):
    db = MotionDatabase(data)
    fps = db.metadata['fps']
    n_frames = db.metadata['n_frames']
    segment_names = db.get_all_segment_names()

    print(f"\nMatplotlib Motion Viewer")
    print(f"Frames: {n_frames}, FPS: {fps}")
    print(f"Segments: {len(segment_names)}")
    print("\nControlli:")
    print("  [←/→] Frame precedente/successivo")
    print("  [SPACE] Play/Pause")
    print("  [R] Reset vista")
    print("  [Q] Esci")

    center, max_range = compute_bounding_box(db)

    def transform_point(p):
        if p is None:
            return None
        return [p[0], -p[2], p[1]]

    center_transformed = transform_point(center.tolist()) if center is not None else [0, 0, 0]

    fig = plt.figure(figsize=(16, 8))
    fig.canvas.manager.set_window_title("Matplotlib Motion Viewer - C3D")

    ax_3d = fig.add_subplot(121, projection='3d')
    ax_legend = fig.add_subplot(122)
    ax_legend.axis('off')
    plt.subplots_adjust(bottom=0.2, right=0.85)

    ax_3d.set_xlim(center_transformed[0] - max_range, center_transformed[0] + max_range)
    ax_3d.set_ylim(center_transformed[1] - max_range, center_transformed[1] + max_range)
    ax_3d.set_zlim(center_transformed[2] - max_range, center_transformed[2] + max_range)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')

    initial_limits = {
        'xlim': ax_3d.get_xlim(),
        'ylim': ax_3d.get_ylim(),
        'zlim': ax_3d.get_zlim()
    }

    current_frame = [0]
    is_playing = [False]

    def get_point_transformed(name, frame, point_type):
        pt = None
        if point_type == 'start':
            pt = db.get_start_point(name, frame)
        elif point_type == 'end':
            pt = db.get_end_point(name, frame)
        elif point_type == 'mid':
            pt = db.get_midpoint(name, frame)
        return transform_point(pt) if pt else None

    def update_frame(frame):
        ax_3d.clear()
        
        ax_3d.set_xlim(initial_limits['xlim'])
        ax_3d.set_ylim(initial_limits['ylim'])
        ax_3d.set_zlim(initial_limits['zlim'])
        ax_3d.set_title(f'Frame: {frame} | Time: {frame/fps:.2f}s')

        for name in segment_names:
            color = SEGMENT_COLORS.get(name, '#888888')
            rgb = hex_to_rgb(color)

            start = get_point_transformed(name, frame, 'start')
            end = get_point_transformed(name, frame, 'end')
            mid = get_point_transformed(name, frame, 'mid')

            if start and end:
                ax_3d.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                          color=rgb, linewidth=2)

            if mid:
                ax_3d.scatter(mid[0], mid[1], mid[2], c=[rgb], s=30)

        for parent, child in SKELETON_CONNECTIONS:
            parent_mid = get_point_transformed(parent, frame, 'mid')
            child_mid = get_point_transformed(child, frame, 'mid')

            if parent_mid and child_mid:
                color = SEGMENT_COLORS.get(child, '#888888')
                rgb = hex_to_rgb(color)
                ax_3d.plot([parent_mid[0], child_mid[0]], 
                          [parent_mid[1], child_mid[1]], 
                          [parent_mid[2], child_mid[2]], color=rgb, linewidth=2)

        fig.canvas.draw_idle()
        print(f"Frame: {frame}/{n_frames-1} | Time: {frame/fps:.2f}s", end='\r')

    update_frame(0)

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, n_frames-1, valinit=0, valstep=1)

    def on_slider_change(val):
        current_frame[0] = int(val)
        update_frame(current_frame[0])

    slider.on_changed(on_slider_change)

    ax_legend.set_title('Legend', fontsize=10, fontweight='bold')
    y_pos = 0.95
    for name in sorted(SEGMENT_COLORS.keys()):
        if name in segment_names:
            color = SEGMENT_COLORS[name]
            rgb = hex_to_rgb(color)
            label = SEGMENT_LABELS.get(name, name)
            ax_legend.add_patch(mpatches.Rectangle((0.05, y_pos - 0.025), 0.06, 0.03, facecolor=rgb))
            ax_legend.text(0.13, y_pos, label, fontsize=8, va='center')
            y_pos -= 0.04

    ax_play = plt.axes([0.3, 0.01, 0.1, 0.04])
    btn_play = Button(ax_play, 'Play')

    def on_play_click(event):
        is_playing[0] = not is_playing[0]
        btn_play.label.set_text('Pause' if is_playing[0] else 'Play')

    btn_play.on_clicked(on_play_click)

    ax_reset = plt.axes([0.5, 0.01, 0.1, 0.04])
    btn_reset = Button(ax_reset, 'Reset')

    def on_reset_click(event):
        slider.set_val(0)

    btn_reset.on_clicked(on_reset_click)

    def on_key(event):
        if event.key == 'right':
            current_frame[0] = min(n_frames - 1, current_frame[0] + 1)
            slider.set_val(current_frame[0])
        elif event.key == 'left':
            current_frame[0] = max(0, current_frame[0] - 1)
            slider.set_val(current_frame[0])
        elif event.key == ' ':
            is_playing[0] = not is_playing[0]
            btn_play.label.set_text('Pause' if is_playing[0] else 'Play')
        elif event.key == 'r':
            slider.set_val(0)
        elif event.key == 'q':
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)

    from matplotlib.animation import FuncAnimation

    def update_animation(frame):
        if is_playing[0]:
            current_frame[0] = (current_frame[0] + 1) % n_frames
            slider.set_val(current_frame[0])
        return []

    ani = FuncAnimation(fig, update_animation, interval=1000/fps, blit=False)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='C3D Motion Viewer - Matplotlib')
    parser.add_argument('directory', nargs='?', default='.', help='Working directory')
    parser.add_argument('--json', default=None, help='JSON input file')
    parser.add_argument('--c3d', default=None, help='C3D input file')

    args = parser.parse_args()

    working_dir = args.directory if args.directory != '.' else '.'

    if not os.path.isdir(working_dir):
        print(f"Error: {working_dir} is not a directory")
        return

    session_dirs = [d for d in os.listdir(working_dir)
                  if os.path.isdir(os.path.join(working_dir, d)) and d.startswith('running_')]

    if not session_dirs:
        print("No session directories found")
        return

    if len(session_dirs) == 1:
        chosen = session_dirs[0]
        print(f"Auto-selected: {chosen}")
    else:
        print("Available sessions:")
        for i, session in enumerate(sorted(session_dirs)):
            print(f"  {i+1}. {session}")
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
        print("No data file found")
        return

    print(f"Frames: {data['metadata']['n_frames']}, FPS: {data['metadata']['fps']}")
    print(f"Segments: {len(data['segments'])}")

    visualize_matplotlib(data)


if __name__ == "__main__":
    main()