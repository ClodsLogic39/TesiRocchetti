#!/usr/bin/env python3
"""
Process MediaPipe motion capture data to C3D with quaternions and joint angles.
Generic version: auto-detects available segments and computes all possible joints.

Session: Any running_* folder with segments_frames.json and segments_summary.json
"""

import csv
import json
import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d

# Try importing ezc3d for C3D export
try:
    import ezc3d
    EZC3D_AVAILABLE = True
except ImportError:
    EZC3D_AVAILABLE = False
    print("Warning: ezc3d not available. Will save as JSON/CSV only.")

# =============================================================================
# CONFIGURATION - Segment to Joint Mapping (ISB Standard)
# =============================================================================

# Mapping: which segments form which joints
SEGMENT_TO_JOINT = {
    # Spine/Pelvis
    "pelvis_center": "pelvis",
    "pelvis_width": "pelvis",
    "torso": "spine",
    "head_neck": "neck",
    "shoulder_width": "shoulder_center",
    
    # Left lower limb
    "femur_left": "hip_left",
    "tibia_left": "knee_left",
    "foot_left": "ankle_left",
    "heel_left": "heel_left",
    "stride_left": "stride_left",
    "contact_point_left": "contact_left",
    
    # Right lower limb
    "femur_right": "hip_right",
    "tibia_right": "knee_right",
    "foot_right": "ankle_right",
    "heel_right": "heel_right",
    "stride_right": "stride_right",
    "contact_point_right": "contact_right",
    
    # Left upper limb
    "humerus_left": "shoulder_left",
    "forearm_left": "elbow_left",
    
    # Right upper limb
    "humerus_right": "shoulder_right",
    "forearm_right": "elbow_right",
}

# Joint definitions: proximal segment → distal segment
JOINT_DEFINITIONS = {
    # Spine
    "spine": {
        "proximal": "pelvis_center",
        "distal": "torso",
        "sequence": "Y-X-Z",
        "axes": ["flexion_extension", "lateral_bending", "axial_rotation"],
        "dof": 3
    },
    "neck": {
        "proximal": "torso",
        "distal": "head_neck",
        "sequence": "Y-X-Z",
        "axes": ["flexion_extension", "lateral_bending", "axial_rotation"],
        "dof": 3
    },
    
    # Left hip and lower limb
    "hip_left": {
        "proximal": "pelvis_center",
        "distal": "femur_left",
        "sequence": "Y-X-Z",
        "axes": ["flexion_extension", "abduction_adduction", "internal_external_rotation"],
        "dof": 3
    },
    "knee_left": {
        "proximal": "femur_left",
        "distal": "tibia_left",
        "sequence": "Y-X-Y",
        "axes": ["flexion_extension", "varus_valgus"],
        "dof": 2
    },
    "ankle_left": {
        "proximal": "tibia_left",
        "distal": "foot_left",
        "sequence": "Y-X-Z",
        "axes": ["plantar_dorsiflexion", "inversion_eversion"],
        "dof": 2
    },
    
    # Right hip and lower limb
    "hip_right": {
        "proximal": "pelvis_center",
        "distal": "femur_right",
        "sequence": "Y-X-Z",
        "axes": ["flexion_extension", "abduction_adduction", "internal_external_rotation"],
        "dof": 3
    },
    "knee_right": {
        "proximal": "femur_right",
        "distal": "tibia_right",
        "sequence": "Y-X-Y",
        "axes": ["flexion_extension", "varus_valgus"],
        "dof": 2
    },
    "ankle_right": {
        "proximal": "tibia_right",
        "distal": "foot_right",
        "sequence": "Y-X-Z",
        "axes": ["plantar_dorsiflexion", "inversion_eversion"],
        "dof": 2
    },
    
    # Left upper limb
    "shoulder_left": {
        "proximal": "torso",
        "distal": "humerus_left",
        "sequence": "Y-X-Z",
        "axes": ["flexion_extension", "abduction_adduction", "internal_external_rotation"],
        "dof": 3
    },
    "elbow_left": {
        "proximal": "humerus_left",
        "distal": "forearm_left",
        "sequence": "Y-X-Z",
        "axes": ["flexion_extension", "pronation_supination"],
        "dof": 2
    },
    
    # Right upper limb
    "shoulder_right": {
        "proximal": "torso",
        "distal": "humerus_right",
        "sequence": "Y-X-Z",
        "axes": ["flexion_extension", "abduction_adduction", "internal_external_rotation"],
        "dof": 3
    },
    "elbow_right": {
        "proximal": "humerus_right",
        "distal": "forearm_right",
        "sequence": "Y-X-Z",
        "axes": ["flexion_extension", "pronation_supination"],
        "dof": 2
    },
}

# All possible segments in source data
ALL_SEGMENTS = [
    "pelvis_center", "pelvis_width",
    "femur_left", "femur_right",
    "tibia_left", "tibia_right",
    "foot_left", "foot_right",
    "torso", "head_neck",
    "shoulder_width",
    "humerus_left", "humerus_right",
    "forearm_left", "forearm_right",
    "heel_left", "heel_right",
    "stride_left", "stride_right",
    "contact_point_left", "contact_point_right"
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_json(filepath):
    """Load JSON data."""
    with open(filepath, 'r') as f:
        return json.load(f)


def convert_coords(point):
    """
    Convert MediaPipe coordinates to biomechanics convention.
    Flips Y and Z axes: MediaPipe Y positive = down, Z negative = toward camera.
    """
    if point is None:
        return None
    return [point[0], -point[1], -point[2]]


def extract_segment_positions(data, segment_name):
    """
    Extract start and end point positions for a segment across all frames.
    Returns: dict con start_points, end_points, lengths per frame
    """
    start_points = []
    end_points = []
    lengths = []
    
    for frame in data['frames']:
        seg = frame['segments'].get(segment_name)
        if seg and 'start_point' in seg and 'end_point' in seg:
            start = seg['start_point']
            end = seg['end_point']
            length = seg.get('length_m', 0)
            
            converted_start = convert_coords(start)
            converted_end = convert_coords(end)
            
            # Verify points are valid (not [0,0,0])
            if converted_start and any(abs(x) > 1e-6 for x in converted_start):
                start_points.append(converted_start)
            else:
                start_points.append(None)
                
            if converted_end and any(abs(x) > 1e-6 for x in converted_end):
                end_points.append(converted_end)
            else:
                end_points.append(None)
                
            lengths.append(length if length else 0)
        else:
            start_points.append(None)
            end_points.append(None)
            lengths.append(0)
    
    return {
        'start_points': start_points,
        'end_points': end_points,
        'lengths': lengths
    }


def build_local_frame(start_point, end_point):
    """
    Build rotation matrix from segment start/end points.
    Local frame: y-axis = segment direction, x/z = orthogonal
    """
    direction = np.array(end_point) - np.array(start_point)
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm < 1e-6:
        return None  # Invalid segment
    
    y = direction / direction_norm
    
    # Build orthogonal axes
    if abs(y[0]) < 0.9:
        ref = np.array([1, 0, 0])
    else:
        ref = np.array([0, 1, 0])
    
    x = np.cross(y, ref)
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-6:
        ref = np.array([0, 0, 1])
        x = np.cross(y, ref)
        x_norm = np.linalg.norm(x)
    
    x = x / x_norm
    z = np.cross(x, y)
    
    # Rotation matrix (columns are x, y, z axes)
    R = np.column_stack([x, y, z])
    
    return R


def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    r = Rotation.from_matrix(R)
    return r.as_quat()  # Returns [x, y, z, w]


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    r = Rotation.from_quat(q)  # Expects [x, y, z, w]
    return r.as_matrix()


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions: q1 * q2
    q = [w, x, y, z]
    """
    w1, v1 = q1[3], q1[:3]
    w2, v2 = q2[3], q2[:3]
    
    w = w1 * w2 - np.dot(v1, v2)
    v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
    
    return np.concatenate([v, [w]])


def quaternion_inverse(q):
    """Quaternion inverse = conjugate."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quaternion_to_euler(q, sequence="Y-X-Z"):
    """
    Convert quaternion to Euler angles (degrees).
    sequence: e.g., "Y-X-Z", "Y-X-Y"
    """
    r = Rotation.from_quat(q)  # [x, y, z, w]
    
    # Map sequence to scipy rotation conventions
    seq_map = {
        "Y-X-Z": "YXZ",
        "Y-X-Y": "YXY",
        "Z-Y-X": "ZYX",
        "X-Y-Z": "XYZ",
    }
    
    euler = r.as_euler(seq_map.get(sequence, "YXZ"), degrees=True)
    return euler


def smooth_data(data, sigma=2.0):
    """Apply Gaussian smoothing to reduce noise before differentiation."""
    return gaussian_filter1d(data, sigma=sigma, axis=0)


def quaternion_derivative(q, dt):
    """
    Compute angular velocity from quaternion time series using vectorized operations.
    q: array of shape (n, 4) with [x, y, z, w] quaternions
    Returns omega: array of shape (n, 3) with angular velocity vectors (rad/s)
    """
    n = len(q)
    omega = np.zeros((n, 3))

    # Vectorized finite differences over all frames at once
    q_dot = np.diff(q, axis=0) / dt          # shape (n-1, 4)
    q_curr = q[:-1]                           # shape (n-1, 4)

    # Quaternion conjugate: negate x,y,z; keep w
    q_conj = q_curr.copy()
    q_conj[:, :3] *= -1

    v_dot  = q_dot[:, :3]   # imaginary part of derivative
    w_dot  = q_dot[:, 3]    # real part of derivative
    v_conj = q_conj[:, :3]
    w_conj = q_conj[:, 3]

    # omega = 2 * (v_dot * w_conj + w_dot * v_conj + cross(v_dot, v_conj))
    omega[1:] = 2 * (
        v_dot * w_conj[:, np.newaxis]
        + v_conj * w_dot[:, np.newaxis]
        + np.cross(v_dot, v_conj)
    )

    return omega


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def auto_detect_available_segments(summary_data):
    """
    Automatically detect segments with coverage > 0%.
    """
    available = []
    for seg in ALL_SEGMENTS:
        coverage = summary_data.get('coverage_stats', {}).get(seg, {}).get('coverage_pct', 0)
        if coverage > 0:
            available.append(seg)
    return available


def compute_available_joints(available_segments, joint_definitions):
    """
    Compute which joints can be calculated based on available segments.
    """
    available_joints = {}
    
    for joint_name, joint_def in joint_definitions.items():
        proximal = joint_def["proximal"]
        distal = joint_def["distal"]
        
        if proximal in available_segments and distal in available_segments:
            available_joints[joint_name] = joint_def
    
    return available_joints


def compute_segment_quaternions(frames_data, available_segments):
    """
    Compute quaternions for all available segments.
    """
    n_frames = len(frames_data['frames'])
    segment_data = {}
    
    for seg_name in available_segments:
        pos_data = extract_segment_positions(frames_data, seg_name)
        start_points = pos_data['start_points']
        end_points = pos_data['end_points']
        lengths = pos_data['lengths']
        
        starts_array = []
        ends_array = []
        midpoints_array = []
        quaternions_array = []
        
        for i in range(n_frames):
            start = start_points[i]
            end = end_points[i]
            
            if start is None or end is None:
                starts_array.append([0, 0, 0])
                ends_array.append([0, 0, 0])
                midpoints_array.append([0, 0, 0])
                quaternions_array.append([1, 0, 0, 0])  # identity quaternion
                continue
            
            # Calculate midpoint
            mid = [(start[j] + end[j]) / 2 for j in range(3)]
            
            # Calculate quaternion from segment direction
            R = build_local_frame(start, end)
            if R is not None:
                q = rotation_matrix_to_quaternion(R)
            else:
                q = [1, 0, 0, 0]  # identity
            
            starts_array.append(start)
            ends_array.append(end)
            midpoints_array.append(mid)
            quaternions_array.append(q)
        
        # Convert to numpy arrays
        segment_data[seg_name] = {
            'start_points': np.array(starts_array),
            'end_points': np.array(ends_array),
            'midpoints': np.array(midpoints_array),
            'quaternions': np.array(quaternions_array),
            'lengths': np.array(lengths)
        }
    
    return segment_data


def compute_joint_angles(segment_data, available_joints, dt):
    """
    Compute joint angles (Euler) for all available joints.
    """
    joint_angles = {}
    
    for joint_name, joint_def in available_joints.items():
        proximal = joint_def["proximal"]
        distal = joint_def["distal"]
        sequence = joint_def["sequence"]
        axes = joint_def["axes"]
        
        if proximal not in segment_data or distal not in segment_data:
            continue
        
        q_proximal = segment_data[proximal]['quaternions']
        q_distal = segment_data[distal]['quaternions']
        
        n_frames = len(q_proximal)
        angles = {axis: np.zeros(n_frames) for axis in axes}
        
        for i in range(n_frames):
            if np.allclose(q_proximal[i], [0, 0, 0, 1]) or np.allclose(q_distal[i], [0, 0, 0, 1]):
                # Invalid quaternion, skip
                continue
            
            # Relative quaternion: q_proximal_inv * q_distal
            q_prox_inv = quaternion_inverse(q_proximal[i])
            q_rel = quaternion_multiply(q_prox_inv, q_distal[i])
            
            # Convert to Euler angles
            euler = quaternion_to_euler(q_rel, sequence)
            
            # Fill angles based on DOF
            for j, axis in enumerate(axes):
                if j < len(euler):
                    angles[axis][i] = euler[j]
        
        joint_angles[joint_name] = angles
    
    return joint_angles


def compute_angular_derivatives(segment_data, dt):
    """
    Compute angular velocity and acceleration for each segment.
    """
    angular_velocity = {}
    angular_acceleration = {}
    
    for seg_name, seg_data in segment_data.items():
        q = seg_data['quaternions']
        
        # Angular velocity from quaternion derivative
        omega = quaternion_derivative(q, dt)
        omega = smooth_data(omega, sigma=1.0)
        
        # Angular acceleration
        alpha = np.gradient(omega, dt, axis=0)
        alpha = smooth_data(alpha, sigma=1.0)
        
        angular_velocity[seg_name] = omega
        angular_acceleration[seg_name] = alpha
    
    return angular_velocity, angular_acceleration


def export_c3d(segment_data, joint_angles, angular_velocity, angular_acceleration,
                metadata, output_path):
    """
    Export data to C3D format.
    """
    if not EZC3D_AVAILABLE:
        print("Warning: ezc3d not available, skipping C3D export")
        return False
    
    try:
        n_frames = metadata['n_frames']
        fps = metadata['fps']
        
        # Build point labels - only segments (midpoints)
        point_labels = list(segment_data.keys())
        n_points = len(point_labels)
        
        # Build analog labels for angular velocity/acceleration
        analog_labels = []
        for seg_name in segment_data:
            for axis in ['x', 'y', 'z']:
                analog_labels.append(f"{seg_name}_omega_{axis}")
            for axis in ['x', 'y', 'z']:
                analog_labels.append(f"{seg_name}_alpha_{axis}")
        
        n_analogs = len(analog_labels)
        
        # Create C3D structure
        c3d = ezc3d.c3d()
        
        # Add point data (midpoints only)
        points_data = np.zeros((4, n_points, n_frames))
        for i, seg_name in enumerate(point_labels):
            points_data[:3, i, :] = segment_data[seg_name]['midpoints'].T
            points_data[3, i, :] = 1.0  # residual
        
        # Add parameters first
        c3d.add_parameter('POINT', 'LABELS', point_labels)
        c3d.add_parameter('POINT', 'USED', n_points)
        c3d.add_parameter('POINT', 'RATE', float(fps))
        
        c3d.add_parameter('ANALOG', 'LABELS', analog_labels)
        c3d.add_parameter('ANALOG', 'USED', n_analogs)
        c3d.add_parameter('ANALOG', 'RATE', float(fps))
        
        # Add data
        c3d['data']['points'] = points_data
        
        # Add analog data
        analogs_data = np.zeros((1, n_analogs, n_frames))
        analog_idx = 0
        for seg_name in segment_data:
            omega = angular_velocity.get(seg_name, np.zeros((n_frames, 3)))
            alpha = angular_acceleration.get(seg_name, np.zeros((n_frames, 3)))
            
            for axis in range(3):
                analogs_data[0, analog_idx, :] = omega[:, axis]
                analog_idx += 1
            for axis in range(3):
                analogs_data[0, analog_idx, :] = alpha[:, axis]
                analog_idx += 1
        
        c3d['data']['analogs'] = analogs_data
        
        # Write C3D
        c3d.write(output_path)
        print(f"C3D saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Warning: C3D export failed: {e}")
        print("Continuing with JSON/CSV export...")
        return False


def export_json(segment_data, joint_angles, angular_velocity, angular_acceleration,
                metadata, available_segments, available_joints, output_path):
    """
    Export data to JSON format.
    """
    output_data = {
        "metadata": {
            "session": metadata.get('session', 'unknown'),
            "fps": metadata['fps'],
            "n_frames": metadata['n_frames'],
            "duration_seconds": metadata['n_frames'] / metadata['fps'],
            "view": metadata.get('view', 'unknown'),
            "available_segments": available_segments,
            "computed_joints": list(available_joints.keys())
        },
        "segments": {},
        "joints": {}
    }
    
    # Add segment data
    for seg_name, seg_data in segment_data.items():
        omega = angular_velocity.get(seg_name)
        alpha = angular_acceleration.get(seg_name)
        
        output_data["segments"][seg_name] = {
            "start_points": seg_data['start_points'].tolist(),
            "end_points": seg_data['end_points'].tolist(),
            "midpoints": seg_data['midpoints'].tolist(),
            "quaternions": seg_data['quaternions'].tolist(),
            "angular_velocity": omega.tolist() if omega is not None else [],
            "angular_acceleration": alpha.tolist() if alpha is not None else [],
            "lengths": seg_data['lengths'].tolist()
        }
    
    # Add joint angles (convert numpy arrays to lists)
    for joint_name, angles in joint_angles.items():
        output_data["joints"][joint_name] = {}
        for axis, values in angles.items():
            output_data["joints"][joint_name][axis] = values.tolist()
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"JSON saved: {output_path}")


def process_session(session_dir):
    """
    Main processing function for a session.
    """
    print(f"\n{'='*60}")
    print(f"Processing session: {session_dir}")
    print(f"{'='*60}")
    
    # Load data
    frames_path = os.path.join(session_dir, "segments_frames.json")
    summary_path = os.path.join(session_dir, "segments_summary.json")
    
    if not os.path.exists(frames_path):
        print(f"Error: {frames_path} not found")
        return None
    
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found")
        return None
    
    print("Loading JSON files...")
    frames_data = load_json(frames_path)
    summary_data = load_json(summary_path)
    
    # Get metadata
    fps = summary_data.get('metadata', {}).get('fps', 30.0)
    n_frames = len(frames_data['frames'])
    view = summary_data.get('metadata', {}).get('view', 'unknown')
    session_name = os.path.basename(session_dir)
    
    metadata = {
        'session': session_name,
        'fps': fps,
        'n_frames': n_frames,
        'view': view
    }
    
    print(f"FPS: {fps}, Frames: {n_frames}, View: {view}")
    
    # Auto-detect available segments
    print("\nAuto-detecting available segments...")
    available_segments = auto_detect_available_segments(summary_data)
    print(f"Found {len(available_segments)} segments with coverage > 0%:")
    for seg in available_segments:
        coverage = summary_data.get('coverage_stats', {}).get(seg, {}).get('coverage_pct', 0)
        print(f"  - {seg}: {coverage:.1f}%")
    
    # Compute available joints
    print("\nComputing available joints...")
    available_joints = compute_available_joints(available_segments, JOINT_DEFINITIONS)
    print(f"Found {len(available_joints)} computable joints:")
    for joint_name, joint_def in available_joints.items():
        print(f"  - {joint_name}: {joint_def['dof']} DOF ({', '.join(joint_def['axes'])})")
    
    # Compute segment quaternions
    print("\nComputing segment quaternions...")
    segment_data = compute_segment_quaternions(frames_data, available_segments)
    
    # Smooth quaternions
    print("Smoothing data...")
    for seg_name in segment_data:
        segment_data[seg_name]['quaternions'] = smooth_data(
            segment_data[seg_name]['quaternions'], sigma=1.5
        )
    
    # Compute angular derivatives
    print("Computing angular velocity and acceleration...")
    dt = 1.0 / fps
    angular_velocity, angular_acceleration = compute_angular_derivatives(segment_data, dt)
    
    # Compute joint angles
    print("Computing joint angles (ISB)...")
    joint_angles = compute_joint_angles(segment_data, available_joints, dt)
    
    # Export outputs
    output_dir = f"{session_name}_processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export C3D
    c3d_path = os.path.join(output_dir, f"{session_name}_generic.c3d")
    export_c3d(segment_data, joint_angles, angular_velocity, angular_acceleration,
               metadata, c3d_path)
    
    # Export JSON
    json_path = os.path.join(output_dir, f"{session_name}_generic.json")
    export_json(segment_data, joint_angles, angular_velocity, angular_acceleration,
                metadata, available_segments, available_joints, json_path)
    
    # Export CSV for each segment
    print("\nExporting CSV files...")
    time_arr = np.arange(n_frames) * dt
    
    for seg_name, seg_data in segment_data.items():
        csv_path = os.path.join(output_dir, f"{session_name}_{seg_name}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame", "time",
                "start_x", "start_y", "start_z",
                "end_x", "end_y", "end_z",
                "mid_x", "mid_y", "mid_z",
                "quat_w", "quat_x", "quat_y", "quat_z",
                "omega_x", "omega_y", "omega_z",
                "alpha_x", "alpha_y", "alpha_z",
                "length"
            ])
            for i in range(n_frames):
                row = [
                    i, time_arr[i],
                    *seg_data['start_points'][i],
                    *seg_data['end_points'][i],
                    *seg_data['midpoints'][i],
                    *seg_data['quaternions'][i],
                    *angular_velocity[seg_name][i],
                    *angular_acceleration[seg_name][i],
                    seg_data['lengths'][i] if i < len(seg_data['lengths']) else 0
                ]
                writer.writerow(row)
        print(f"  CSV: {csv_path}")
    
    # Export joint angles CSV
    if joint_angles:
        joint_csv_path = os.path.join(output_dir, f"{session_name}_joints.csv")
        with open(joint_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            header = ["frame", "time"]
            for joint_name, joint_def in available_joints.items():
                for axis in joint_def['axes']:
                    header.append(f"{joint_name}_{axis}")
            writer.writerow(header)
            
            # Data
            for i in range(n_frames):
                row = [i, time_arr[i]]
                for joint_name, joint_def in available_joints.items():
                    for axis in joint_def['axes']:
                        row.append(joint_angles[joint_name][axis][i])
                writer.writerow(row)
        print(f"  Joint CSV: {joint_csv_path}")
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Session: {session_name}")
    print(f"Segments processed: {len(available_segments)}")
    print(f"Joints computed: {len(available_joints)}")
    print(f"Output directory: {output_dir}")
    
    return {
        'available_segments': available_segments,
        'available_joints': available_joints,
        'segment_data': segment_data,
        'joint_angles': joint_angles
    }


def main():
    parser = argparse.ArgumentParser(
        description='Process MediaPipe motion capture to C3D with joint angles'
    )
    parser.add_argument(
        'session',
        nargs='?',
        default=None,
        help='Session directory (default: auto-detect from current directory)'
    )
    parser.add_argument(
        '--list-sessions',
        action='store_true',
        help='List available sessions in current directory'
    )
    
    args = parser.parse_args()
    
    if args.list_sessions:
        # List available sessions
        print("Available sessions:")
        for item in os.listdir('.'):
            if os.path.isdir(item) and item.startswith('running_'):
                if os.path.exists(os.path.join(item, 'segments_frames.json')):
                    print(f"  - {item}")
        return
    
    # Determine session directory
    if args.session:
        session_dir = args.session
    else:
        # Auto-detect: look for running_* directory
        available = []
        for item in os.listdir('.'):
            if os.path.isdir(item) and item.startswith('running_'):
                if os.path.exists(os.path.join(item, 'segments_frames.json')):
                    available.append(item)
        
        if not available:
            print("Error: No session directory found")
            print("Usage: python process_session_generic.py <session_dir>")
            print("Or run from a directory containing running_* folders")
            sys.exit(1)
        
        if len(available) == 1:
            session_dir = available[0]
            print(f"Auto-selected: {session_dir}")
        else:
            print("Available sessions:")
            for i, sess in enumerate(sorted(available)):
                print(f"  {i+1}. {sess}")
            try:
                choice = int(input("Select session number: ")) - 1
                if choice < 0 or choice >= len(available):
                    print("Invalid selection")
                    sys.exit(1)
                session_dir = sorted(available)[choice]
            except ValueError:
                print("Invalid input")
                sys.exit(1)
    
    # Process session
    result = process_session(session_dir)
    
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()