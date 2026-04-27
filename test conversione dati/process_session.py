#!/usr/bin/env python3
"""
Process MediaPipe motion capture data to C3D with quaternions.
Session: running_20260420_143730
"""

import json
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d

# Try importing Kinetics Toolkit for C3D export
try:
    import kineticstoolkit.lab as ktk
    KTK_AVAILABLE = True
except ImportError:
    KTK_AVAILABLE = False
    print("Warning: Kinetics Toolkit not available. Will save as NumPy arrays instead.")

# Configuration
SESSION_DIR = "running_20260420_143730"
FPS = 30.0
DT = 1.0 / FPS

# Segmenti con coverage > 0% (dalla analisi precedente)
VALID_SEGMENTS = [
    "head_neck",           # 100%
    "shoulder_width",      # 100%
    "humerus_left",        # 99.4%
    "forearm_left",        # 97.7%
    "humerus_right",       # 73.7%
    "forearm_right",       # 73.7%
    "pelvis_center",       # 2.3%
    "pelvis_width",        # 2.3%
    "torso",               # 2.3%
]

# Skeleton hierarchy (parent -> child)
SKELETON = {
    "pelvis_center": None,      # Root
    "pelvis_width": "pelvis_center",
    "torso": "pelvis_center",
    "head_neck": "torso",
    "shoulder_width": "torso",
    "humerus_left": "shoulder_width",
    "forearm_left": "humerus_left",
    "humerus_right": "shoulder_width",
    "forearm_right": "humerus_right",
}


def load_json(filepath):
    """Load JSON data."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_segment_positions(data, segment_name):
    """
    Extract start and end point positions for a segment across all frames.
    Returns: (n_frames, 2, 3) array of start/end points
    """
    positions = []
    for frame in data['frames']:
        seg = frame['segments'].get(segment_name)
        if seg and 'start_point' in seg and 'end_point' in seg:
            start = seg['start_point']
            end = seg['end_point']
            positions.append([start, end])
        else:
            positions.append(None)
    return positions


def build_local_frame(start_point, end_point):
    """
    Build rotation matrix from segment start/end points.
    Local frame: y-axis = segment direction, x/z = orthogonal
    """
    # Direction vector (y-axis in local frame)
    direction = np.array(end_point) - np.array(start_point)
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm < 1e-6:
        return None  # Invalid segment
    
    y = direction / direction_norm
    
    # Build orthogonal axes
    # Use a reference vector to create orthogonal basis
    if abs(y[0]) < 0.9:
        ref = np.array([1, 0, 0])
    else:
        ref = np.array([0, 1, 0])
    
    # x = normalized(cross(y, ref))
    x = np.cross(y, ref)
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-6:
        # y and ref are parallel, try another reference
        ref = np.array([0, 0, 1])
        x = np.cross(y, ref)
        x_norm = np.linalg.norm(x)
    
    x = x / x_norm
    
    # z = cross(x, y)
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


def quaternion_derivative(q, dt):
    """
    Compute angular velocity from quaternion time series.
    q_dot = 0.5 * omega * q (for small angles approximation)
    Solved for omega: omega = 2 * q_dot * q_inv
    """
    n = len(q)
    omega = np.zeros((n, 3))
    
    for i in range(n - 1):
        # Quaternion derivative (finite difference)
        q_dot = (q[i + 1] - q[i]) / dt
        
        # omega = 2 * q_dot * q_conjugate
        q_conj = np.array([-q[i, 0], -q[i, 1], -q[i, 2], q[i, 3]])
        
        # Quaternion multiplication: q_dot * q_conj
        w = q_dot[3]
        v = q_dot[:3]
        w_c = q_conj[3]
        v_c = q_conj[:3]
        
        # Result: [v*w_c + w*v_c + cross(v, v_c), w*w_c - v·v_c]
        omega[i + 1] = 2 * (v * w_c + w * v_c + np.cross(v, v_c))
    
    return omega


def smooth_data(data, sigma=2.0):
    """Apply Gaussian smoothing to reduce noise before differentiation."""
    return gaussian_filter1d(data, sigma=sigma, axis=0)


def process_session():
    """Main processing function."""
    print(f"Processing session: {SESSION_DIR}")
    
    # Load data
    print("Loading JSON files...")
    frames_data = load_json(f"{SESSION_DIR}/segments_frames.json")
    summary_data = load_json(f"{SESSION_DIR}/segments_summary.json")
    
    n_frames = len(frames_data['frames'])
    print(f"Total frames: {n_frames}")
    
    # Filter valid segments
    available_segments = []
    for seg in VALID_SEGMENTS:
        coverage = summary_data['coverage_stats'].get(seg, {}).get('coverage_pct', 0)
        if coverage > 0:
            available_segments.append(seg)
            print(f"  - {seg}: {coverage:.1f}% coverage")
    
    print(f"\nProcessing {len(available_segments)} segments with valid data")
    
    # Step 1: Extract positions and build quaternions
    # Modifica: ora salviamo start_points, end_points e midpoints separatamente
    # invece di usare solo midpoints. I punti (articolazioni/giunture) sono gli
    # estremi dei segmenti, non il centro.
    segment_quaternions = {}
    segment_data = {}  # Contains start_points, end_points, midpoints
    
    for seg_name in available_segments:
        print(f"Processing segment: {seg_name}")
        
        # Extract positions
        positions = extract_segment_positions(frames_data, seg_name)
        
        # Build local frames and convert to quaternions
        quaternions = []
        starts = []
        ends = []
        midpoints = []
        
        for i, pos in enumerate(positions):
            if pos is None:
                quaternions.append(None)
                starts.append(None)
                ends.append(None)
                midpoints.append(None)
            else:
                R = build_local_frame(pos[0], pos[1])
                if R is not None:
                    q = rotation_matrix_to_quaternion(R)
                    quaternions.append(q)
                    # Start and end points as segment extremities (joint positions)
                    starts.append(pos[0])
                    ends.append(pos[1])
                    # Midpoint as marker position
                    mid = [(pos[0][j] + pos[1][j]) / 2 for j in range(3)]
                    midpoints.append(mid)
                else:
                    quaternions.append(None)
                    starts.append(None)
                    ends.append(None)
                    midpoints.append(None)
        
        # Convert to arrays, handle missing data
        # Identity quaternion [0,0,0,1] = w,x,y,z for missing frames
        q_array = np.array([q if q is not None else [0, 0, 0, 1] for q in quaternions])
        starts_array = np.array([s if s is not None else [0, 0, 0] for s in starts])
        ends_array = np.array([e if e is not None else [0, 0, 0] for e in ends])
        mid_array = np.array([m if m is not None else [0, 0, 0] for m in midpoints])
        
        # Store all three point types
        segment_quaternions[seg_name] = q_array
        segment_data[seg_name] = {
            'start_points': starts_array,
            'end_points': ends_array,
            'midpoints': mid_array
        }
    
    # Step 2: Smooth data before differentiation
    print("\nSmoothing data...")
    for seg_name in segment_quaternions:
        segment_quaternions[seg_name] = smooth_data(segment_quaternions[seg_name], sigma=1.5)
    
    # Step 3: Compute angular velocity and acceleration
    print("Computing angular velocity and acceleration...")
    segment_angular_velocity = {}
    segment_angular_accel = {}
    
    for seg_name in segment_quaternions:
        q = segment_quaternions[seg_name]
        
        # Angular velocity from quaternion derivative
        omega = quaternion_derivative(q, DT)
        omega = smooth_data(omega, sigma=1.0)
        
        # Angular acceleration
        alpha = np.gradient(omega, DT, axis=0)
        alpha = smooth_data(alpha, sigma=1.0)
        
        segment_angular_velocity[seg_name] = omega
        segment_angular_accel[seg_name] = alpha
    
    # Step 4: Compute joint angles (relative quaternions)
    print("Computing joint angles...")
    joint_angles = {}
    
    for seg_name, parent in SKELETON.items():
        if seg_name not in segment_quaternions:
            continue
            
        q_child = segment_quaternions[seg_name]
        
        if parent is None:
            # Root segment - use absolute orientation
            joint_angles[seg_name] = q_child
        elif parent in segment_quaternions:
            # Relative orientation: q_parent_inv * q_child
            q_parent = segment_quaternions[parent]
            
            # Compute relative quaternion
            q_rel = np.zeros_like(q_child)
            for i in range(len(q_child)):
                # q_rel = q_parent^-1 * q_child
                q_p = q_parent[i]
                q_c = q_child[i]
                
                # Quaternion inverse = conjugate (for unit quaternions)
                q_p_inv = np.array([-q_p[0], -q_p[1], -q_p[2], q_p[3]])
                
                # Multiplication
                w1, v1 = q_p_inv[3], q_p_inv[:3]
                w2, v2 = q_c[3], q_c[:3]
                
                q_rel[i] = np.concatenate([
                    [w1 * w2 - np.dot(v1, v2)],
                    w1 * v2 + w2 * v1 + np.cross(v1, v2)
                ])
            
            joint_angles[seg_name] = q_rel
    
    # Step 5: Prepare output data
    print("\nPreparing output...")
    
    # Create time vector
    time = np.arange(n_frames) * DT
    output_dir = f"{SESSION_DIR}_processed"
    
    # Try C3D export using ezc3d directly
    # Modifica: usiamo midpoints per C3D (formato standard usa un punto per segmento)
    c3d_success = False
    
    if KTK_AVAILABLE:
        try:
            import ezc3d
            
            # Prepare point data - usando midpoints per C3D
            point_labels = list(segment_data.keys())
            n_points = len(point_labels)
            
            # Build points array: (3, n_points, n_frames)
            points_array = np.zeros((3, n_points, n_frames))
            for i, seg_name in enumerate(point_labels):
                pos = segment_data[seg_name]['midpoints']  # (n_frames, 3) - usiamo midpoints
                points_array[0, i, :] = pos[:, 0]  # x
                points_array[1, i, :] = pos[:, 1]  # y
                points_array[2, i, :] = pos[:, 2]  # z
            
            # Analog data: angular velocity + acceleration
            analog_labels = []
            analog_data_list = []
            for seg_name in segment_angular_velocity:
                omega = segment_angular_velocity[seg_name]
                alpha = segment_angular_accel[seg_name]
                analog_labels.extend([
                    f"{seg_name}_omega_x", f"{seg_name}_omega_y", f"{seg_name}_omega_z",
                    f"{seg_name}_alpha_x", f"{seg_name}_alpha_y", f"{seg_name}_alpha_z"
                ])
                analog_data_list.append(np.column_stack([omega, alpha]))
            
            n_analogs = len(analog_labels)
            analog_array = np.hstack(analog_data_list)
            
            # ezc3d structure - start with empty and fill data
            c3d = ezc3d.c3d()
            
            # Set parameters - RATE must be float
            c3d.add_parameter('POINT', 'LABELS', point_labels)
            c3d.add_parameter('POINT', 'USED', n_points)
            c3d.add_parameter('POINT', 'RATE', float(FPS))
            
            c3d.add_parameter('ANALOG', 'LABELS', analog_labels)
            c3d.add_parameter('ANALOG', 'USED', n_analogs)
            c3d.add_parameter('ANALOG', 'RATE', float(FPS))
            
            # Store data - already (3, n_points, n_frames)
            c3d['data']['points'] = points_array
            
            # Store analog data (n_analogs, n_frames)
            c3d['data']['analogs'] = analog_array.T.reshape(1, n_analogs, n_frames)
            
            # Write C3D
            output_path = f"{output_dir}.c3d"
            print(f"Writing C3D file: {output_path}")
            c3d.write(output_path)
            print(f"C3D file saved: {output_path}")
            c3d_success = True
            
        except Exception as e:
            import traceback
            print(f"Note: C3D export failed ({e}). Using JSON/CSV format instead.")
            traceback.print_exc()
    
    # Fallback: Save as JSON (universal, portable)
    # Modifica: ora salviamo start_points, end_points e midpoints separatamente
    if not c3d_success:
        print(f"Saving as JSON and CSV formats...")
        
        # Save comprehensive JSON
        output_data = {
            "metadata": {
                "session": SESSION_DIR,
                "fps": FPS,
                "n_frames": n_frames,
                "duration_seconds": n_frames * DT,
            },
            "segments": {},
            "angular_velocity": {},
            "angular_acceleration": {},
        }
        
        # Modifica: includiamo start_points, end_points e midpoints invece di un singolo positions
        for seg_name in segment_data:
            output_data["segments"][seg_name] = {
                "start_points": segment_data[seg_name]['start_points'].tolist(),
                "end_points": segment_data[seg_name]['end_points'].tolist(),
                "midpoints": segment_data[seg_name]['midpoints'].tolist(),
                "quaternions": segment_quaternions[seg_name].tolist(),
                "angular_velocity": segment_angular_velocity[seg_name].tolist(),
                "angular_acceleration": segment_angular_accel[seg_name].tolist(),
            }
        
        # Save JSON
        json_path = f"{output_dir}.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f)
        print(f"JSON saved: {json_path}")
        
        # Save time series as CSV (one file per segment)
        # Modifica: includiamo start_points, end_points e midpoints invece di un singolo pos
        import csv
        for seg_name in segment_data:
            csv_path = f"{output_dir}_{seg_name}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header - now includes start, end, and mid points
                writer.writerow([
                    "time", 
                    "start_x", "start_y", "start_z",
                    "end_x", "end_y", "end_z",
                    "mid_x", "mid_y", "mid_z",
                    "quat_w", "quat_x", "quat_y", "quat_z",
                    "omega_x", "omega_y", "omega_z",
                    "alpha_x", "alpha_y", "alpha_z"
                ])
                # Data
                for i in range(n_frames):
                    row = [
                        time[i],
                        *segment_data[seg_name]['start_points'][i],
                        *segment_data[seg_name]['end_points'][i],
                        *segment_data[seg_name]['midpoints'][i],
                        *segment_quaternions[seg_name][i],
                        *segment_angular_velocity[seg_name][i],
                        *segment_angular_accel[seg_name][i],
                    ]
                    writer.writerow(row)
            print(f"CSV saved: {csv_path}")
    
    # Summary
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Session: {SESSION_DIR}")
    print(f"Frames: {n_frames}")
    print(f"FPS: {FPS}")
    print(f"Segments processed: {len(available_segments)}")
    print(f"Output format: {'.c3d' if c3d_success else 'JSON + CSV'}")
    print(f"\nOutput files:")
    if c3d_success:
        print(f"  - {output_dir}.c3d")
    else:
        print(f"  - {output_dir}.json")
        for seg in available_segments:
            print(f"  - {output_dir}_{seg}.csv")
    
    # Modifica: restituiamo anche start_points e end_points oltre a midpoints
    return {
        'start_points': {seg: segment_data[seg]['start_points'] for seg in segment_data},
        'end_points': {seg: segment_data[seg]['end_points'] for seg in segment_data},
        'midpoints': {seg: segment_data[seg]['midpoints'] for seg in segment_data},
        'quaternions': segment_quaternions,
        'angular_velocity': segment_angular_velocity,
        'angular_acceleration': segment_angular_accel,
    }


if __name__ == "__main__":
    process_session()
