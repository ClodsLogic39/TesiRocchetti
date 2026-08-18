#!/usr/bin/env python3
"""
Converte i dati motion-capture MediaPipe in C3D con quaternioni e angoli articolari.

Versione generica: rileva automaticamente i segmenti disponibili e calcola tutte le
articolazioni possibili. Output canonico = <sessione>_generic.json (+ CSV e C3D).

Sessione: una cartella running_* con segments_frames.json e segments_summary.json.
"""

import csv
import json
import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import gaussian_filter1d

# Import opzionale di ezc3d per l'esportazione C3D
try:
    import ezc3d
    EZC3D_AVAILABLE = True
except ImportError:
    EZC3D_AVAILABLE = False
    print("Attenzione: ezc3d non disponibile. Esporto solo JSON/CSV.")

# Repo layout: questo file vive in src/conversion/, quindi la repo-root e' due
# livelli sopra. Le sessioni grezze stanno in data/raw, gli output in data/processed.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_RAW = os.path.join(REPO_ROOT, "data", "raw")
DEFAULT_DATA_PROCESSED = os.path.join(REPO_ROOT, "data", "processed")

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])  # scipy order: [x, y, z, w]
MIN_SEGMENT_COVERAGE_PCT = 0.0
LOW_COVERAGE_WARNING_PCT = 10.0
MAX_INTERPOLATION_GAP_FRAMES = 10

# =============================================================================
# CONFIGURAZIONE - Mappatura segmento -> articolazione (standard ISB)
# =============================================================================

# Mappatura: quali segmenti formano quali articolazioni.
SEGMENT_TO_JOINT = {
    # Colonna/Bacino
    "pelvis_center": "pelvis",
    "pelvis_width": "pelvis",
    "torso": "spine",
    "head_neck": "neck",
    "shoulder_width": "shoulder_center",

    # Arto inferiore sinistro
    "femur_left": "hip_left",
    "tibia_left": "knee_left",
    "foot_left": "ankle_left",
    "heel_left": "heel_left",
    "stride_left": "stride_left",
    "contact_point_left": "contact_left",

    # Arto inferiore destro
    "femur_right": "hip_right",
    "tibia_right": "knee_right",
    "foot_right": "ankle_right",
    "heel_right": "heel_right",
    "stride_right": "stride_right",
    "contact_point_right": "contact_right",

    # Arto superiore sinistro
    "humerus_left": "shoulder_left",
    "forearm_left": "elbow_left",

    # Arto superiore destro
    "humerus_right": "shoulder_right",
    "forearm_right": "elbow_right",
}

# Definizioni delle articolazioni: segmento prossimale -> segmento distale.
#
# NOTA: i campi "sequence" e "dof" descrivono la vecchia convenzione Euler ed
# esistono solo per compatibilita': il calcolo attuale degli angoli e'
# direction-based (angolo tra direzioni dei segmenti proiettate sui piani
# globali), quindi queste sequenze NON vengono piu' usate. Vengono conservate
# come riferimento storico delle scelte di asse (vedi quaternion_to_euler, legacy).
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
        "sequence": "Z-X-Y",
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
        "sequence": "Z-X-Y",
        "axes": ["plantar_dorsiflexion", "inversion_eversion"],
        "dof": 2
    },
    
    # Right hip and lower limb
    "hip_right": {
        "proximal": "pelvis_center",
        "distal": "femur_right",
        "sequence": "Z-X-Y",
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
        "sequence": "Z-X-Y",
        "axes": ["plantar_dorsiflexion", "inversion_eversion"],
        "dof": 2
    },
    
    # Left upper limb
    "shoulder_left": {
        "proximal": "torso",
        "distal": "humerus_left",
        "sequence": "Z-X-Y",
        "axes": ["flexion_extension", "abduction_adduction", "internal_external_rotation"],
        "dof": 3
    },
    "elbow_left": {
        "proximal": "humerus_left",
        "distal": "forearm_left",
        "sequence": "Z-X-Y",
        "axes": ["flexion_extension", "pronation_supination"],
        "dof": 2
    },
    
    # Right upper limb
    "shoulder_right": {
        "proximal": "torso",
        "distal": "humerus_right",
        "sequence": "Z-X-Y",
        "axes": ["flexion_extension", "abduction_adduction", "internal_external_rotation"],
        "dof": 3
    },
    "elbow_right": {
        "proximal": "humerus_right",
        "distal": "forearm_right",
        "sequence": "Z-X-Y",
        "axes": ["flexion_extension", "pronation_supination"],
        "dof": 2
    },
}

# Mappatura degli estremi usata per esportare in C3D centri articolari
# anatomicamente sensati. L'orientamento start/end e' dedotto dalla semantica
# dei segmenti grezzi:
# - shoulder_width.start == spalla sinistra, shoulder_width.end == spalla destra
# - pelvis_center.start == anca sinistra, pelvis_center.end == anca destra
# ATTENZIONE: su una ripresa frontale questa assunzione sinistra/destra non e'
# verificata (rischio di specchiamento); validare su un frame noto.
JOINT_CENTER_ENDPOINTS = {
    "spine": [("torso", "start"), ("pelvis_center", "mid")],
    "neck": [("torso", "end"), ("head_neck", "start"), ("shoulder_width", "mid")],
    "hip_left": [("pelvis_center", "start"), ("femur_left", "start")],
    "knee_left": [("femur_left", "end"), ("tibia_left", "start")],
    "ankle_left": [("tibia_left", "end"), ("foot_left", "start")],
    "hip_right": [("pelvis_center", "end"), ("femur_right", "start")],
    "knee_right": [("femur_right", "end"), ("tibia_right", "start")],
    "ankle_right": [("tibia_right", "end"), ("foot_right", "start")],
    "shoulder_left": [("shoulder_width", "start"), ("humerus_left", "start")],
    "elbow_left": [("humerus_left", "end"), ("forearm_left", "start")],
    "shoulder_right": [("shoulder_width", "end"), ("humerus_right", "start")],
    "elbow_right": [("humerus_right", "end"), ("forearm_right", "start")],
}

# Tutti i segmenti possibili nei dati di origine.
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
# FUNZIONI DI SUPPORTO
# =============================================================================

def load_json(filepath):
    """Carica dati da un file JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def is_valid_point(point):
    """Un punto valido deve essere finito e diverso dal sentinella [0, 0, 0]."""
    if point is None:
        return False
    p = np.asarray(point, dtype=float)
    return p.shape == (3,) and np.all(np.isfinite(p)) and not np.allclose(p, [0, 0, 0], atol=1e-6)


def is_finite_row(arr_2d):
    return np.all(np.isfinite(arr_2d), axis=1)


def get_segment_anchor_series(segment_data, segment_name, anchor):
    if segment_name not in segment_data:
        return None
    seg = segment_data[segment_name]
    if anchor == "start":
        return seg['start_points']
    if anchor == "end":
        return seg['end_points']
    if anchor == "mid":
        return seg['midpoints']
    return None


def interpolate_small_gaps_1d(values, max_gap_frames=MAX_INTERPOLATION_GAP_FRAMES):
    """Interpola solo i brevi gap interni di NaN, lasciando NaN i gap grandi e i bordi."""
    out = np.asarray(values, dtype=float).copy()
    n = out.shape[0]
    valid = np.isfinite(out)
    if valid.sum() < 2:
        return out

    idx = np.arange(n)
    interp_all = np.interp(idx, idx[valid], out[valid])
    missing_idx = np.where(~valid)[0]
    if missing_idx.size == 0:
        return out

    run_start = missing_idx[0]
    prev = missing_idx[0]
    for curr in missing_idx[1:]:
        if curr != prev + 1:
            run_end = prev
            run_len = run_end - run_start + 1
            if run_start > 0 and run_end < n - 1 and run_len <= max_gap_frames:
                out[run_start:run_end + 1] = interp_all[run_start:run_end + 1]
            run_start = curr
        prev = curr

    run_end = prev
    run_len = run_end - run_start + 1
    if run_start > 0 and run_end < n - 1 and run_len <= max_gap_frames:
        out[run_start:run_end + 1] = interp_all[run_start:run_end + 1]

    return out


def interpolate_small_gaps_2d(values, max_gap_frames=MAX_INTERPOLATION_GAP_FRAMES):
    out = np.asarray(values, dtype=float).copy()
    for axis in range(out.shape[1]):
        out[:, axis] = interpolate_small_gaps_1d(out[:, axis], max_gap_frames=max_gap_frames)
    return out


def interpolate_small_gaps_quaternions(quaternions, max_gap_frames=MAX_INTERPOLATION_GAP_FRAMES):
    """Interpola i brevi gap di NaN tramite SLERP su quaternioni unitari."""
    q = np.asarray(quaternions, dtype=float).copy()
    n = q.shape[0]
    valid = is_finite_row(q)
    if valid.sum() < 2:
        return q

    missing_idx = np.where(~valid)[0]
    if missing_idx.size == 0:
        return q

    def fill_gap(start_idx, end_idx):
        run_len = end_idx - start_idx + 1
        if run_len > max_gap_frames:
            return
        if start_idx <= 0 or end_idx >= n - 1:
            return
        left = start_idx - 1
        right = end_idx + 1
        if not (valid[left] and valid[right]):
            return
        q_left = q[left]
        q_right = q[right]
        if np.dot(q_left, q_right) < 0:
            q_right = -q_right
        key_rots = Rotation.from_quat([q_left, q_right])
        slerp = Slerp([0.0, 1.0], key_rots)
        times = np.linspace(0.0, 1.0, run_len + 2)[1:-1]
        q[start_idx:end_idx + 1] = slerp(times).as_quat()

    run_start = missing_idx[0]
    prev = missing_idx[0]
    for curr in missing_idx[1:]:
        if curr != prev + 1:
            fill_gap(run_start, prev)
            run_start = curr
        prev = curr
    fill_gap(run_start, prev)

    return q


def normalize_quaternion_series(quaternions):
    q = np.asarray(quaternions, dtype=float).copy()
    finite = is_finite_row(q)
    norms = np.linalg.norm(q[finite], axis=1)
    nonzero = norms > 1e-10
    q_idx = np.where(finite)[0]
    q[q_idx[nonzero]] /= norms[nonzero, np.newaxis]
    q[q_idx[~nonzero]] = np.nan
    return q


def enforce_quaternion_continuity(quaternions):
    """Inverte i segni frame-per-frame per mantenere la traiettoria di quaternioni a percorso minimo."""
    q = np.asarray(quaternions, dtype=float).copy()
    valid_idx = np.where(is_finite_row(q))[0]
    if len(valid_idx) < 2:
        return q

    prev = valid_idx[0]
    for idx in valid_idx[1:]:
        if np.dot(q[prev], q[idx]) < 0:
            q[idx] *= -1.0
        prev = idx
    return q


def smooth_vector_series(values, sigma=1.0):
    """Liscia i tratti contigui finiti senza scavalcare i lunghi gap di NaN."""
    out = np.asarray(values, dtype=float).copy()
    n, d = out.shape
    for axis in range(d):
        col = out[:, axis]
        valid = np.isfinite(col)
        i = 0
        while i < n:
            if not valid[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and valid[j + 1]:
                j += 1
            if (j - i + 1) >= 3:
                out[i:j + 1, axis] = gaussian_filter1d(col[i:j + 1], sigma=sigma, axis=0, mode='nearest')
            i = j + 1
    return out


def wrap_degrees(values):
    return (values + 180.0) % 360.0 - 180.0


def unwrap_angle_series(values, degrees=True, wrap=True):
    """Srotola gli angoli sui tratti contigui finiti per evitare i salti di ±180°."""
    out = np.asarray(values, dtype=float).copy()
    if degrees:
        radians = np.deg2rad(out)
    else:
        radians = out
    valid = np.isfinite(radians)
    n = radians.shape[0]
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and valid[j + 1]:
            j += 1
        radians[i:j + 1] = np.unwrap(radians[i:j + 1])
        i = j + 1
    unwrapped = np.rad2deg(radians) if degrees else radians
    if degrees and wrap:
        return wrap_degrees(unwrapped)
    return unwrapped


def convert_coords(point):
    """
    Converte le coordinate MediaPipe nella convenzione biomeccanica.
    Ribalta gli assi Y e Z: in MediaPipe Y positivo = giu', Z negativo = verso la camera.
    """
    if point is None:
        return None
    return [point[0], -point[1], -point[2]]


def extract_segment_positions(data, segment_name):
    """
    Estrae le posizioni dei punti start ed end di un segmento su tutti i frame.
    Restituisce: dict con start_points, end_points e lengths per frame.
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
            
            if is_valid_point(converted_start):
                start_points.append(converted_start)
            else:
                start_points.append(None)
                
            if is_valid_point(converted_end):
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


def build_local_frame(start_point, end_point, previous_x=None, preferred_up=np.array([0.0, 1.0, 0.0])):
    """
    Costruisce la matrice di rotazione dai punti start/end del segmento.
    Terna locale: asse y = direzione del segmento, x/z ortogonali.
    """
    direction = np.array(end_point) - np.array(start_point)
    direction_norm = np.linalg.norm(direction)

    if direction_norm < 1e-6:
        return None  # Segmento non valido
    
    y = direction / direction_norm

    x = None
    if previous_x is not None and np.all(np.isfinite(previous_x)):
        prev_x = np.asarray(previous_x, dtype=float)
        prev_x_proj = prev_x - np.dot(prev_x, y) * y
        proj_norm = np.linalg.norm(prev_x_proj)
        if proj_norm > 1e-6:
            x = prev_x_proj / proj_norm

    if x is None:
        candidates = [preferred_up, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])]
        for ref in candidates:
            x_candidate = np.cross(y, ref)
            x_norm = np.linalg.norm(x_candidate)
            if x_norm > 1e-6:
                x = x_candidate / x_norm
                break
        if x is None:
            return None, None

    z = np.cross(x, y)
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-6:
        return None, None
    z = z / z_norm
    x = np.cross(y, z)
    x = x / np.linalg.norm(x)

    R = np.column_stack([x, y, z])
    return R, x


def rotation_matrix_to_quaternion(R):
    """Converte una matrice di rotazione in quaternione (ordine SciPy [x, y, z, w])."""
    r = Rotation.from_matrix(R)
    return r.as_quat()  # Restituisce [x, y, z, w]


def quaternion_to_rotation_matrix(q):
    """Converte un quaternione in matrice di rotazione."""
    r = Rotation.from_quat(q)  # Si aspetta [x, y, z, w]
    return r.as_matrix()


def quaternion_multiply(q1, q2):
    """
    Moltiplica due quaternioni: q1 * q2.
    NOTA: questo helper usa la semantica [w, x, y, z] su q[3]/q[:3]
    (convenzione diversa dall'ordine SciPy usato altrove: leggere prima di riusare).
    """
    w1, v1 = q1[3], q1[:3]
    w2, v2 = q2[3], q2[:3]
    
    w = w1 * w2 - np.dot(v1, v2)
    v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
    
    return np.concatenate([v, [w]])


def quaternion_inverse(q):
    """Inverso del quaternione = coniugato (per quaternioni unitari, ordine [x, y, z, w])."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quaternion_to_euler(q, sequence="Y-X-Z"):
    """
    Converte un quaternione in angoli di Eulero (gradi).

    LEGACY: non piu' usata dal calcolo angoli direction-based; conservata come
    riferimento. sequence: es. "Y-X-Z", "Y-X-Y".
    """
    r = Rotation.from_quat(q)  # [x, y, z, w]

    # Mappa la sequenza sulle convenzioni di rotazione di scipy.
    seq_map = {
        "Y-X-Z": "YXZ",
        "Y-X-Y": "YXY",
        "Z-Y-X": "ZYX",
        "X-Y-Z": "XYZ",
        "Z-X-Y": "ZXY",
        "X-Z-Y": "XZY",
    }

    scipy_seq = seq_map.get(sequence)
    if scipy_seq is None:
        raise ValueError(f"Sequenza di Eulero '{sequence}' non supportata. Supportate: {list(seq_map.keys())}")
    euler = r.as_euler(scipy_seq, degrees=True)
    return euler


def smooth_data(data, sigma=2.0):
    """Applica un lisciamento gaussiano per ridurre il rumore prima della derivazione."""
    return gaussian_filter1d(data, sigma=sigma, axis=0)


def quaternion_derivative(q, dt, valid_mask):
    """
    Calcola la velocita' angolare dalla serie temporale di quaternioni.
    q: array (n, 4) di quaternioni [x, y, z, w].
    Restituisce omega: array (n, 3) di vettori di velocita' angolare (rad/s).
    """
    n = len(q)
    omega = np.full((n, 3), np.nan, dtype=float)

    for i in range(1, n):
        if not (valid_mask[i - 1] and valid_mask[i]):
            continue

        q_prev = q[i - 1]
        q_curr = q[i]
        if np.dot(q_prev, q_curr) < 0:
            q_curr = -q_curr

        r_prev = Rotation.from_quat(q_prev)
        r_curr = Rotation.from_quat(q_curr)
        delta = r_prev.inv() * r_curr
        omega[i] = delta.as_rotvec() / dt

    return omega


# =============================================================================
# FUNZIONI PRINCIPALI DI ELABORAZIONE
# =============================================================================

def auto_detect_available_segments(summary_data, min_coverage_pct=MIN_SEGMENT_COVERAGE_PCT):
    """Rileva automaticamente i segmenti con copertura >= min_coverage_pct."""
    available = []
    for seg in ALL_SEGMENTS:
        coverage = summary_data.get('coverage_stats', {}).get(seg, {}).get('coverage_pct', 0)
        if coverage > 0 and coverage >= min_coverage_pct:
            available.append(seg)
    return available


def compute_available_joints(available_segments, joint_definitions):
    """Determina quali articolazioni sono calcolabili in base ai segmenti disponibili."""
    available_joints = {}
    
    for joint_name, joint_def in joint_definitions.items():
        proximal = joint_def["proximal"]
        distal = joint_def["distal"]
        
        if proximal in available_segments and distal in available_segments:
            available_joints[joint_name] = joint_def
    
    return available_joints


def compute_segment_quaternions(frames_data, available_segments):
    """Calcola i quaternioni (per il rendering) di tutti i segmenti disponibili, con interpolazione dei gap brevi."""
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
        valid_raw_mask = np.zeros(n_frames, dtype=bool)
        prev_x_axis = None
        
        for i in range(n_frames):
            start = start_points[i]
            end = end_points[i]
            
            if start is None or end is None:
                starts_array.append([np.nan, np.nan, np.nan])
                ends_array.append([np.nan, np.nan, np.nan])
                midpoints_array.append([np.nan, np.nan, np.nan])
                quaternions_array.append([np.nan, np.nan, np.nan, np.nan])
                continue
            
            # Punto medio del segmento
            mid = [(start[j] + end[j]) / 2 for j in range(3)]

            # Quaternione dalla direzione del segmento
            R, prev_x_axis = build_local_frame(start, end, previous_x=prev_x_axis)
            if R is not None:
                q = rotation_matrix_to_quaternion(R)
                valid_raw_mask[i] = True
            else:
                q = [np.nan, np.nan, np.nan, np.nan]
            
            starts_array.append(start)
            ends_array.append(end)
            midpoints_array.append(mid)
            quaternions_array.append(q)
        
        # Conversione in array numpy
        starts_array = np.asarray(starts_array, dtype=float)
        ends_array = np.asarray(ends_array, dtype=float)
        midpoints_array = np.asarray(midpoints_array, dtype=float)
        quaternions_array = np.asarray(quaternions_array, dtype=float)

        starts_interp = interpolate_small_gaps_2d(starts_array, MAX_INTERPOLATION_GAP_FRAMES)
        ends_interp = interpolate_small_gaps_2d(ends_array, MAX_INTERPOLATION_GAP_FRAMES)
        mids_interp = np.full_like(starts_interp, np.nan)
        valid_mid = is_finite_row(starts_interp) & is_finite_row(ends_interp)
        mids_interp[valid_mid] = (starts_interp[valid_mid] + ends_interp[valid_mid]) / 2.0

        quaternions_proc = normalize_quaternion_series(quaternions_array)
        quaternions_proc = enforce_quaternion_continuity(quaternions_proc)
        quaternions_proc = interpolate_small_gaps_quaternions(quaternions_proc, MAX_INTERPOLATION_GAP_FRAMES)
        quaternions_proc = normalize_quaternion_series(quaternions_proc)
        quaternions_proc = enforce_quaternion_continuity(quaternions_proc)
        valid_quat_mask = is_finite_row(quaternions_proc)
        quaternions_proc[~valid_quat_mask] = np.nan

        segment_data[seg_name] = {
            'start_points': starts_interp,
            'end_points': ends_interp,
            'midpoints': mids_interp,
            'quaternions': quaternions_proc,
            'lengths': np.array(lengths, dtype=float),
            'valid_raw_mask': valid_raw_mask,
            'valid_quaternion_mask': valid_quat_mask
        }
    
    return segment_data


# =============================================================================
# JOINT ANGLES (direction-based, 2-DOF)
# =============================================================================
# CALCOLO DEGLI ANGOLI ARTICOLARI (direction-based, 2 gradi di liberta')
# =============================================================================
# Un segmento a due estremi definisce solo una DIREZIONE (nessun roll), quindi gli
# angoli articolari di Eulero a 3 DOF non sono recuperabili. Gli angoli sono
# calcolati come angoli (con segno) tra le direzioni dei segmenti proiettate sui
# piani anatomici globali. Sono derivabili solo sagittale + frontale; gli assi
# trasversi (rotazione) richiederebbero il roll del segmento e sono volutamente
# omessi. I campi "sequence" di JOINT_DEFINITIONS e i nomi degli assi trasversi
# sono ormai legacy/non usati.

WORLD_VERTICAL = np.array([0.0, 1.0, 0.0])     # Y verso l'alto (dopo convert_coords)
SAGITTAL_NORMAL = np.array([1.0, 0.0, 0.0])    # mediolaterale (X)
FRONTAL_NORMAL = np.array([0.0, 0.0, 1.0])     # antero-posteriore (Z)

SAGITTAL_AXIS_NAMES = {"flexion_extension", "plantar_dorsiflexion"}
FRONTAL_AXIS_NAMES = {"abduction_adduction", "varus_valgus", "lateral_bending", "inversion_eversion"}

# Direzione di riferimento prossimale per articolazione. Le articolazioni il cui
# segmento prossimale non e' un arto longitudinale (pelvis_center e' mediolaterale)
# usano il tronco o la verticale del mondo.
JOINT_PROXIMAL_REF = {
    "spine": "vertical",
    "neck": "torso",
    "hip_left": "torso",
    "hip_right": "torso",
    "shoulder_left": "torso",
    "shoulder_right": "torso",
}


def segment_direction_series(segment_data, seg_name):
    """Direzione unitaria (end - start) per frame, NaN dove il segmento non e' valido."""
    sd = segment_data.get(seg_name)
    if sd is None:
        return None
    start = np.asarray(sd['start_points'], dtype=float)
    end = np.asarray(sd['end_points'], dtype=float)
    d = end - start
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        unit = d / norm
    unit[norm[:, 0] < 1e-6] = np.nan
    # Limita ai frame realmente osservati: i frame interpolati/riempiti producono
    # direzioni rumorose e picchi spuri negli angoli.
    mask = sd.get('valid_raw_mask')
    if mask is not None:
        unit[~np.asarray(mask, dtype=bool)] = np.nan
    return unit


def included_angle(ref, vec):
    """Angolo senza segno (gradi, 0..180) tra due serie di direzioni. Senza wrap."""
    dot = np.clip(np.sum(ref * vec, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def lateral_deviation(vec):
    """Inclinazione con segno (gradi, -90..90) di una direzione fuori dal piano
    sagittale, misurata attorno all'asse mediolaterale. Senza wrap."""
    return np.degrees(np.arcsin(np.clip(vec @ SAGITTAL_NORMAL, -1.0, 1.0)))


def despike_isolated(series, threshold=90.0):
    """Mette a NaN gli outlier di singolo frame che distano da entrambi i vicini piu' di threshold."""
    out = np.asarray(series, dtype=float).copy()
    n = out.shape[0]
    for i in range(1, n - 1):
        a, b, c = out[i - 1], out[i], out[i + 1]
        if np.isfinite(a) and np.isfinite(b) and np.isfinite(c):
            if abs(b - a) > threshold and abs(b - c) > threshold:
                out[i] = np.nan
    return out


def signed_planar_angle(ref, vec, normal):
    """Angolo con segno (gradi) da ref a vec, entrambi proiettati sul piano ⟂ normal."""
    n = normal / np.linalg.norm(normal)
    ref_p = ref - np.outer(ref @ n, n)
    vec_p = vec - np.outer(vec @ n, n)
    rp = np.linalg.norm(ref_p, axis=1)
    vp = np.linalg.norm(vec_p, axis=1)
    cross = np.cross(ref_p, vec_p) @ n
    dot = np.sum(ref_p * vec_p, axis=1)
    ang = np.degrees(np.arctan2(cross, dot))
    ang[(rp < 1e-6) | (vp < 1e-6)] = np.nan
    return ang


def _proximal_reference_series(segment_data, joint_name, joint_def, n_frames):
    ref_key = JOINT_PROXIMAL_REF.get(joint_name, "segment")
    if ref_key == "vertical":
        return np.tile(WORLD_VERTICAL, (n_frames, 1))
    if ref_key == "segment":
        return segment_direction_series(segment_data, joint_def["proximal"])
    # segmento nominato (es. torso) con ripiego sulla verticale del mondo per frame
    ref = segment_direction_series(segment_data, ref_key)
    if ref is None:
        return np.tile(WORLD_VERTICAL, (n_frames, 1))
    bad = ~np.all(np.isfinite(ref), axis=1)
    ref[bad] = WORLD_VERTICAL
    return ref


def compute_joint_angles(segment_data, available_joints, dt):
    """
    Angoli articolari dalle direzioni dei segmenti (con segno, proiettate sui piani
    globali sagittale e frontale). La rotazione trasversa non e' derivabile da
    segmenti a 2 punti e viene omessa.
    """
    joint_angles = {}

    for joint_name, joint_def in available_joints.items():
        distal_dir = segment_direction_series(segment_data, joint_def["distal"])
        if distal_dir is None:
            continue
        n_frames = distal_dir.shape[0]
        ref_dir = _proximal_reference_series(segment_data, joint_name, joint_def, n_frames)
        if ref_dir is None:
            continue

        angles = {}
        for axis in joint_def["axes"]:
            if axis in SAGITTAL_AXIS_NAMES:
                # entita' della flessione = angolo incluso tra le direzioni (0..180),
                # robusto e senza wrap (piano di moto dominante nella corsa).
                raw = included_angle(ref_dir, distal_dir)
            elif axis in FRONTAL_AXIS_NAMES:
                # asse frontale = inclinazione laterale assoluta del segmento distale
                # fuori dal piano sagittale (limitata -90..90, senza wrap). Solo una
                # stima: un angolo nel piano con segno "ruota", e sottrarre un tronco
                # di riferimento rumoroso gonfia il range oltre i limiti fisici su
                # questa ripresa.
                raw = lateral_deviation(distal_dir)
            else:
                continue  # trasverso / rotazione: non derivabile senza il roll
            angles[axis] = despike_isolated(raw)

        if angles:
            joint_angles[joint_name] = angles

    return joint_angles


def compute_angular_derivatives(segment_data, dt):
    """Calcola velocita' e accelerazione angolare per ogni segmento."""
    angular_velocity = {}
    angular_acceleration = {}

    for seg_name, seg_data in segment_data.items():
        q = seg_data['quaternions']
        valid_mask = seg_data.get('valid_quaternion_mask', np.ones(len(q), dtype=bool))

        # Velocita' angolare dalla derivata del quaternione
        omega = quaternion_derivative(q, dt, valid_mask)
        omega = smooth_vector_series(omega, sigma=1.0)

        # Accelerazione angolare
        alpha = np.full_like(omega, np.nan, dtype=float)
        for i in range(1, len(omega)):
            if np.all(np.isfinite(omega[i])) and np.all(np.isfinite(omega[i - 1])):
                alpha[i] = (omega[i] - omega[i - 1]) / dt
        alpha = smooth_vector_series(alpha, sigma=1.0)
        
        angular_velocity[seg_name] = omega
        angular_acceleration[seg_name] = alpha
    
    return angular_velocity, angular_acceleration


def export_c3d(segment_data, joint_angles, angular_velocity, angular_acceleration,
                metadata, output_path):
    """Esporta i dati in formato C3D (punti + copia opzionale dei canali analogici)."""
    if not EZC3D_AVAILABLE:
        print("Attenzione: ezc3d non disponibile, esportazione C3D saltata")
        return False
    
    try:
        n_frames = metadata['n_frames']
        fps = metadata['fps']
        
        # Costruisce le label dei punti: mid/start/end dei segmenti + centri articolari.
        point_series = {}
        for seg_name, seg_data in segment_data.items():
            point_series[seg_name] = seg_data['midpoints']
            point_series[f"{seg_name}_start"] = seg_data['start_points']
            point_series[f"{seg_name}_end"] = seg_data['end_points']

        for joint_name, joint_def in JOINT_DEFINITIONS.items():
            anchors = JOINT_CENTER_ENDPOINTS.get(joint_name)
            if anchors is None:
                prox = joint_def["proximal"]
                dist = joint_def["distal"]
                anchors = [(prox, "end"), (dist, "start")]

            anchor_series = []
            for seg_name, anchor in anchors:
                series = get_segment_anchor_series(segment_data, seg_name, anchor)
                if series is not None:
                    anchor_series.append(np.asarray(series, dtype=float))

            if not anchor_series:
                continue

            center = np.full_like(anchor_series[0], np.nan, dtype=float)
            counts = np.zeros(center.shape[0], dtype=float)
            accum = np.zeros_like(center, dtype=float)
            for series in anchor_series:
                valid = is_finite_row(series)
                accum[valid] += series[valid]
                counts[valid] += 1.0
            valid_any = counts > 0
            center[valid_any] = accum[valid_any] / counts[valid_any, np.newaxis]
            point_series[f"{joint_name}_center"] = center

        point_labels = list(point_series.keys())
        n_points = len(point_labels)

        # Costruisce le label analogiche per velocita'/accelerazione angolare + angoli articolari.
        analog_labels = []
        for seg_name in segment_data:
            for axis in ['x', 'y', 'z']:
                analog_labels.append(f"{seg_name}_omega_{axis}")
            for axis in ['x', 'y', 'z']:
                analog_labels.append(f"{seg_name}_alpha_{axis}")
        for joint_name, angles in joint_angles.items():
            for axis_name in angles.keys():
                analog_labels.append(f"{joint_name}_{axis_name}")
        
        n_analogs = len(analog_labels)

        # Crea la struttura C3D
        c3d = ezc3d.c3d()

        # Aggiunge i dati dei punti
        points_data = np.zeros((4, n_points, n_frames))
        for i, point_name in enumerate(point_labels):
            series = np.asarray(point_series[point_name], dtype=float)
            valid = is_finite_row(series)
            safe = np.nan_to_num(series, nan=0.0)
            points_data[:3, i, :] = safe.T
            points_data[3, i, :] = np.where(valid, 1.0, -1.0)
        
        # Aggiunge prima i parametri
        c3d.add_parameter('POINT', 'LABELS', point_labels)
        c3d.add_parameter('POINT', 'USED', n_points)
        c3d.add_parameter('POINT', 'RATE', float(fps))

        c3d.add_parameter('ANALOG', 'LABELS', analog_labels)
        c3d.add_parameter('ANALOG', 'USED', n_analogs)
        c3d.add_parameter('ANALOG', 'RATE', float(fps))

        # Aggiunge i dati
        c3d['data']['points'] = points_data

        # Aggiunge i dati analogici (mantiene i NaN per evitare falsi zeri)
        analogs_data = np.full((1, n_analogs, n_frames), np.nan, dtype=float)
        analog_idx = 0

        def ensure_vector_series(series, n_frames):
            if series is None:
                return np.full((n_frames, 3), np.nan, dtype=float)
            arr = np.asarray(series, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3:
                fixed = np.full((n_frames, 3), np.nan, dtype=float)
                k = min(n_frames, arr.shape[0]) if arr.ndim > 0 else 0
                if k > 0:
                    fixed[:k] = np.asarray(arr[:k]).reshape(k, -1)[:, :3]
                return fixed
            if arr.shape[0] != n_frames:
                fixed = np.full((n_frames, 3), np.nan, dtype=float)
                k = min(n_frames, arr.shape[0])
                fixed[:k] = arr[:k]
                return fixed
            return arr

        for seg_name in segment_data:
            omega = ensure_vector_series(angular_velocity.get(seg_name), n_frames)
            alpha = ensure_vector_series(angular_acceleration.get(seg_name), n_frames)
            
            for axis in range(3):
                analogs_data[0, analog_idx, :] = omega[:, axis]
                analog_idx += 1
            for axis in range(3):
                analogs_data[0, analog_idx, :] = alpha[:, axis]
                analog_idx += 1

        # Aggiunge gli angoli articolari come canali analogici (gradi)
        for joint_name, angles in joint_angles.items():
            for axis_name, values in angles.items():
                values_arr = np.asarray(values, dtype=float)
                if values_arr.shape[0] != n_frames:
                    fixed = np.full(n_frames, np.nan, dtype=float)
                    k = min(n_frames, values_arr.shape[0])
                    fixed[:k] = values_arr[:k]
                    values_arr = fixed
                analogs_data[0, analog_idx, :] = values_arr
                analog_idx += 1
        
        c3d['data']['analogs'] = analogs_data

        # Scrive il C3D
        c3d.write(output_path)
        print(f"C3D salvato: {output_path}")

        return True

    except Exception as e:
        print(f"Attenzione: esportazione C3D fallita: {e}")
        print("Proseguo con l'esportazione JSON/CSV...")
        return False


def export_json(segment_data, joint_angles, angular_velocity, angular_acceleration,
                metadata, available_segments, available_joints, output_path):
    """Esporta i dati in formato JSON (output canonico della pipeline)."""
    output_data = {
        "metadata": {
            "session": metadata.get('session', 'unknown'),
            "sport": metadata.get('sport', 'unknown'),
            "fps": metadata['fps'],
            "n_frames": metadata['n_frames'],
            "duration_seconds": metadata['n_frames'] / metadata['fps'],
            "view": metadata.get('view', 'unknown'),
            "available_segments": available_segments,
            "computed_joints": list(available_joints.keys()),
            "angle_convention": {
                "method": "signed angle between segment direction vectors projected onto global sagittal/frontal planes",
                "computed_planes": ["sagittal", "frontal"],
                "dropped_planes": ["transverse"],
                "note": "Joint angles are 2-DOF, derived from segment directions only (2-point segments carry no roll). Transverse/rotation axes are not computed. Not full ISB; sagittal uses the AP axis (Z), the least reliable dimension on a frontal-view capture."
            }
        },
        "segments": {},
        "joints": {}
    }
    
    # Dati dei segmenti
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
    
    # Angoli articolari (converte gli array numpy in liste)
    for joint_name, angles in joint_angles.items():
        output_data["joints"][joint_name] = {}
        for axis, values in angles.items():
            output_data["joints"][joint_name][axis] = values.tolist()
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"JSON salvato: {output_path}")


def process_session(session_dir, output_root=None):
    """Elabora una singola sessione end-to-end: carica, calcola e scrive C3D/JSON/CSV."""
    print(f"\n{'='*60}")
    print(f"Elaborazione sessione: {session_dir}")
    print(f"{'='*60}")

    # Carica i dati
    frames_path = os.path.join(session_dir, "segments_frames.json")
    summary_path = os.path.join(session_dir, "segments_summary.json")

    if not os.path.exists(frames_path):
        print(f"Errore: {frames_path} non trovato")
        return None

    if not os.path.exists(summary_path):
        print(f"Errore: {summary_path} non trovato")
        return None

    print("Caricamento dei file JSON...")
    frames_data = load_json(frames_path)
    summary_data = load_json(summary_path)

    # Estrae i metadati
    fps = summary_data.get('metadata', {}).get('fps', 30.0)
    n_frames = len(frames_data['frames'])
    view = summary_data.get('metadata', {}).get('view', 'unknown')
    session_name = os.path.basename(session_dir)

    metadata = {
        'session': session_name,
        'sport': summary_data.get('metadata', {}).get('sport', 'unknown'),
        'fps': fps,
        'n_frames': n_frames,
        'view': view
    }

    print(f"FPS: {fps}, Frame: {n_frames}, Vista: {view}")

    # Rileva automaticamente i segmenti disponibili
    print("\nRilevamento automatico dei segmenti disponibili...")
    available_segments = auto_detect_available_segments(summary_data, MIN_SEGMENT_COVERAGE_PCT)
    print(f"Trovati {len(available_segments)} segmenti con copertura >= {MIN_SEGMENT_COVERAGE_PCT:.1f}%:")
    for seg in available_segments:
        coverage = summary_data.get('coverage_stats', {}).get(seg, {}).get('coverage_pct', 0)
        print(f"  - {seg}: {coverage:.1f}%")

    low_coverage_segments = []
    for seg in ALL_SEGMENTS:
        coverage = summary_data.get('coverage_stats', {}).get(seg, {}).get('coverage_pct', 0)
        if 0 < coverage < LOW_COVERAGE_WARNING_PCT:
            low_coverage_segments.append((seg, coverage))
    if low_coverage_segments:
        print("\nSegmenti a bassa copertura (inclusi, ma potenzialmente rumorosi):")
        for seg, cov in low_coverage_segments:
            print(f"  - {seg}: {cov:.1f}% (soglia di avviso {LOW_COVERAGE_WARNING_PCT:.1f}%)")

    # Determina le articolazioni calcolabili
    print("\nCalcolo delle articolazioni disponibili...")
    available_joints = compute_available_joints(available_segments, JOINT_DEFINITIONS)
    print(f"Trovate {len(available_joints)} articolazioni calcolabili:")
    for joint_name, joint_def in available_joints.items():
        print(f"  - {joint_name}: {joint_def['dof']} DOF ({', '.join(joint_def['axes'])})")

    # Calcola i quaternioni dei segmenti
    print("\nCalcolo dei quaternioni dei segmenti...")
    segment_data = compute_segment_quaternions(frames_data, available_segments)

    print("Applico continuita'/normalizzazione dei quaternioni e riempimento controllato dei gap...")

    # Calcola le derivate angolari
    print("Calcolo di velocita' e accelerazione angolare...")
    dt = 1.0 / fps
    angular_velocity, angular_acceleration = compute_angular_derivatives(segment_data, dt)

    # Calcola gli angoli articolari
    print("Calcolo degli angoli articolari (direction-based)...")
    joint_angles = compute_joint_angles(segment_data, available_joints, dt)

    # Esportazione output (in data/processed/<sessione>_processed/ per default)
    if output_root is None:
        output_root = DEFAULT_DATA_PROCESSED
    output_dir = os.path.join(output_root, f"{session_name}_processed")
    os.makedirs(output_dir, exist_ok=True)

    # Esporta C3D
    c3d_path = os.path.join(output_dir, f"{session_name}_generic.c3d")
    export_c3d(segment_data, joint_angles, angular_velocity, angular_acceleration,
               metadata, c3d_path)

    # Esporta JSON
    json_path = os.path.join(output_dir, f"{session_name}_generic.json")
    export_json(segment_data, joint_angles, angular_velocity, angular_acceleration,
                metadata, available_segments, available_joints, json_path)

    # Esporta un CSV per ogni segmento
    print("\nEsportazione dei file CSV...")
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
                "quat_x", "quat_y", "quat_z", "quat_w",
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

    # Esporta il CSV degli angoli articolari
    if joint_angles:
        joint_csv_path = os.path.join(output_dir, f"{session_name}_joints.csv")
        angle_cols = [(j, ax) for j in joint_angles for ax in joint_angles[j]]
        with open(joint_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "time"] + [f"{j}_{ax}" for j, ax in angle_cols])
            for i in range(n_frames):
                writer.writerow([i, time_arr[i]] + [joint_angles[j][ax][i] for j, ax in angle_cols])
        print(f"  CSV articolazioni: {joint_csv_path}")

    print(f"\n{'='*60}")
    print("ELABORAZIONE COMPLETATA")
    print(f"{'='*60}")
    print(f"Sessione: {session_name}")
    print(f"Segmenti elaborati: {len(available_segments)}")
    print(f"Articolazioni calcolate: {len(available_joints)}")
    print(f"Cartella di output: {output_dir}")
    
    return {
        'available_segments': available_segments,
        'available_joints': available_joints,
        'segment_data': segment_data,
        'joint_angles': joint_angles
    }


def main():
    parser = argparse.ArgumentParser(
        description='Converte motion-capture MediaPipe in C3D con angoli articolari'
    )
    parser.add_argument(
        'session',
        nargs='?',
        default=None,
        help='Cartella o nome della sessione (default: rilevamento automatico da --data-root)'
    )
    parser.add_argument(
        '--data-root',
        default=DEFAULT_DATA_RAW,
        help=f'Cartella contenente le sessioni running_* (default: {DEFAULT_DATA_RAW})'
    )
    parser.add_argument(
        '--output-root',
        default=DEFAULT_DATA_PROCESSED,
        help=f'Cartella dove viene scritta <sessione>_processed/ (default: {DEFAULT_DATA_PROCESSED})'
    )
    parser.add_argument(
        '--list-sessions',
        action='store_true',
        help='Elenca le sessioni disponibili in --data-root'
    )

    args = parser.parse_args()
    data_root = args.data_root

    def _discover_sessions(root):
        found = []
        if not os.path.isdir(root):
            return found
        for item in sorted(os.listdir(root)):
            full = os.path.join(root, item)
            if os.path.isdir(full) and item.startswith('running_'):
                if os.path.exists(os.path.join(full, 'segments_frames.json')):
                    found.append(item)
        return found

    if args.list_sessions:
        print(f"Sessioni disponibili in {data_root}:")
        for item in _discover_sessions(data_root):
            print(f"  - {item}")
        return

    # Determina la cartella della sessione
    if args.session:
        # Accetta sia un percorso completo sia il solo nome sotto --data-root
        if os.path.isdir(args.session):
            session_dir = args.session
        else:
            session_dir = os.path.join(data_root, args.session)
    else:
        # Rilevamento automatico: cerca una cartella running_* sotto data_root
        available = _discover_sessions(data_root)

        if not available:
            print(f"Errore: nessuna cartella di sessione trovata in {data_root}")
            print("Uso: python process_session_generic.py <cartella_sessione> [--data-root DIR]")
            sys.exit(1)

        if len(available) == 1:
            session_dir = os.path.join(data_root, available[0])
            print(f"Selezione automatica: {available[0]}")
        else:
            print("Sessioni disponibili:")
            for i, sess in enumerate(available):
                print(f"  {i+1}. {sess}")
            try:
                choice = int(input("Numero della sessione: ")) - 1
                if choice < 0 or choice >= len(available):
                    print("Selezione non valida")
                    sys.exit(1)
                session_dir = os.path.join(data_root, available[choice])
            except ValueError:
                print("Input non valido")
                sys.exit(1)

    # Elabora la sessione
    result = process_session(session_dir, output_root=args.output_root)

    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
