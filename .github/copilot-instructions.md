# Copilot instructions for this repository

## Build, test, and lint commands

This repository has no `pytest`/`Makefile` test-lint toolchain; deps are listed informally in `requirements.txt`.
Use the Python entrypoints below as the operational commands. Code lives in `src/`, data in `data/{raw,processed}/` (git-ignored), docs in `docs/`. Scripts resolve data roots from `__file__`, so they run from any CWD.

### Conversion pipeline (primary executable flow)

```bash
# reads data/raw/, writes data/processed/<session>_processed/
python src/conversion/process_session_generic.py --list-sessions
python src/conversion/process_session_generic.py running_20260420_143730
#   overrides: --data-root DIR  --output-root DIR
```

### Visualization

```bash
# Matplotlib viewer (defaults to data/processed)
python src/visualization/c3d_viewer.py

# Open3D viewer
python src/visualization/c3d_viewer_open3d.py
```

### Single-session run (closest equivalent to a "single test")

Use one `running_*` folder as the test unit:

```bash
python src/conversion/process_session_generic.py running_20260220_121922
```

## High-level architecture

The implemented codebase is centered on a **Python motion-conversion + visualization pipeline**:

1. `src/conversion/process_session_generic.py` loads `segments_frames.json` + `segments_summary.json` from a `running_*` session, computes segment quaternions, joint angles, angular velocity/acceleration, and exports processed outputs.
2. Outputs are written to `<session>_processed/` as:
   - `<session>_generic.json` (main interchange format for viewers/downstream tools)
   - per-segment CSV files
   - optional `<session>_generic.c3d` if `ezc3d` is installed
3. `src/visualization/c3d_viewer.py` (Matplotlib) and `c3d_viewer_open3d.py` (Open3D) consume processed data for interactive playback.

From repository docs (`README.md`, `docs/Idea_CFG.md`), there is also a broader target architecture including **querying biomechanical data via a CFG/NL interface** (lexer → parser → semantic analyzer → executor). In this repository snapshot that query engine is conceptual/documented, while the concrete implemented code is the conversion/rendering stack above.

## Key conventions (repository-specific)

1. **Session discovery convention:** processing/visualization scripts expect folders named `running_*` and auto-detect them under the configured data root (`data/raw` for the converter, `data/processed` for viewers).
2. **Segment availability gate:** segments are considered usable only when `coverage_stats.<segment>.coverage_pct > 0` (`segments_summary.json`).
3. **Coordinate-system normalization in conversion:** raw MediaPipe points are converted with `[x, -y, -z]` before further processing.
4. **Quaternion ordering in code paths:** internal arrays use SciPy-style quaternion order `[x, y, z, w]` (`Rotation.from_quat`/`as_quat`); preserve this when adding math logic.
5. **Optional dependency behavior:** missing `ezc3d` should not block processing (JSON/CSV export remains), and missing Open3D/pytransform3d only affects Open3D viewing features.
6. **Static biomechanical mappings are authoritative:** keep `SEGMENT_TO_JOINT`, `JOINT_DEFINITIONS`, and `SKELETON_CONNECTIONS` synchronized across processor/viewers when introducing new segments or joints.
