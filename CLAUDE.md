# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Thesis project: a biomechanics motion-capture pipeline. Raw 3D pose data (MediaPipe-style segment endpoints) is converted with quaternion math into joint angles + angular kinematics, exported to C3D/JSON/CSV, ingested into PostgreSQL, and queried via an MCP server exposing explicit biomechanical metrics. Most docs and inline prose are in Italian; keep that convention when editing docs.

There is **no test/lint toolchain** (no `pytest`, `Makefile`); deps are listed informally in `requirements.txt`. Run scripts with the in-repo venv: `.venv/bin/python` (Python 3.9). Note this venv has numpy/scipy/open3d but **not** `ezc3d` or `psycopg`/`mcp` (FastMCP) — so locally the converter degrades to JSON/CSV (no C3D), and `ingest_c3d.py` / `src/mcp/server.py` only run in an environment where `psycopg` + `mcp` are installed (e.g. Claude Desktop's MCP runtime). To smoke-test `server.py` helpers offline, stub `psycopg` in `sys.modules` and load by absolute path from a CWD outside the repo (the local `src/mcp/` folder otherwise shadows the installed `mcp` package).

**Repo layout** (post-restructure): code in `src/{conversion,visualization,database,mcp}/`, project docs in `docs/`, motion-capture data in `data/{raw,processed}/`. `data/` is **git-ignored** (not versioned). Scripts resolve data roots as `<repo-root>/data/...` via `__file__`, so they run from any CWD; converter/viewers/ingest accept `--data-root`/`--output-root`/`--root`/positional overrides.

## Pipeline stages (the big picture)

The data flows through four stages, each in its own `src/` subpackage:

1. **Conversion** — `src/conversion/process_session_generic.py` (the single, canonical converter)
   Reads a `running_*` session folder (`segments_frames.json` + `segments_summary.json`), computes per-segment quaternions (for rendering only) plus **direction-based joint angles**: flexion = included angle between proximal-reference and distal segment directions (0–180°, wrap-free, via `included_angle`); frontal axis = `signed_planar_angle` in the global frontal plane. **Transverse/rotation axes are not computed** (2-point segments carry no roll — see ISB gotcha). Isolated single-frame spikes are removed (`despike_isolated`). Writes `<session>_processed/` with `<session>_generic.json` (**canonical** output), per-segment CSVs, `<session>_joints.csv`, and `<session>_generic.c3d` (points only, if `ezc3d` importable). The JSON `metadata` block carries `sport` and an `angle_convention` note.

2. **Visualization** — `src/visualization/c3d_viewer.py` (Matplotlib) and `c3d_viewer_open3d.py` (Open3D) consume processed data for playback.

3. **Storage** — `src/database/` (`schema.sql` + `ingest_c3d.py`). **JSON-canonical ingestion**: `ingest_c3d.py` is driven by `*_processed/*_generic.json` and writes the derived kinematics to `c3d_analogs` — joint angles (`{joint}_{axis}`), per-segment `{seg}_omega_{x,y,z}` / `{seg}_alpha_{x,y,z}`, and quaternions `{seg}_quat_{x,y,z,w}`. The sibling `.c3d` is read **only** for 3D point trajectories → `c3d_points` (interop/rendering; skipped if `ezc3d` absent). `sessions.metadata_json` = the JSON `metadata` block, so `sport` is **top-level** (this is what makes `compare_sports`' `metadata_json ->> 'sport'` work). Gaps are JSON `null` → SQL `NULL` in the `REAL[]`.

4. **Query (MCP)** — `src/mcp/server.py`. FastMCP server exposing ~20 metric tools (rom, peak_flexion, angular_velocity, relative_phase, symmetry_index, compare_sessions, gesture_detection, …) reading from `c3d_analogs`, plus `detect_cycles` (gait cycles = peaks of hip flexion). `movement_variability` auto-detects cycles from hip flexion when none are passed (`cycles_autodetected` warning). `relative_phase` uses amplitude-normalized continuous relative phase. Any non-sagittal query emits the `estimated_non_anatomical_axis` warning.

`docs/Piano_Manipolazione_Dati.md` describes the *target* architecture: pgvector embeddings and a warning-only DSL are still **planned**; deterministic cycle segmentation now exists in a basic form (`detect_cycles`).

## Commands

```bash
# Conversion (paths are repo-relative via __file__; runs from any CWD).
# Reads data/raw/, writes data/processed/<session>_processed/.
python src/conversion/process_session_generic.py --list-sessions
python src/conversion/process_session_generic.py running_20260220_121922
#   overrides: --data-root DIR  --output-root DIR

# Visualization (defaults to data/processed; pass a dir to override)
python src/visualization/c3d_viewer.py
python src/visualization/c3d_viewer_open3d.py

# DB: create schema, then ingest all data/processed/*_processed/*_generic.json
psql -d biomech -f src/database/schema.sql
python src/database/ingest_c3d.py --db "dbname=biomech"   # --root DIR to override data dir

# MCP server (DSN via MCP_DB_DSN env, defaults to "dbname=biomech")
python src/mcp/server.py
```

A single `running_*` folder is the closest thing to a "single test" unit — run the converter on one folder to validate changes end to end.

## Conventions and gotchas (repository-specific)

- **Quaternion order is SciPy `[x, y, z, w]`** in array/storage paths (`Rotation.from_quat`/`as_quat`). But the hand-written `quaternion_multiply`/`quaternion_inverse` helpers in `process_session_generic.py` treat input as `[w, x, y, z]` semantics over `q[3]`/`q[:3]` — read each helper's docstring before reusing; the two conventions coexist deliberately.
- **Coordinate normalization**: raw MediaPipe points are converted `[x, -y, -z]` (`convert_coords`) before any math. Sentinel `[0,0,0]` and non-finite points are treated as missing.
- **Segment availability gate**: a segment is usable only when `coverage_stats.<segment>.coverage_pct > 0` in `segments_summary.json`. A joint is computed only if both its proximal and distal segments are available.
- **NaN propagation is intentional**: gaps stay NaN unless short enough to interpolate (`MAX_INTERPOLATION_GAP_FRAMES = 10`, SLERP for quaternions). C3D analog channels keep NaNs; point residual row encodes validity (`+1`/`-1`). Don't replace NaNs with 0 in math paths — it creates false signal.
- **Authoritative static mappings**: `SEGMENT_TO_JOINT`, `JOINT_DEFINITIONS`, `JOINT_CENTER_ENDPOINTS`, `ALL_SEGMENTS` in `process_session_generic.py`. When adding a segment/joint, update all of them and keep the viewers in sync.
- **Label registry is the query vocabulary**: `src/mcp/label_registry.json` maps `(joint, side, plane) → analog label`. MCP tools resolve labels through it; a query for an unmapped joint/plane returns `{"warnings": ["label_missing"]}` rather than erroring. `knee_left` now resolves, but `tibia_left` coverage is ~8% in the sample data, so left-knee/ankle queries are valid yet mostly `all_nan`/noisy — a data-quality issue, not a code bug.
- **Angles are direction-based, 2-DOF, NOT full ISB (honesty contract)**: a 2-point segment defines only a *direction* (no roll), so 3-DOF Euler joint angles are mathematically unrecoverable from this data — the old quaternion→relative-Euler approach produced physically impossible values (ROM ≈ 360°). Joint angles are now signed/included angles between segment directions projected onto global planes: **sagittal (flexion) + frontal only; transverse dropped**. For hip/shoulder/spine/neck the proximal reference is the trunk/world-vertical (not the mediolateral `pelvis_center`). Even sagittal leans on the AP axis, the noisiest dimension on a frontal capture; non-flexion queries carry the `estimated_non_anatomical_axis` warning. Validate any angle change with `/tmp/test_joint_angles.py <generic.json>` (ROM < 200°, jumps > 90° under 1%). The quaternion Euler helpers (`quaternion_to_euler`, the `sequence` fields) are now legacy/unused.
- **C3D role**: C3D is now an interop/rendering artifact for **3D points only**. Angles/ω/α in C3D analog channels (still written by the converter) are an *optional copy*, **not** the DB source — the DB is fed from the JSON. Don't reintroduce the C3D→`c3d_analogs` path.
- **Left/right on frontal view is UNVERIFIED**: `JOINT_CENTER_ENDPOINTS` assumes `shoulder_width.start`/`pelvis_center.start` == left. On a frontal capture this is a mirroring risk — verify against a known frame before trusting side-specific (symmetry/compare) results.
- **MCP `sys.path` shadowing**: `src/mcp/server.py` removes its own directory from `sys.path` at import time so `from mcp.server.fastmcp import FastMCP` resolves to the installed `mcp` package, not the local `src/mcp/` folder (still named `mcp`, so the shadowing risk persists). Preserve that removal if you reorganize imports.
- **MCP response contract**: every tool returns `input_spec`, `selected_data`, `result`, `explain`, `warnings`. Keep this shape (and the `explain` formula string) when adding tools — it's the explainability guarantee the design depends on.
- **Optional deps degrade, don't block**: missing `ezc3d` → JSON/CSV export still runs; missing Open3D/pytransform3d only affects the Open3D viewer.

## Reference docs

- `docs/IdeeQuery.md` — the 20 target queries the MCP layer must cover.
- `docs/Idea_CFG.md` — grammar/CFG sketch for the planned natural-language/DSL query interface.
- `docs/ValutazioniProgettuali.md` — tool/library evaluation (quaternions, export formats, 3D viewers).
- `docs/Test_Generic_Descrizione.md` — description of the generic conversion + rendering test.
