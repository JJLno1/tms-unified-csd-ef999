# tms-unified-csd-ef999

Unified coil-to-scalp distance (CSD) re-analysis for navigated TMS cohorts:
rebuild every coil at a **fixed CSD above the scalp** (M1 = 2 mm, DLPFC = 1 mm)
and re-evaluate **EF99.9** (99.9th percentile of |E| in the 5 mm GM target
sphere) for the original and the **optimal coil yaw**, with a **real FEM solve
for every candidate angle** (5° steps, no ADM approximation).

## Why

In Nexstim-navigated cohorts the recorded coil centre is often slightly
**below the scalp surface** (signed coil-to-scalp distance < 0 — measured in
21/38 DLPFC targets of one cohort and 34/37 of another). Analysing at the
recorded position mixes anatomy with this placement artefact, and pipelines
that clamp the distance at ≥ 0 cannot even detect it. Unifying the placement
removes the artefact; making the optimisation objective identical to the
reported metric (EF99.9) removes the classic "optimised < original" paradox
(opt ≥ orig holds by construction, since θ = 0 is a candidate).

## Method

Per subject × target (M1 / DLPFC), on a charm head mesh:

1. **Geometry (final Nexstim→SimNIBS rules)** — `sim_pos = (-x,-y,z)`;
   `gm_target` = nearest GM-surface node; coil axis `Z = (gm_target - sim_pos)`;
   handle `Y` = YZ-swap of the ".2" direction, Gram-Schmidt; `X = Y × Z`;
   auto-flip if `X.y < 0`. Signed `c2s`/`c2c` are the nearest line-surface
   intersections along `Z` (negative = below scalp); `SCD = |c2c - c2s|`.
2. **Unified placement** — coil centre moved along `Z` to exactly `csd_mm`
   outside the scalp; orientation unchanged. New `CCD = SCD + CSD` (verified
   numerically per case; SCD is invariant).
3. **Yaw sweep, real FEM** — `simnibs.simulation.fem.tms_many_simulations`
   shares one FEM matrix factorisation across 36 orientations
   (0–175° in 5° steps; 180° covers all distinct |E| states because a
   figure-8 coil yawed 180° produces an identical |E| distribution).
   `θ* = argmax EF99.9`; per-angle curve stored as tiny hdf5/CSV.
4. **Parent-tet E mapping** — E on GM surface triangles is read from the
   parent GM tetrahedron, reproducing `fem.calc_fields` /
   `assign_triangle_values` semantics exactly (the raw batch path differs by
   ~2% otherwise). Cross-validated to 0.00–0.01 V/m against official
   `run_simnibs` output meshes (see `docs/VALIDATION.md`).
5. **Artifact + audit** — θ* is additionally simulated through the official
   `run_simnibs` path; `artifact_check` must be ≈ 0.

Dose stays individual (dI/dt per subject). Conductivities: Wagner et al.
isotropic via `cond_utils.standard_cond()`.

## Usage

```bash
python run_unified_csd.py \
    --mesh-base "C:/data/TMS-SimNIBS" \
    --input-csv  originalInput.csv \
    --out-dir    output/tc38_unified_csd \
    --art-dir    "D:/project/artifacts"      # ~0.7 GB per subject
# options: --cases TC001,TC056  --workers 2  --m1-csd 2 --dlpfc-csd 1
#          --angle-step 5  --no-artifacts
```

The library (`unified_csd_core.py`) is importable on its own for geometry-only
use (`build_matsimnibs_final`, `compute_geometry_signed`, `parent_tet_map`, …).

### Input CSV

Double header (row 1 = groups, row 2 = names); per subject:
`<ID>, dI/dt-rMT (A/µs), {M1,DLPFC}-{x,y,z}, {M1,DLPFC}-{x,y,z} .2`
(Nexstim scanner LPS; ".2" = handle direction).

### Outputs

- `<out-dir>/unified_csd_long.csv` — one row per subject × target with
  signed original c2s/c2c/SCD, new CCD, θ*, orig/opt EF99.9/Emax/global,
  E-direction metrics, `delta_opt_minus_orig`, `artifact_check`
- `<out-dir>/<sub>/<target>_sweep.csv|*.hdf5` — full angle–EF99.9 curves
- `<art-dir>/<sub>/opt/` — θ* result meshes (Gmsh-ready, official path)

## Direction convention (updated 2026-09)

* The induced E of a figure-8 rides on the **coil-Y axis** (measured E-vs-Y
  angle 4–23° across two cohorts). `build_matsimnibs_final` therefore enforces
  an **anterior E at theta = 0** by flipping 180° about Z when coil-Y points
  posterior (`forward_axis='Y'`). The historical X-based flip differs only by
  the 180° twin — **every |E| value is identical** (A/B full-sweep check:
  max 0.001 V/m).
* The **optimal orientation is reported as the forward-E twin** (identical
  |E|), so all reported E directions are anterior, matching navigated-TMS
  practice.
* **Data-quality check**: one cohort's DLPFC direction exports carried a
  sign-flipped y-component (0/37 positive vs 27–36/38 in healthy groups),
  which reconstructed E backwards. `detect_direction_convention()` flags such
  groups; correcting the component sign restored anterior E in 37/37 cases.
  Run this check on any new cohort before analysis.

## Validation

Validated end-to-end on two independent cohorts (37 + 38 subjects, 150
subject-target cases): every `artifact_check = 0.0`, geometry identities hold
to 1e-15, opt ≥ orig in all cases. Details and reference numbers:
**[`docs/VALIDATION.md`](docs/VALIDATION.md)**; example angle–EF curves from
the validation cohorts in **[`data/`](data/)**.

## Requirements

`simnibs >= 4.0` (includes mesh_io + FEM + coil model), `numpy`, `scipy`,
`trimesh >= 3.9`, `h5py`, `pandas` (optional, tables). Run FEM batches on a
workstation: ~35 min/subject at 2 workers (36 solves × 2 targets).

## License

MIT — see [LICENSE](LICENSE).
