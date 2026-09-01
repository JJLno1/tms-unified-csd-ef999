# Unified-CSD EF99.9 analysis — validation record

This analysis was developed and validated on two independent Nexstim-navigated
cohorts with charm head meshes (SimNIBS 4.6, Magstim 70 mm figure-8 dipole
model, Wagner isotropic conductivities, individual dI/dt).

## Cohort A — new cohort, n = 37 (TC065–TC123)

- 34/37 DLPFC targets had the recorded coil centre **below the scalp**
  (signed c2s mean −1.8 mm); motivation for unifying placement.
- Unified CSD: M1 = 2 mm, DLPFC = 1 mm. Results (mean ± SD, V/m):

| target | SCD (mm)   | new CCD (mm) | EF99.9 orig | EF99.9 opt | Δ(opt−orig) |
|--------|------------|--------------|-------------|------------|-------------|
| M1     | 15.9 ± 1.7 | 17.9 ± 1.7   | 169.0 ± 37.1| 179.7 ± 38.4 | +10.7 ± 13.2 |
| DLPFC  | 12.7 ± 1.8 | 13.7 ± 1.8   | 187.8 ± 49.0| 202.2 ± 49.3 | +14.4 ± 16.1 |

- Δ ≥ 0 in 74/74 cases (3 ties at θ* = 0).

## Cohort B — original TC38 cohort, n = 38 (TC001–TC057)

- 22/76 target cases below scalp at the recorded position (21 DLPFC + 1 M1;
  deepest −3.77 mm). The historical pipeline (ray backup 5 mm + `max(…, 0)`
  clamp) could structurally not represent this; all published values looked
  positive. Normal-case (on-scalp) CSD means were 1.94 mm (M1) and 1.20 mm
  (DLPFC) — the unified 2 / 1 mm sit essentially at the cohorts' healthy
  placement means.

| target | SCD (mm)   | new CCD (mm) | EF99.9 orig | EF99.9 opt | Δ(opt−orig) |
|--------|------------|--------------|-------------|------------|-------------|
| M1     | 13.8 ± 1.3 | 15.8 ± 1.3   | 180.4 ± 42.8| 193.3 ± 43.6 | +12.9 ± 13.8 |
| DLPFC  | 12.4 ± 1.6 | 13.4 ± 1.6   | 173.0 ± 37.3| 187.7 ± 40.3 | +14.8 ± 14.7 |

- After unification, the 21 originally-below-scalp DLPFC cases and the 17
  on-scalp cases have statistically indistinguishable dose-normalised
  original-orientation fields (2.30 vs 2.36 V/m per A/µs) — the placement
  artefact is removed.

## Numerical validation

1. **Parent-tet E mapping.** `fem.calc_fields` assigns surface triangles the
   parent tetrahedron's field; the raw `tms_many_simulations` path uses
   in-plane gradients + node-averaged dA/dt (~2% different on GM triangles).
   Reading E at the parent GM tet reproduces the official definition: a
   single-position cross-check matched `run_simnibs` output to
   EF99.9 181.78 vs 181.78, Emax 182.33 vs 182.33, global 119.52 vs 119.52
   (diff ≤ 0.01 V/m). In both cohort batches the per-subject `artifact_check`
   (|sweep − official-path artifact| at θ*) was **0.0 for all 150 cases**.
2. **Geometry identities.** c2s at the unified position = CSD exactly;
   SCD invariant (|Δ| < 0.05 mm, flagged otherwise); CCD = SCD + CSD to
   floating-point precision (max abs err 3.6e-15 mm).
3. **180° symmetry.** Sweeping 0–175° suffices: a figure-8 coil yawed 180°
   about its normal yields identical |E| (verified previously in 8/8 pilot
   cases with unrestricted 360° TMSoptimize searches — results identical).

## Example reference values (Cohort B, TC001, dI/dt = 75.2 A/µs)

- M1 (CSD 2 mm): SCD 14.42 mm, CCD 16.42 mm, θ* = 175°,
  EF99.9 orig 198.6 / opt 198.7 V/m
- DLPFC (CSD 1 mm): signed c2s −0.43 mm (below scalp at recording),
  SCD 13.86 mm, CCD 14.86 mm, θ* = 175°, EF99.9 orig 157.7 / opt 158.3 V/m

Full angle–EF99.9 curves for these two cases are in `data/`; the reported
values above are the θ = 0 and argmax rows of those curves, so a re-run of
this subject must reproduce them exactly (same solver, same mesh).

## Addendum 2026-09-01 — direction convention finalised

1. **E rides on coil-Y.** Across both cohorts the cortical hotspot E vector
   aligns with the coil's Y axis (4–23°) and is ~perpendicular to X. The
   historical auto-flip guarded X; the corrected rule (`forward_axis='Y'`)
   enforces anterior E at theta=0. |E|-equivalence of the two flips verified
   by a full independent 36-angle re-sweep: max |diff| = 0.001 V/m.
2. **Export sign bug found and corrected.** One cohort's DLPFC `.2` vectors
   had a sign-flipped y-component (0/37 positive; healthy groups 27–36/38),
   reconstructing E backwards in 33/37. After correcting the component sign,
   anterior E restored in 37/37 — matching operator practice (all placements
   made with anterior E). Effect on results: optimal EF99.9 and all geometry
   unchanged; original-orientation values recomputed (1 FEM/case).
3. **CSD sensitivity**: EF99.9 falls 4.2 %/mm of CSD (10-subject test,
   1–4 mm, all adjacent pairs p = 0.002); the optimal yaw is invariant to
   coil height.
4. **Final cohort values (forward-enforced, all corrections applied):**
   TC38 EF99.9 orig 176.7 / opt 190.5 V/m (M1+DLPFC pooled);
   new cohort orig 175.0 / opt 191.4 V/m.
