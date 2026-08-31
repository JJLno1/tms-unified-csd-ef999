#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
unified_csd_core
================

Unified coil-to-scalp distance (CSD) re-analysis for navigated TMS cohorts.

Problem this solves
-------------------
In Nexstim-navigated cohorts the RECORDED coil centre is frequently slightly
below the scalp surface (signed coil-to-scalp distance, c2s < 0; observed in
21/38 DLPFC targets of one cohort and 34/37 of another). Recording-position
FEM then mixes anatomy with placement artefacts, and historic pipelines that
clamped c2s at >= 0 could not even see the effect.

Method (per target, M1 and DLPFC)
---------------------------------
1. Geometry (final Nexstim->SimNIBS rules, unchanged from the validated
   production pipeline):
     sim_pos  = (-x, -y, z)                        (LPS -> RAS point)
     gm_target = nearest GM-surface node to sim_pos
     Z = (gm_target - sim_pos) / |.||              (two-point coil axis)
     Y = (-dx, dz, dy)                             (YZ-swap of handle ".2")
     Y <- Gram-Schmidt(Y against Z); X = Y x Z; flip if X.y < 0
   Signed c2s/c2c are the nearest line-surface intersections along Z
   (negative = coil below scalp); SCD = |c2c - c2s| strictly.
2. Unified placement: move the coil centre along Z to exactly
   ``csd_mm`` OUTSIDE the scalp (default M1 = 2 mm, DLPFC = 1 mm, the same
   unified gap used for the HCP healthy-cohort simulations). Orientation is
   kept; the new CCD = SCD + csd by construction (verified numerically).
3. EF99.9: 99.9th percentile of |E| over GM-surface triangle barycentres
   within 5 mm of gm_target, dose = individual dI/dt.
4. Optimal yaw: theta* = argmax EF99.9 over 0..175 deg in 5 deg steps,
   evaluated with a REAL FEM solve for every candidate angle via
   ``simnibs.simulation.fem.tms_many_simulations`` (one shared matrix
   factorisation; no ADM approximation). 180 deg covers all distinct |E|
   states because a figure-8 coil rotated 180 deg about its normal produces
   an identical |E| distribution. Position is FIXED, so the optimisation is
   angle-only and opt >= orig by construction (theta = 0 is a candidate).
5. E-values on surface triangles are taken from their PARENT GM tetrahedron,
   reproducing ``fem.calc_fields``/``assign_triangle_values`` semantics
   exactly (validated to 0.00-0.01 V/m against run_simnibs output meshes).

Self-contained: only numpy, scipy, trimesh, h5py and simnibs are required.
"""
import os
import csv
import glob
import numpy as np
import h5py
from scipy.spatial import cKDTree

from simnibs.mesh_tools import mesh_io
from simnibs.utils import cond_utils
from simnibs.simulation import fem

# ---------------------------------------------------------------------------
# Defaults (all overridable)
# ---------------------------------------------------------------------------
TARGETS  = ['M1', 'DLPFC']
CSD_MM   = {'M1': 2.0, 'DLPFC': 1.0}   # unified coil-to-scalp distance per target
ANGLES   = list(range(0, 180, 5))      # yaw candidates in degrees (180-deg symmetric)
DATASET  = 'metrics'                   # hdf5 dataset name for the sweep curves
COIL_TAG_SCALP = 1005                  # charm scalp surface tag
COIL_TAG_GM    = 1002                  # charm GM surface tag

FIELDS = ['subject', 'target', 'exclusion_note', 'didt_Aus', 'rMT', 'status',
          'coil_to_scalp_orig', 'coil_to_cortex_orig', 'scd',
          'csd_mm', 'ccd_new', 'c2s_new_check', 'scd_new_check', 'shift_mm',
          'new_coil_x', 'new_coil_y', 'new_coil_z',
          'handle_angle_orig_rel_anterior',
          'theta_star_deg', 'handle_angle_opt_rel_anterior',
          'ef99p9_orig_u', 'emax_orig_u', 'global_ef99p9_orig_u',
          'efield_angle_midline_orig_u', 'forward_orig_u',
          'ef99p9_opt_u', 'emax_opt_u', 'global_ef99p9_opt_u',
          'efield_angle_midline_opt_u', 'forward_opt_u',
          'delta_ef99p9_opt_minus_orig_u', 'artifact_check_v_per_m',
          'notes']


# ---------------------------------------------------------------------------
# Input: double-header cohort CSV (row 1 = group labels, row 2 = names)
# ---------------------------------------------------------------------------
def load_subjects(input_csv, targets=TARGETS):
    """Rows: <ID>, dI/dt-rMT (A/us), {T}-x/y/z, {T}-x/y/z .2 per target."""
    with open(input_csv, encoding='utf-8-sig') as f:
        lines = f.read().splitlines()
    header = [h.strip() for h in lines[1].split(',')]
    subs = []
    for line in lines[2:]:
        c = [x.strip() for x in line.split(',')]
        if not c or not c[0].startswith('TC'):
            continue
        d = dict(zip(header, c))
        entry = {'subject': c[0],
                 'intensity': float(d['dI/dt-rMT']),
                 'rMT': d.get('rMT', '')}
        for t in targets:
            entry[t] = {
                'pos': [float(d[f'{t}-x']), float(d[f'{t}-y']), float(d[f'{t}-z'])],
                'dir': [float(d[f'{t}-x .2']), float(d[f'{t}-y .2']),
                        float(d[f'{t}-z .2'])]}
        subs.append(entry)
    return subs


# ---------------------------------------------------------------------------
# Mesh helpers / geometry (final rules)
# ---------------------------------------------------------------------------
def build_trimesh_from_tag(mesh, tag):
    """trimesh.Trimesh from all triangles of a charm surface tag."""
    import trimesh
    sub = mesh.crop_mesh(tags=[tag])
    verts = np.asarray(sub.nodes.node_coord)
    tri_mask = sub.elm.elm_type == 2
    tris = sub.elm.node_number_list[tri_mask][:, :3] - 1
    return trimesh.Trimesh(vertices=verts, faces=tris, process=False)


def build_matsimnibs_final(raw_pos, raw_dir, gm_target):
    """4x4 matsimnibs, final rule: two-point Z, YZ-swap handle, auto-flip."""
    sim_pos = np.array([-raw_pos[0], -raw_pos[1], raw_pos[2]])
    traj = np.asarray(gm_target) - sim_pos
    Z = traj / np.linalg.norm(traj)
    Y = np.array([-raw_dir[0], raw_dir[2], raw_dir[1]])   # YZ swap (-x, z, y)
    Y = Y - np.dot(Y, Z) * Z
    if np.linalg.norm(Y) < 1e-6:
        Y = np.array([0., 1., 0.])
        Y = Y - np.dot(Y, Z) * Z
    Y /= np.linalg.norm(Y)
    X = np.cross(Y, Z)
    X /= np.linalg.norm(X)
    m = np.eye(4)
    m[:3, 0] = X; m[:3, 1] = Y; m[:3, 2] = Z; m[:3, 3] = sim_pos
    if X[1] < 0:                                          # E-field forward
        m = m @ np.diag([-1., -1., 1., 1.])
    return m


def signed_line_hits(tm, point, direction):
    """All line-surface intersections both ways; (signed_t, hit_point)."""
    out = []
    for sgn in (1.0, -1.0):
        locs, _, _ = tm.ray.intersects_location(
            ray_origins=np.array([point]), ray_directions=np.array([sgn * direction]))
        for p in locs:
            out.append((sgn * float(np.linalg.norm(p - point)), p))
    return out


def compute_geometry_signed(matsimnibs, scalp_tm, gm_tm):
    """Signed c2s / c2c along the coil axis; SCD = |c2c - c2s| strictly."""
    sim_pos = matsimnibs[:3, 3]
    Z = matsimnibs[:3, 2]
    def nearest(tm):
        hits = signed_line_hits(tm, sim_pos, Z)
        return min(hits, key=lambda h: abs(h[0])) if hits else (None, None)
    c2s, scalp_pt = nearest(scalp_tm)
    c2c, cortex_pt = nearest(gm_tm)
    scd = abs(c2c - c2s) if (c2c is not None and c2s is not None) else None
    return c2s, c2c, scd, scalp_pt, cortex_pt


def parent_tet_map(mesh, tri_pos):
    """0-based element position of the parent GM tet (tag 2) per GM triangle.

    fem.calc_fields assigns surface triangles the value of the corresponding
    tetrahedron; the raw tms_many_simulations path fills triangles with
    in-plane gradients + node-averaged dA/dt instead (~2% different). Reading
    E at the parent GM tet reproduces the official output definition exactly.
    """
    nl = mesh.elm.node_number_list
    if mesh.nodes.nr >= 2 ** 21:
        raise ValueError('node count exceeds 21-bit key space; widen shifts')
    gm_tet_pos = np.where((mesh.elm.elm_type == 4) & (mesh.elm.tag1 == 2))[0]
    tet_nodes = nl[gm_tet_pos][:, :4] - 1
    faces = np.concatenate(
        [tet_nodes[:, s] for s in ([1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2])],
        axis=0)
    faces = np.take_along_axis(faces, np.argsort(faces, axis=1), axis=1)
    owner = np.concatenate([gm_tet_pos] * 4)     # block-aligned like `faces`
    fk = (faces[:, 0].astype(np.int64) << 42) | \
         (faces[:, 1].astype(np.int64) << 21) | faces[:, 2].astype(np.int64)
    order = np.argsort(fk)
    fk, owner = fk[order], owner[order]
    tri_nodes = nl[tri_pos][:, :3] - 1
    tri_s = np.take_along_axis(tri_nodes, np.argsort(tri_nodes, axis=1), axis=1)
    tk = (tri_s[:, 0].astype(np.int64) << 42) | \
         (tri_s[:, 1].astype(np.int64) << 21) | tri_s[:, 2].astype(np.int64)
    loc = np.clip(np.searchsorted(fk, tk), 0, len(fk) - 1)
    return np.where(fk[loc] == tk, owner[loc], -1)


# ---------------------------------------------------------------------------
# Angle machinery
# ---------------------------------------------------------------------------
def rotate_about_coil_z(mat, theta_deg, new_pos):
    """Yaw the coil by theta about its own Z axis; centre -> new_pos."""
    c, s = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
    Rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    m = mat.copy()
    m[:3, :3] = mat[:3, :3] @ Rz
    m[:3, 3] = new_pos
    return m


def handle_angle_rel_anterior(mat):
    """Signed yaw of the coil handle vs the anterior seed (+Y reference)."""
    Z, Y = mat[:3, 2], mat[:3, 1]
    y_seed = np.array([0., 1., 0.])
    y_ref = y_seed - np.dot(y_seed, Z) * Z
    n = np.linalg.norm(y_ref)
    if n < 1e-9:
        return None
    y_ref /= n
    return float(np.degrees(np.arctan2(np.dot(np.cross(y_ref, Y), Z),
                                       np.dot(y_ref, Y))))


def angle_to_midline(E_unit):
    """(angle to midline, forward?) from a unit mean hotspot E vector."""
    if E_unit is None:
        return None, None
    y = abs(E_unit[1])
    return float(np.degrees(np.arccos(np.clip(y, 0, 1)))), bool(E_unit[1] > 0)


def dir_metrics(dirv):
    if dirv is None or np.linalg.norm(dirv) < 1e-9:
        return None, None
    return angle_to_midline(np.asarray(dirv) / np.linalg.norm(dirv))


# ---------------------------------------------------------------------------
# Sweep: real FEM per angle, compact hdf5 curve (one shared factorisation)
# ---------------------------------------------------------------------------
def sweep_exists(fn_hdf5):
    return os.path.exists(fn_hdf5) and os.path.exists(fn_hdf5 + '.done')


def read_sweep(fn_hdf5):
    with h5py.File(fn_hdf5, 'r') as f:
        return f[DATASET][:]                      # (n_angles, 6)


def run_sweep(mesh, cond_field, fn_coil, mats, didt, fn_hdf5,
              use_idx, d5, d10, angles=ANGLES):
    """36-angle sweep; stores [ef99p9, emax, global_ef99p9, dir(3)] per angle.

    use_idx : element positions carrying each GM triangle's E value
              (parent GM tet where found, else the triangle itself)
    d5/d10  : boolean masks over GM triangles (5 mm / 10 mm of gm_target)
    """
    def post_pro(E):
        Egm = E[use_idx]
        mags = np.linalg.norm(Egm, axis=1)
        local = mags[d5]
        if local.size == 0:
            return np.zeros(6)
        e99 = float(np.percentile(local, 99.9)) if local.size > 10 \
            else float(local.max())
        emax = float(local.max())
        g99 = float(np.percentile(mags, 99.9))
        m10, v10 = mags[d10], Egm[d10]
        dirv = np.zeros(3)
        if m10.size >= 5:
            thr = np.percentile(m10, 90)
            meanE = v10[m10 >= thr].mean(axis=0)
            n = np.linalg.norm(meanE)
            if n > 1e-9:
                dirv = meanE / n
        return np.array([e99, emax, g99, dirv[0], dirv[1], dirv[2]])

    fem.tms_many_simulations(
        mesh, cond_field, fn_coil, mats, [didt] * len(mats),
        fn_hdf5, DATASET, roi=None, field='E', post_pro=post_pro,
        solver_options=None, n_workers=1)
    with open(fn_hdf5 + '.done', 'w') as f:
        f.write(f'{len(mats)} sims, angles {angles[0]}..{angles[-1]} deg\n')


# ---------------------------------------------------------------------------
# EF extraction from an official run_simnibs result mesh (validation / audit)
# ---------------------------------------------------------------------------
def get_hotspot_efield(msh_path, gm_target, radius=5.0):
    m = mesh_io.read_msh(msh_path)
    E = m.field.get('E')
    if not E:
        del m
        return None
    e_vals = np.asarray(E.value)
    e_mags = np.linalg.norm(e_vals, axis=1)
    nodes = np.asarray(m.nodes.node_coord)
    nl = m.elm.node_number_list
    gm_mask = m.elm.tag1 == COIL_TAG_GM
    gm_bc = nodes[nl[gm_mask][:, :3] - 1].mean(axis=1)
    gm_e, gm_e_vecs = e_mags[gm_mask], e_vals[gm_mask]
    dists = np.linalg.norm(gm_bc - gm_target, axis=1)
    local_mags, local_vecs = gm_e[dists <= radius], gm_e_vecs[dists <= radius]
    if len(local_mags) == 0:
        del m
        return None
    out = {'emax': float(local_mags.max()),
           'ef99p9': float(np.percentile(local_mags, 99.9))
               if len(local_mags) > 10 else float(local_mags.max()),
           'global_ef99p9': float(np.percentile(gm_e, 99.9))}
    near10 = dists <= 10.0
    l10_mags, l10_vecs = gm_e[near10], gm_e_vecs[near10]
    if len(l10_mags) >= 5:
        thr = np.percentile(l10_mags, 90)
        mean_E = l10_vecs[l10_mags >= thr].mean(axis=0)
        n = np.linalg.norm(mean_E)
        if n > 1e-9:
            out['E_dir'] = mean_E / n
    del m
    return out
