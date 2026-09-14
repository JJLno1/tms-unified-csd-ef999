#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_csd_sensitivity.py — sensitivity of EF99.9 to coil-to-scalp distance.

Runs the full 36-angle sweep at multiple CSD values (default 1–4 mm) on a
representative subset of subjects, then reports per-CSD metrics and paired
adjacent-CSD differences with Wilcoxon tests.

Subset selection: systematic rank-based (every ~Nth subject by ID,
deterministic). Override with --cases.

Example:
  python run_csd_sensitivity.py \\
      --mesh-base "C:/data/NIfTI_anoymous" \\
      --input-csv  originalInput_newcohort.csv \\
      --out-dir    output/csd_sensitivity \\
      --n-subjects 10  --csd-values 1,2,3,4
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "10")
import sys
import csv
import time
import argparse
import traceback
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_csd_core as core
from run_standard_targets import (log_factory, sweep_exists, run_sweep,
                                   read_sweep, find_coil)
from simnibs.mesh_tools import mesh_io
from simnibs.utils import cond_utils
from scipy.spatial import cKDTree

TARGETS = ['M1', 'DLPFC']
DEFAULT_CSDS = [1.0, 2.0, 3.0, 4.0]
DEFAULT_REF = {'M1': 2.0, 'DLPFC': 1.0}

FIELDS = ['subject', 'target', 'csd_mm', 'scd', 'ccd_new',
          'theta_star_deg', 'ef99p9_orig', 'ef99p9_opt',
          'pct_vs_ref_orig', 'pct_vs_ref_opt', 'notes']


def load_subjects(input_csv, targets):
    with open(input_csv, encoding='utf-8-sig') as f:
        lines = f.read().splitlines()
    hdr = [h.strip() for h in lines[1].split(',')]
    subs = []
    for line in lines[2:]:
        c = [x.strip() for x in line.split(',')]
        if not c or not c[0].startswith('TC'):
            continue
        d = dict(zip(hdr, c))
        entry = {'subject': c[0], 'num': c[0][2:] if c[0][2:].isdigit() else c[0],
                 'intensity': float(d['dI/dt-rMT'])}
        for t in targets:
            entry[t] = {'pos': [float(d[f'{t}-x']), float(d[f'{t}-y']),
                                float(d[f'{t}-z'])],
                        'dir': [float(d[f'{t}-x .2']), float(d[f'{t}-y .2']),
                                float(d[f'{t}-z .2'])]}
        subs.append(entry)
    return subs


def pick_subjects(entries, n):
    subs = sorted(e['subject'] for e in entries)
    idx = sorted({round(i * (len(subs) - 1) / (n - 1)) for i in range(n)})
    chosen = [subs[i] for i in idx]
    k = 0
    while len(chosen) < n:
        if subs[k] not in chosen:
            chosen.append(subs[k])
        k += 1
    return [e for e in entries if e['subject'] in sorted(chosen)]


def process_subject(entry, mesh_base, out_dir, coil_path, csds, ref_csd):
    sub = entry['subject']
    num = entry['num']
    sub_dir = os.path.join(out_dir, sub)
    os.makedirs(sub_dir, exist_ok=True)
    log = log_factory(os.path.join(out_dir, 'csd_sensitivity.log'))
    rows = []
    try:
        mesh_path = os.path.join(mesh_base, num, f'm2m_{num}', f'{num}.msh')
        mesh = mesh_io.read_msh(mesh_path)
        gm_nodes = np.asarray(mesh.crop_mesh(tags=[1002]).nodes.node_coord)
        tree = cKDTree(gm_nodes)
        cond_field = cond_utils.cond2elmdata(
            mesh, [c.value for c in cond_utils.standard_cond()])
        nodes = np.asarray(mesh.nodes.node_coord)
        nl = mesh.elm.node_number_list
        gm_idx = np.where(mesh.elm.tag1 == 1002)[0]
        gm_bc = nodes[nl[gm_idx][:, :3] - 1].mean(axis=1)
        parent = core.parent_tet_map(mesh, gm_idx)
        use_idx = np.where(parent >= 0, parent, gm_idx)

        for t in TARGETS:
            raw = entry[t]
            sim_pos = np.array([-raw['pos'][0], -raw['pos'][1], raw['pos'][2]])
            gm_target = gm_nodes[tree.query(sim_pos)[1]]
            mat = core.build_matsimnibs_final(raw['pos'], raw['dir'], gm_target)
            Z = mat[:3, 2]
            # scalp hit from base frame
            import trimesh
            scalp_tm = core.build_trimesh_from_tag(mesh, 1005)
            gm_tm = core.build_trimesh_from_tag(mesh, 1002)
            c2s, c2c, scd, scalp_pt, _ = core.compute_geometry_signed(
                mat, scalp_tm, gm_tm)

            dists = np.linalg.norm(gm_bc - gm_target, axis=1)
            d5, d10 = dists <= 5.0, dists <= 10.0
            curves = {}
            for csd in csds:
                fn = os.path.join(sub_dir, f'{t}_csd{csd:g}.hdf5')
                if not sweep_exists(fn):
                    new_pos = np.asarray(scalp_pt) - csd * Z
                    mats = [core.rotate_about_coil_z(mat, th, new_pos)
                            for th in core.ANGLES]
                    t0 = time.time()
                    log(f'[{sub}] {t} CSD={csd:g}mm: sweep')
                    run_sweep(mesh, cond_field, coil_path, mats,
                              entry['intensity'] * 1e6, fn, use_idx, d5, d10)
                    log(f'[{sub}] {t} CSD={csd:g}mm: {(time.time()-t0)/60:.1f} min')
                curves[csd] = read_sweep(fn)

            # reference curve
            ref = ref_csd[t]
            c_ref = curves[ref]
            th_ref = int(np.argmax(c_ref[:, 0]))
            e_ref_o = c_ref[0, 0]
            e_ref_s = c_ref[th_ref, 0]

            for csd in csds:
                c = curves[csd]
                th = int(np.argmax(c[:, 0]))
                rows.append(dict(
                    subject=sub, target=t, csd_mm=csd,
                    scd=round(scd, 2), ccd_new=round(scd + csd, 2),
                    theta_star_deg=int(core.ANGLES[th]),
                    ef99p9_orig=round(float(c[0, 0]), 1),
                    ef99p9_opt=round(float(c[th, 0]), 1),
                    pct_vs_ref_orig=round(100 * (c[0, 0] - e_ref_o) / e_ref_o, 1),
                    pct_vs_ref_opt=round(100 * (c[th, 0] - e_ref_s) / e_ref_s, 1),
                    notes=''))
        del mesh, cond_field
    except Exception as e:
        log(f'FAIL {sub}: {e}')
        log(traceback.format_exc())
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mesh-base', required=True)
    ap.add_argument('--input-csv', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n-subjects', type=int, default=10)
    ap.add_argument('--csd-values', default='1,2,3,4')
    ap.add_argument('--m1-ref-csd', type=float, default=2.0)
    ap.add_argument('--dlpfc-ref-csd', type=float, default=1.0)
    ap.add_argument('--coil', default=find_coil())
    ap.add_argument('--cases', default='')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    csds = [float(x) for x in args.csd_values.split(',')]
    ref_csd = {'M1': args.m1_ref_csd, 'DLPFC': args.dlpfc_ref_csd}
    os.makedirs(args.out_dir, exist_ok=True)
    log = log_factory(os.path.join(args.out_dir, 'csd_sensitivity.log'))

    entries = load_subjects(args.input_csv, TARGETS)
    if args.cases:
        w = {c.strip() for c in args.cases.split(',')}
        entries = [e for e in entries if e['subject'] in w]
    else:
        entries = pick_subjects(entries, args.n_subjects)
    log(f'Subset ({len(entries)}): {[e["subject"] for e in entries]}')

    all_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_subject, e, args.mesh_base, args.out_dir,
                          args.coil, csds, ref_csd) for e in entries]
        for f in futs:
            all_rows.extend(f.result())

    df = pd.DataFrame(all_rows).reindex(columns=FIELDS)
    long_csv = os.path.join(args.out_dir, 'csd_sensitivity_long.csv')
    df.to_csv(long_csv, index=False)

    # summary report
    from scipy import stats as st
    lines = ['# CSD sensitivity', '',
             f'Subjects: {df.subject.nunique()}, CSDs: {csds}', '']
    for t in TARGETS:
        g = df[df.target == t]
        lines.append(f'## {t} (ref CSD={ref_csd[t]:g} mm)')
        lines.append('| CSD | EF99.9 orig | EF99.9 opt | %vs-ref opt | theta* |')
        lines.append('|-----|------------|------------|-------------|--------|')
        for csd in csds:
            c = g[g.csd_mm == csd]
            lines.append(
                f'| {csd:g} | {c.ef99p9_orig.mean():.1f}±{c.ef99p9_orig.std():.1f} '
                f'| {c.ef99p9_opt.mean():.1f}±{c.ef99p9_opt.std():.1f} '
                f'| {c.pct_vs_ref_opt.mean():+.1f}±{c.pct_vs_ref_opt.std():.1f}% '
                f'| {c.theta_star_deg.median():.0f} |')
        lines.append('')
        piv = g.pivot_table(index='subject', columns='csd_mm',
                            values='ef99p9_opt')
        for a, b in zip(csds, csds[1:]):
            d = (piv[b] - piv[a]).dropna()
            if len(d) >= 3:
                p = st.wilcoxon(d).pvalue if (d != 0).any() else 1.0
                lines.append(f'- {a:g}→{b:g} mm: {d.mean():+.1f} V/m '
                             f'({100*d.mean()/piv[a].mean():+.1f}%), p={p:.4f}')
        lines.append('')
    with open(os.path.join(args.out_dir, 'csd_sensitivity_REPORT.md'), 'w') as f:
        f.write('\n'.join(lines))
    log(f'DONE -> {long_csv}')


if __name__ == '__main__':
    main()
