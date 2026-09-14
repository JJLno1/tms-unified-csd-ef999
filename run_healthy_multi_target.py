#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_healthy_multi_target.py — unified-CSD analysis for HCP-style cohorts
with 4 targets per subject: individualised M1/DLPFC (from operator Excel)
plus standard C3/F3 (from charm EEG 10-10 positions), at a uniform dose.

Per-subject × target:
  * M1 / DLPFC: scalp point from the cohort Excel (mesh RAS coords);
    gm_target = nearest GM node; CSD = 2 mm (M1) / 1 mm (DLPFC).
  * C3 / F3: charm EEG position; gm_target = nearest GM node;
    CSD = 2 mm (C3) / 1 mm (F3).
  * All targets: uniform dI/dt (default 75 A/us), base frame Y = anterior
    seed, 36-yaw real-FEM sweep (0-175 deg step 5), EF99.9 objective,
    optimal reported as forward-E twin.

Example:
  python run_healthy_multi_target.py \\
      --mesh-base "E:/project/HCP_Project/White" \\
      --excel     "White M1 DLPFC.xlsx" \\
      --out-dir   output/white_results
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
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_csd_core as core
from run_standard_targets import (read_eeg_position, log_factory,
                                   sweep_exists, run_sweep, read_sweep,
                                   find_coil, FIELDS)
from simnibs.mesh_tools import mesh_io
from simnibs.utils import cond_utils
from scipy.spatial import cKDTree

TARGETS = ['M1', 'C3', 'DLPFC', 'F3']
CSD_MAP = {'M1': 2.0, 'C3': 2.0, 'DLPFC': 1.0, 'F3': 1.0}
DIDT = 75.0


def load_excel_positions(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(h).strip() for h in rows[0]]
    out = {}
    for r in rows[1:]:
        d = dict(zip(hdr, r))
        out[str(d['ID'])] = {
            'M1': np.array([float(d['M1-X']), float(d['M1-Y']), float(d['M1-Z'])]),
            'DLPFC': np.array([float(d['DLPFC-X']), float(d['DLPFC-Y']),
                               float(d['DLPFC-Z'])])}
    wb.close()
    return out


def process_subject(args):
    (sub, mesh_path, excel_pos, out_dir, coil_path, didt) = args
    sub_dir = os.path.join(out_dir, str(sub))
    os.makedirs(sub_dir, exist_ok=True)
    log = log_factory(os.path.join(out_dir, 'healthy_batch.log'))
    log(f'START {sub}')
    rows = []
    try:
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
        eeg_csv = os.path.join(os.path.dirname(mesh_path),
                               'eeg_positions', 'EEG10-10_Neuroelectrics.csv')

        for t in TARGETS:
            csd = CSD_MAP[t]
            if t in ('M1', 'DLPFC'):
                scalp = np.asarray(excel_pos[str(sub)][t], float)
                src = 'excel'
            else:
                scalp = read_eeg_position(eeg_csv, t)
                src = 'eeg10-10'
            gm_target = gm_nodes[tree.query(scalp)[1]]
            Z = gm_target - scalp; Z /= np.linalg.norm(Z)
            new_pos = scalp - csd * Z
            Y = np.array([0., 1., 0.]); Y -= Y.dot(Z) * Z
            Y /= np.linalg.norm(Y)
            X = np.cross(Y, Z); X /= np.linalg.norm(X)
            mat = np.eye(4)
            mat[:3, 0], mat[:3, 1], mat[:3, 2], mat[:3, 3] = \
                X, Y, Z, new_pos

            scd = float(np.linalg.norm(gm_target - scalp))
            ccd = scd + csd
            dists = np.linalg.norm(gm_bc - gm_target, axis=1)
            d5, d10 = dists <= 5.0, dists <= 10.0
            fn_hdf5 = os.path.join(sub_dir, f'{t}_sweep.hdf5')
            if not sweep_exists(fn_hdf5):
                for f in (fn_hdf5, fn_hdf5 + '.done'):
                    if os.path.exists(f):
                        os.remove(f)
                mats = [core.rotate_about_coil_z(mat, th, new_pos)
                        for th in core.ANGLES]
                t0 = time.time()
                log(f'[{sub}] {t}: sweep FEM (CSD={csd}mm)')
                run_sweep(mesh, cond_field, coil_path, mats, didt * 1e6,
                          fn_hdf5, use_idx, d5, d10)
                log(f'[{sub}] {t}: done {(time.time()-t0)/60:.1f} min')
            curve = read_sweep(fn_hdf5)
            with open(os.path.join(sub_dir, f'{t}_sweep.csv'), 'w',
                      newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['theta_deg','ef99p9','emax','global_ef99p9',
                            'dir_x','dir_y','dir_z'])
                for th, r in zip(core.ANGLES, curve):
                    w.writerow([th] + [round(float(x), 4) for x in r])
            i_star = int(np.argmax(curve[:, 0]))
            rS = curve[i_star]
            dS = np.asarray(rS[3:6], float)
            if dS[1] < 0:
                dS = -dS
            theta_star = int(core.ANGLES[i_star])
            rows.append(dict(
                subject=str(sub), target=t, csd_mm=csd, didt_Aus=didt,
                status='OK', scd=round(scd, 2), ccd_new=round(ccd, 2),
                theta_star_deg=theta_star,
                handle_angle_opt_rel_anterior=(
                    theta_star if rS[4] > 0 else theta_star - 180),
                ef99p9_opt=round(float(rS[0]), 1),
                emax_opt=round(float(rS[1]), 1),
                global_ef99p9_opt=round(float(rS[2]), 1),
                efield_angle_midline_opt='',
                forward_opt='Y',
                e_dir_opt_x=round(float(dS[0]), 3),
                e_dir_opt_y=round(float(dS[1]), 3),
                e_dir_opt_z=round(float(dS[2]), 3),
                ef99p9_at_handle0=round(float(curve[0, 0]), 1),
                delta_opt_vs_handle0=round(float(rS[0] - curve[0, 0]), 1),
                gm_target_x=round(float(gm_target[0]), 1),
                gm_target_y=round(float(gm_target[1]), 1),
                gm_target_z=round(float(gm_target[2]), 1),
                notes=f'source={src}'))
            log(f'[{sub}] {t}: scd={scd:.2f} theta*={theta_star} '
                f'EF99.9={rS[0]:.1f}')
        del mesh, cond_field
    except Exception as e:
        log(f'FAIL {sub}: {e}')
        log(traceback.format_exc())
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mesh-base', required=True)
    ap.add_argument('--excel', required=True,
                    help='xlsx with ID, M1-X/Y/Z, DLPFC-X/Y/Z columns')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--didt', type=float, default=DIDT)
    ap.add_argument('--coil', default=find_coil())
    ap.add_argument('--cases', default='')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log = log_factory(os.path.join(args.out_dir, 'healthy_batch.log'))
    excel_pos = load_excel_positions(args.excel)

    tasks = []
    for d in sorted(os.listdir(args.mesh_base)):
        m2m = os.path.join(args.mesh_base, d, f'm2m_{d}')
        mp = os.path.join(m2m, f'{d}.msh')
        if os.path.isfile(mp) and d in excel_pos:
            tasks.append((d, mp, excel_pos, args.out_dir, args.coil,
                          args.didt))
    if args.cases:
        w = {c.strip() for c in args.cases.split(',')}
        tasks = [t for t in tasks if t[0] in w]
    log(f'{len(tasks)} subjects, {len(TARGETS)} targets each')

    todo = [t for t in tasks
            if not all(sweep_exists(os.path.join(
                args.out_dir, str(t[0]), f'{tg}_sweep.hdf5'))
                for tg in TARGETS)]
    log(f'{len(todo)} need sweeps')

    all_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_subject, t): t[0] for t in todo}
        for fut in as_completed(futs):
            try:
                all_rows.extend(fut.result())
            except Exception as e:
                log(f'[{futs[fut]}] crashed: {e}')
    for t in [t for t in tasks if t not in todo]:
        all_rows.extend(process_subject(t))

    df = pd.DataFrame(all_rows).reindex(columns=FIELDS)
    long_csv = os.path.join(args.out_dir, 'healthy_multi_target_long.csv')
    df.to_csv(long_csv, index=False)
    # summary
    ok = df[df.status == 'OK']
    lines = ['# Healthy multi-target results', '']
    for t in TARGETS:
        g = ok[ok.target == t]
        lines.append(f'- {t}: SCD {g.scd.mean():.1f}±{g.scd.std():.1f} | '
                      f'EF99.9 opt {g.ef99p9_opt.mean():.1f}±'
                      f'{g.ef99p9_opt.std():.1f} | '
                      f'theta* median {g.theta_star_deg.median():.0f}')
    with open(os.path.join(args.out_dir, 'healthy_REPORT.md'), 'w') as f:
        f.write('\n'.join(lines))
    log(f'DONE -> {long_csv}')


if __name__ == '__main__':
    main()
