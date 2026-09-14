#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_standard_targets.py — unified-CSD analysis for standard EEG-position
targets (C3, F3, or any charm EEG 10-10 position).

Differences from the clinical M1/DLPFC workflow (run_unified_csd.py):
  * Target = charm eeg_positions/EEG10-10_Neuroelectrics.csv scalp point
    (RAS mesh coords); gm_target = nearest GM-surface node.
  * Base frame Y = anterior seed [0,1,0] projected in-plane (no operator
    direction exists) -> theta* is directly the optimal handle angle
    relative to the sagittal midline.
  * Dose: uniform dI/dt for all subjects (default 75 A/us).
  * CSD map: C3 = 2 mm, F3 = 1 mm (customisable).

Works with any charm m2m directory structure (clinical or HCP-style).

Example:
  python run_standard_targets.py \\
      --mesh-base "C:/data/TMS-SimNIBS" \\
      --out-dir    output/c3f3_results \\
      --targets    C3,F3  --didt 75
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "10")
import sys
import csv
import glob
import time
import argparse
import traceback
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_csd_core as core
from simnibs.mesh_tools import mesh_io
from simnibs.utils import cond_utils
from scipy.spatial import cKDTree

DEFAULT_CSD = {'C3': 2.0, 'F3': 1.0, 'C4': 2.0, 'F4': 1.0,
               'M1': 2.0, 'DLPFC': 1.0}
DEFAULT_DIDT = 75.0
ANGLES = core.ANGLES if hasattr(core, 'ANGLES') else list(range(0, 180, 5))

FIELDS = ['subject', 'target', 'csd_mm', 'didt_Aus', 'status',
          'scd', 'ccd_new', 'theta_star_deg', 'handle_angle_opt_rel_anterior',
          'ef99p9_opt', 'emax_opt', 'global_ef99p9_opt',
          'efield_angle_midline_opt', 'forward_opt',
          'e_dir_opt_x', 'e_dir_opt_y', 'e_dir_opt_z',
          'ef99p9_at_handle0', 'delta_opt_vs_handle0',
          'gm_target_x', 'gm_target_y', 'gm_target_z', 'notes']


def read_eeg_position(csv_path, name):
    with open(csv_path, newline='', encoding='utf-8') as f:
        for r in csv.reader(f):
            if len(r) >= 5 and r[4].strip().upper() == name.upper():
                return np.asarray([float(r[1]), float(r[2]), float(r[3])])
    raise ValueError(name)


def log_factory(path):
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except Exception:
            pass
    return log


def sweep_exists(fn):
    return os.path.exists(fn) and os.path.exists(fn + '.done')


def run_sweep(mesh, cond_field, fn_coil, mats, didt, fn_hdf5, use_idx, d5, d10):
    import h5py
    from simnibs.simulation import fem

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
        fn_hdf5, 'metrics', roi=None, field='E', post_pro=post_pro,
        solver_options=None, n_workers=1)
    with open(fn_hdf5 + '.done', 'w') as f:
        f.write(f'{len(mats)} sims\n')


def read_sweep(fn):
    import h5py
    with h5py.File(fn, 'r') as f:
        return f['metrics'][:]


def process_subject(args):
    sub, mesh_path, targets, csd_map, didt, out_dir, coil_path = args
    sub_dir = os.path.join(out_dir, str(sub))
    os.makedirs(sub_dir, exist_ok=True)
    log = log_factory(os.path.join(out_dir, 'standard_targets.log'))
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
        for t in targets:
            csd = csd_map.get(t, 2.0)
            eeg = read_eeg_position(eeg_csv, t)
            gm_target = gm_nodes[tree.query(eeg)[1]]
            Z = gm_target - eeg; Z /= np.linalg.norm(Z)
            new_pos = eeg - csd * Z
            Y = np.array([0., 1., 0.]); Y -= Y.dot(Z) * Z; Y /= np.linalg.norm(Y)
            X = np.cross(Y, Z); X /= np.linalg.norm(X)
            mat = np.eye(4)
            mat[:3, 0], mat[:3, 1], mat[:3, 2], mat[:3, 3] = X, Y, Z, new_pos

            scd = float(np.linalg.norm(gm_target - eeg))
            ccd = scd + csd
            dists = np.linalg.norm(gm_bc - gm_target, axis=1)
            d5, d10 = dists <= 5.0, dists <= 10.0
            fn_hdf5 = os.path.join(sub_dir, f'{t}_sweep.hdf5')
            if not sweep_exists(fn_hdf5):
                for f in (fn_hdf5, fn_hdf5 + '.done'):
                    if os.path.exists(f):
                        os.remove(f)
                mats = [core.rotate_about_coil_z(mat, th, new_pos)
                        for th in ANGLES]
                t0 = time.time()
                log(f'[{sub}] {t}: sweep FEM ({len(ANGLES)} angles, CSD={csd}mm)')
                run_sweep(mesh, cond_field, coil_path, mats, didt * 1e6,
                          fn_hdf5, use_idx, d5, d10)
                log(f'[{sub}] {t}: done in {(time.time()-t0)/60:.1f} min')
            curve = read_sweep(fn_hdf5)
            with open(os.path.join(sub_dir, f'{t}_sweep.csv'), 'w',
                      newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['theta_deg','ef99p9','emax','global_ef99p9',
                            'dir_x','dir_y','dir_z'])
                for th, r in zip(ANGLES, curve):
                    w.writerow([th] + [round(float(x), 4) for x in r])
            i_star = int(np.argmax(curve[:, 0]))
            rS = curve[i_star]
            dS = np.asarray(rS[3:6], float)
            forward = dS[1] > 0
            if not forward:
                dS = -dS
            ang, _ = core.dir_metrics(rS[3:6]) if hasattr(core, 'dir_metrics') \
                else (None, None)
            theta_star = int(ANGLES[i_star])
            rows.append(dict(
                subject=str(sub), target=t, csd_mm=csd, didt_Aus=didt,
                status='OK', scd=round(scd, 2), ccd_new=round(ccd, 2),
                theta_star_deg=theta_star,
                handle_angle_opt_rel_anterior=(
                    theta_star if forward else theta_star - 180),
                ef99p9_opt=round(float(rS[0]), 1),
                emax_opt=round(float(rS[1]), 1),
                global_ef99p9_opt=round(float(rS[2]), 1),
                efield_angle_midline_opt=round(ang, 1) if ang else '',
                forward_opt='Y',
                e_dir_opt_x=round(float(dS[0]), 3),
                e_dir_opt_y=round(float(dS[1]), 3),
                e_dir_opt_z=round(float(dS[2]), 3),
                ef99p9_at_handle0=round(float(curve[0, 0]), 1),
                delta_opt_vs_handle0=round(float(rS[0] - curve[0, 0]), 1),
                gm_target_x=round(float(gm_target[0]), 1),
                gm_target_y=round(float(gm_target[1]), 1),
                gm_target_z=round(float(gm_target[2]), 1), notes=''))
            log(f'[{sub}] {t}: scd={scd:.2f} | theta*={theta_star} '
                f'EF99.9 opt={rS[0]:.1f}')
        del mesh, cond_field
    except Exception as e:
        log(f'FAIL {sub}: {e}')
        log(traceback.format_exc())
        rows = []
    return rows


def find_coil():
    d = os.path.dirname(cond_utils.__file__)
    return os.path.abspath(os.path.join(
        d, '..', 'resources', 'coil_models', 'legacy_and_other',
        'Magstim_70mm_Fig8.ccd'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mesh-base', required=True,
                    help='dir containing <id>/m2m_<id>/<id>.msh')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--targets', default='C3,F3')
    ap.add_argument('--didt', type=float, default=DEFAULT_DIDT)
    ap.add_argument('--m1-csd', type=float, default=2.0)
    ap.add_argument('--dlpfc-csd', type=float, default=1.0)
    ap.add_argument('--coil', default=find_coil())
    ap.add_argument('--cases', default='')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(',')]
    csd_map = dict(DEFAULT_CSD)
    csd_map['M1'] = args.m1_csd
    csd_map['DLPFC'] = args.dlpfc_csd
    os.makedirs(args.out_dir, exist_ok=True)
    log = log_factory(os.path.join(args.out_dir, 'standard_targets.log'))

    tasks = []
    for d in sorted(os.listdir(args.mesh_base)):
        m2m = os.path.join(args.mesh_base, d, f'm2m_{d}')
        mp = os.path.join(m2m, f'{d}.msh')
        if os.path.isfile(mp):
            tasks.append((d, mp, targets, csd_map, args.didt,
                          args.out_dir, args.coil))
    if args.cases:
        w = {c.strip() for c in args.cases.split(',')}
        tasks = [t for t in tasks if t[0] in w]
    log(f'{len(tasks)} subjects, targets={targets}, didt={args.didt}')

    todo = [t for t in tasks
            if not all(sweep_exists(os.path.join(
                args.out_dir, str(t[0]), f'{tg}_sweep.hdf5'))
                for tg in targets)]
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
    df.to_csv(os.path.join(args.out_dir, 'standard_targets_long.csv'),
              index=False)
    log(f'DONE -> {os.path.join(args.out_dir, "standard_targets_long.csv")}')


if __name__ == '__main__':
    main()
