#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_unified_csd.py — cohort batch runner for unified-CSD EF99.9 re-analysis.

Per subject and target (M1 / DLPFC):
  1. final-rule matsimnibs + signed geometry (c2s/c2c/SCD along coil axis),
  2. coil centre unified to csd_mm outside the scalp along the coil axis,
  3. real-FEM yaw sweep 0..175 deg step 5 (shared factorisation) with the
     EF99.9-in-5mm objective -> theta*, orig (theta=0) and opt metrics,
  4. optional theta* artifact mesh via the official run_simnibs path +
     cross-validation (artifact_check, expected ~0 V/m).

Idempotent: sweeps/artifacts on disk are reused; rows are always re-extracted.

Example (two validated cohorts):
  python run_unified_csd.py \
      --mesh-base "C:/.../2025_TMS-CBM/TMS-SimNIBS" \
      --input-csv  originalInput.csv \
      --out-dir    output/tc38_unified_csd \
      --art-dir    "D:/project/Nexstim_TC38_unified_csd"
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
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_csd_core as core

from simnibs import sim_struct, run_simnibs
from simnibs.utils import cond_utils
from simnibs.mesh_tools import mesh_io
from scipy.spatial import cKDTree

# default coil model shipped with simnibs (Magstim 70 mm figure-8, dipole ccd)
DEFAULT_COIL = os.path.join(
    os.path.dirname(cond_utils.__file__), '..', 'resources', 'coil_models',
    'legacy_and_other', 'Magstim_70mm_Fig8.ccd')


def log_factory(log_path):
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except Exception:
            pass
    return log


def artifact_exists(art_root, sub):
    d = os.path.join(art_root, sub, 'opt')
    return bool(glob.glob(os.path.join(d, '*_TMS_1-0001_*_scalar.msh'))) \
       and bool(glob.glob(os.path.join(d, '*_TMS_1-0002_*_scalar.msh')))


def run_artifact_session(mesh_path, mats, didt, out_dir, fn_coil):
    """Official-path FEM for the theta* mats (one session, all targets)."""
    os.makedirs(out_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(out_dir, '*')):
        try:
            os.remove(stale)
        except OSError:
            pass
    s = sim_struct.SESSION()
    s.open_in_gmsh = False
    s.map_to_fsavg = False; s.map_to_MNI = False; s.map_to_surf = False
    s.fields = 'eE'
    s.fnamehead = mesh_path
    s.pathfem = out_dir
    tms = s.add_tmslist()
    tms.fnamecoil = fn_coil
    tms.cond = cond_utils.standard_cond()
    for mat in mats:
        p = tms.add_position()
        p.matsimnibs = mat
        p.didt = didt * 1e6
    run_simnibs(s)


def process_subject(entry, mesh_base, out_dir, art_root, fn_coil,
                    csd_mm, targets, angles, exclusion_notes, log):
    sub = entry['subject']
    mesh_path = os.path.join(mesh_base, sub, f'm2m_{sub}', f'{sub}.msh')
    sub_dir = os.path.join(out_dir, sub)
    os.makedirs(sub_dir, exist_ok=True)
    t_start = time.time()
    log(f'========== START {sub} ==========')
    rows = []
    try:
        if not os.path.exists(mesh_path):
            raise RuntimeError(f'no mesh: {mesh_path}')

        mesh = mesh_io.read_msh(mesh_path)
        scalp_tm = core.build_trimesh_from_tag(mesh, core.COIL_TAG_SCALP)
        gm_tm = core.build_trimesh_from_tag(mesh, core.COIL_TAG_GM)
        gm_nodes = np.asarray(mesh.crop_mesh(tags=[core.COIL_TAG_GM]).nodes.node_coord)
        gm_tree = cKDTree(gm_nodes)

        cond_field = cond_utils.cond2elmdata(
            mesh, [c.value for c in cond_utils.standard_cond()])

        nodes = np.asarray(mesh.nodes.node_coord)
        nl = mesh.elm.node_number_list
        gm_idx = np.where(mesh.elm.tag1 == core.COIL_TAG_GM)[0]
        gm_bc = nodes[nl[gm_idx][:, :3] - 1].mean(axis=1)
        parent = core.parent_tet_map(mesh, gm_idx)
        if (parent < 0).any():
            log(f'WARNING: {int((parent < 0).sum())} GM triangles orphan')
        use_idx = np.where(parent >= 0, parent, gm_idx)

        info = {}
        for t in targets:
            csd = csd_mm[t]
            raw = entry[t]
            sim_pos = np.array([-raw['pos'][0], -raw['pos'][1], raw['pos'][2]])
            _, idx = gm_tree.query(sim_pos)
            gm_target = gm_nodes[idx]
            mat = core.build_matsimnibs_final(raw['pos'], raw['dir'], gm_target)
            c2s, c2c, scd, scalp_pt, _ = \
                core.compute_geometry_signed(mat, scalp_tm, gm_tm)
            if c2s is None or scalp_pt is None:
                raise RuntimeError(f'{t}: no scalp hit along coil axis')
            new_pos = np.asarray(scalp_pt) - csd * mat[:3, 2]

            mat0 = core.rotate_about_coil_z(mat, 0.0, new_pos)
            c2s_n, c2c_n, scd_n, _, _ = \
                core.compute_geometry_signed(mat0, scalp_tm, gm_tm)
            notes = []
            if c2s_n is None or abs(c2s_n - csd) > 0.05:
                notes.append(f'c2s_new={c2s_n} (want {csd})')
            if scd_n is None or abs(scd_n - scd) > 0.05:
                notes.append(f'scd_new={scd_n} vs {scd}')
            if c2c_n is None or abs(c2c_n - (scd + csd)) > 0.05:
                notes.append(f'ccd_new={c2c_n} vs {scd + csd}')

            dists = np.linalg.norm(gm_bc - gm_target, axis=1)
            d5, d10 = dists <= 5.0, dists <= 10.0
            fn_hdf5 = os.path.join(sub_dir, f'{t}_sweep.hdf5')
            mats = [core.rotate_about_coil_z(mat, th, new_pos) for th in angles]
            didt = entry['intensity'] * 1e6
            if core.sweep_exists(fn_hdf5):
                log(f'[{sub}] {t}: sweep exists, reusing')
            else:
                for f in (fn_hdf5, fn_hdf5 + '.done'):
                    if os.path.exists(f):
                        os.remove(f)
                t0 = time.time()
                log(f'[{sub}] {t}: angle sweep FEM ({len(angles)} angles, '
                    f'CSD={csd}mm, didt={entry["intensity"]} A/us)')
                core.run_sweep(mesh, cond_field, fn_coil, mats, didt,
                               fn_hdf5, use_idx, d5, d10, angles)
                log(f'[{sub}] {t}: sweep done in {(time.time()-t0)/60:.1f} min')

            curve = core.read_sweep(fn_hdf5)
            with open(os.path.join(sub_dir, f'{t}_sweep.csv'), 'w',
                      newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['theta_deg', 'ef99p9', 'emax', 'global_ef99p9',
                            'dir_x', 'dir_y', 'dir_z'])
                for th, r in zip(angles, curve):
                    w.writerow([th] + [round(float(x), 4) for x in r])

            theta_star = int(angles[int(np.argmax(curve[:, 0]))])
            info[t] = dict(mat=mat, new_pos=new_pos, theta_star=theta_star,
                           curve=curve, c2s=c2s, c2c=c2c, scd=scd,
                           c2s_n=c2s_n, c2c_n=c2c_n, scd_n=scd_n, csd=csd,
                           notes=notes, gm_target=gm_target)
            log(f'[{sub}] {t}: c2s_orig={c2s:.2f} scd={scd:.2f} '
                f'ccd_new={c2c_n:.2f} | theta*={theta_star} '
                f'EF99.9 orig_u={curve[0,0]:.1f} '
                f'opt_u={curve[theta_star//5,0]:.1f}')

        if not artifact_exists(art_root, sub):
            opt_mats = [core.rotate_about_coil_z(info[t]['mat'],
                                                 info[t]['theta_star'],
                                                 info[t]['new_pos'])
                        for t in targets]
            t0 = time.time()
            log(f'[{sub}] theta* artifact FEM ({len(targets)} positions)')
            run_artifact_session(mesh_path, opt_mats, entry['intensity'],
                                 os.path.join(art_root, sub, 'opt'), fn_coil)
            log(f'[{sub}] artifact done in {(time.time()-t0)/60:.1f} min')
        art_dir = os.path.join(art_root, sub, 'opt')

        for i, t in enumerate(targets, 1):
            d = info[t]
            curve, th = d['curve'], d['theta_star']
            r0, rs = curve[0], curve[angles.index(th)]
            fn = glob.glob(os.path.join(art_dir, f'*_TMS_1-000{i}_*_scalar.msh'))
            art_check = ''
            if fn:
                ef_art = core.get_hotspot_efield(fn[0], d['gm_target'])
                if ef_art:
                    art_check = round(abs(ef_art['ef99p9'] - rs[0]), 2)
                    if art_check > max(1.0, 0.01 * rs[0]):
                        d['notes'].append(f'artifact mismatch {art_check} V/m')
            ang_o, fwd_o = core.dir_metrics(r0[3:6])
            ang_s, fwd_s = core.dir_metrics(rs[3:6])
            rows.append({
                'subject': sub, 'target': t,
                'exclusion_note': exclusion_notes.get(sub, ''),
                'didt_Aus': entry['intensity'], 'rMT': entry.get('rMT', ''),
                'status': 'OK',
                'coil_to_scalp_orig': round(d['c2s'], 2),
                'coil_to_cortex_orig': round(d['c2c'], 2),
                'scd': round(d['scd'], 2),
                'csd_mm': d['csd'],
                'ccd_new': round(d['c2c_n'], 2),
                'c2s_new_check': round(d['c2s_n'], 2),
                'scd_new_check': round(d['scd_n'], 2),
                'shift_mm': round(float(np.linalg.norm(
                    d['new_pos'] - d['mat'][:3, 3])), 2),
                'new_coil_x': round(float(d['new_pos'][0]), 1),
                'new_coil_y': round(float(d['new_pos'][1]), 1),
                'new_coil_z': round(float(d['new_pos'][2]), 1),
                'handle_angle_orig_rel_anterior':
                    round(core.handle_angle_rel_anterior(d['mat']), 1),
                'theta_star_deg': th,
                'handle_angle_opt_rel_anterior':
                    round(core.handle_angle_rel_anterior(core.rotate_about_coil_z(
                        d['mat'], th, d['new_pos'])), 1),
                'ef99p9_orig_u': round(float(r0[0]), 1),
                'emax_orig_u': round(float(r0[1]), 1),
                'global_ef99p9_orig_u': round(float(r0[2]), 1),
                'efield_angle_midline_orig_u':
                    round(ang_o, 1) if ang_o is not None else '',
                'forward_orig_u': 'Y' if fwd_o else ('N' if fwd_o is not None else ''),
                'ef99p9_opt_u': round(float(rs[0]), 1),
                'emax_opt_u': round(float(rs[1]), 1),
                'global_ef99p9_opt_u': round(float(rs[2]), 1),
                'efield_angle_midline_opt_u':
                    round(ang_s, 1) if ang_s is not None else '',
                'forward_opt_u': 'Y' if fwd_s else ('N' if fwd_s is not None else ''),
                'delta_ef99p9_opt_minus_orig_u': round(float(rs[0] - r0[0]), 1),
                'artifact_check_v_per_m': art_check,
                'notes': '; '.join(d['notes']),
            })
        status = 'OK'
        del mesh, cond_field
    except Exception as e:
        log(f'========== FAIL {sub}: {e} ==========')
        log(traceback.format_exc())
        rows, status = [], f'FAIL: {e}'
    elapsed = round((time.time() - t_start) / 60, 1)
    log(f'========== {status} {sub} ({elapsed} min) ==========')
    return {'subject': sub, 'status': status, 'elapsed_min': elapsed,
            'rows': rows}


# Windows spawn needs a top-level, picklable worker
def _worker(entry, mesh_base, out_dir, art_root, fn_coil, csd_mm, targets,
            angles, exclusion_notes, log_path):
    log = log_factory(log_path)
    return process_subject(entry, mesh_base, out_dir, art_root, fn_coil,
                           csd_mm, targets, angles, exclusion_notes, log)


def write_long(path, all_rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=core.FIELDS)
        w.writeheader()
        w.writerows(sorted(all_rows, key=lambda r: (r['subject'], r['target'])))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mesh-base', required=True,
                    help='folder containing <ID>/m2m_<ID>/<ID>.msh')
    ap.add_argument('--input-csv', required=True,
                    help='cohort CSV (double header, 2nd row = names)')
    ap.add_argument('--out-dir', required=True, help='sweeps + tables (small)')
    ap.add_argument('--art-dir', default=None,
                    help='theta* artifact meshes (~330 MB x 2 per subject); '
                         'default <out-dir>/artifacts')
    ap.add_argument('--coil', default=os.path.abspath(DEFAULT_COIL))
    ap.add_argument('--targets', default=','.join(core.TARGETS))
    ap.add_argument('--m1-csd', type=float, default=2.0)
    ap.add_argument('--dlpfc-csd', type=float, default=1.0)
    ap.add_argument('--angle-step', type=float, default=5.0)
    ap.add_argument('--cases', default='', help='comma list, e.g. TC001,TC056')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--no-artifacts', action='store_true',
                    help='sweeps only (saves ~0.7 GB/subject disk)')
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(',')]
    csd_mm = {'M1': args.m1_csd, 'DLPFC': args.dlpfc_csd}
    angles = [round(x, 2) for x in np.arange(0, 180, args.angle_step)]
    art_root = args.art_dir or os.path.join(args.out_dir, 'artifacts')
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.no_artifacts:
        os.makedirs(art_root, exist_ok=True)
    log_path = os.path.join(args.out_dir, 'unified_csd_batch.log')
    log = log_factory(log_path)

    entries = core.load_subjects(args.input_csv, targets)
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(',') if c.strip()}
        entries = [e for e in entries if e['subject'] in wanted]
        log(f'*** SUBSET: {sorted(wanted)} ***')

    def have_all(e):
        return all(core.sweep_exists(os.path.join(
                    args.out_dir, e['subject'], f'{t}_sweep.hdf5'))
                for t in targets) and (
            args.no_artifacts or artifact_exists(art_root, e['subject']))
    todo = [e for e in entries if not have_all(e)]
    log(f'{len(todo)} subjects need sweeps/artifacts; '
        f'{len(entries)-len(todo)} reuse existing')

    all_rows, results = [], []
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker, e, args.mesh_base, args.out_dir,
                              art_root if not args.no_artifacts else args.out_dir,
                              args.coil, csd_mm, targets, angles, {},
                              log_path): e['subject'] for e in todo}
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    res = fut.result()
                    all_rows.extend(res['rows'])
                    results.append(res)
                except Exception as e:
                    log(f'[{s}] worker crashed: {e}')
                write_long(os.path.join(args.out_dir, 'unified_csd_long.csv'),
                           all_rows)

    for e in [e for e in entries if e not in todo]:
        try:
            res = _worker(e, args.mesh_base, args.out_dir, art_root, args.coil,
                          csd_mm, targets, angles, {}, log_path)
            all_rows.extend(res['rows'])
        except Exception as ex:
            log(f'[{e["subject"]}] extraction failed: {ex}')
    write_long(os.path.join(args.out_dir, 'unified_csd_long.csv'), all_rows)
    ok = sum(1 for r in results if r['status'] == 'OK')
    log(f'Batch finished: {ok}/{len(results)} computed OK; '
        f'total rows {len(all_rows)}')


if __name__ == '__main__':
    main()
