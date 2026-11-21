"""Agreement between the head IMU and the accelerometer of the EEG headset (CGX, on the EEG clock) in every segment, for
the IMU as originally released (before) and the IMU of the dataset (after). The measure is sync/resync.alignment():
whole-segment rho and best lag (+/-0.5 s), and 5-s windows (hop 1 s) below 0.5 with their class (quiet, offset,
no match).

    python imu_analysis/imu_alignment.py --bids /path/to/dataset --out /path/to/output [--workers 3]

-> imu_alignment_segments.csv   one row per segment: rho, best lag, low windows, before and after
   imu_low_windows.csv          one row per window below 0.5, before and after
"""
import argparse
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sync import resync as rs  # noqa: E402
from imu_analysis import imu_data as dat  # noqa: E402

LOW = 0.5
ARGS = None


def rotation(X, Y):
    """The proper rotation R with Y @ R ~ X (the IMU turned into the CGX axes)."""
    U, _, Vt = np.linalg.svd((Y - Y.mean(0)).T @ (X - X.mean(0)))
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def segment_data(sub, bids, tables=dat.TABLES):
    """Per segment of the subject with IMU data: the segment row, the CGX accelerometer (n, 3) in g, the IMU as
    originally released and the IMU of the dataset (m, 6)."""
    S, P = dat.load_tables(tables)
    S = S[(S['sub'] == sub) & (S.imu_slots.astype(int) > 0)]
    raw = dat.read_raw_imu(bids, sub, S)
    for g in S.itertuples():
        data, names = dat.read_eeg(bids, g)
        before = dat.released_imu(raw, P, g)
        after, _ = dat.read_motion(dat.motion_path(bids, g))
        yield g, dat.cgx_acc(data, names), before, after


def run_subject(sub):
    seg_rows, low_rows = [], []
    for g, acc, before, after in segment_data(sub, ARGS.bids):
        r = dict(sub=g.sub, cond=g.cond, seg=g.seg, correction=g.correction, imu_usable=g.imu_usable)
        for tag, Y in (('before', before), ('after', after)):
            A = rs.alignment(acc, Y[:, :3], low=LOW)
            W = A['windows']
            L = W[W.rho < LOW]
            r.update({f'rho_{tag}': A['rho'], f'best_lag_{tag}': A['best_lag'], f'windows_{tag}': len(W),
                      f'low_{tag}': len(L), f'low_quiet_{tag}': int((L.cls == 'quiet').sum()),
                      f'low_offset_{tag}': int((L.cls == 'offset').sum()), f'low_no_match_{tag}': int((L.cls == 'no match').sum()),
                      f'min_window_rho_{tag}': float(W.rho.min())})
            low_rows += L.assign(sub=g.sub, cond=g.cond, seg=g.seg, when=tag).to_dict('records')
        seg_rows.append(r)
    print(sub, 'done', flush=True)
    return seg_rows, low_rows


def init(a):
    global ARGS
    ARGS = a


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids', required=True, help='dataset root')
    ap.add_argument('--out', required=True, help='output folder (new or empty)')
    ap.add_argument('--workers', type=int, default=3)
    ARGS = ap.parse_args()
    out = dat.output_folder(ARGS.out)
    with ProcessPoolExecutor(max_workers=ARGS.workers, initializer=init, initargs=(ARGS,)) as ex:
        res = list(ex.map(run_subject, dat.SUBJECTS))
    S = pd.DataFrame([r for a, _ in res for r in a])
    L = pd.DataFrame([r for _, b in res for r in b])
    S.to_csv(out / 'imu_alignment_segments.csv', index=False)
    cols = ['sub', 'cond', 'seg', 'when', 't_start_s', 't_end_s', 'rho', 'amp_g', 'best_lag', 'rho_best', 'cls']
    L[cols].to_csv(out / 'imu_low_windows.csv', index=False)
    for tag in ('before', 'after'):
        f = S[S[f'low_{tag}'] > 0]
        print(f'{tag}: segments with a 5-s window below {LOW}: {len(f)} (Move {int((f.seg == "Move").sum())}, '
              f'Rest {int((f.seg == "Rest").sum())}); with movement (offset / no match): '
              f'{int(((f[f"low_offset_{tag}"] + f[f"low_no_match_{tag}"]) > 0).sum())}; '
              f'median rho Move {S[S.seg == "Move"][f"rho_{tag}"].median():.3f}, Rest {S[S.seg == "Rest"][f"rho_{tag}"].median():.3f}')
