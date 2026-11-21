"""Rebuild every segment from the raw recordings with the cut tables and check it against the values in the tables.

    python sync/verify_resync.py --bids /path/to/dataset --out verify_out [--tables tables] [--workers 3]

  1 positions  EEG: number of samples, first and last xdf sample. IMU: number of slots and of empty slots, first and
               last txt row with data (segments.csv)
  2 values     check_first and check_last of every EEG and IMU piece (pieces.csv); IMU: check_imu_sum (segments.csv)
  3 alignment  segments of 40 s or more: whole-segment rho within 0.02 of `rho` (segments.csv) and best lag 0 (+/-1);
               shorter segments are reported only
-> <out>/verify_segments.csv, one row per segment
"""
import os, sys, argparse, pathlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sync import resync as rs

ARGS = T = None


def result(ok, failed=()):
    return 'ok' if ok else 'FAIL: ' + ', '.join(failed) if failed else 'FAIL'


def check(key, cache):
    C = rs.cut(T, *key)
    eeg, imu = rs.cut_from_raw(C, ARGS.bids, cache)
    g = C['info']
    P = T['pieces']
    pcs = P[(P['sub'] == key[0]) & (P.cond == key[1]) & (P.seg == key[2]) & ~P.note.str.startswith('tail')]
    out = dict(sub=key[0], cond=key[1], seg=key[2], bids_name=g.bids_name, imu_usable=g.imu_usable)
    ok = (len(eeg) == g.eeg_samples and f"{C['eeg'][0][0]}:{C['eeg'][0][1]}" == g.eeg_start
          and f"{C['eeg'][-1][0]}:{C['eeg'][-1][2]}" == g.eeg_end)
    bad = [f'piece {p.piece}' for p in pcs[pcs.stream == 'EEG'].itertuples()
           if not np.allclose([eeg[p.seg_first - 1, 0], eeg[p.seg_last - 1, 0]],
                              [float(p.check_first), float(p.check_last)], rtol=1e-7, atol=1e-3)]
    out.update(eeg_positions=result(ok), eeg_values=result(not bad, bad))
    if imu is None:
        return {**out, 'imu_positions': 'no IMU', 'imu_values': 'no IMU', 'alignment': 'no IMU'}
    m = C['imu']
    raw = m[m.status == 'raw']
    ok = (len(imu) == g.imu_slots and int(np.isnan(imu[:, 0]).sum()) == g.imu_empty
          and f'{raw.file.iloc[0]}:{raw.row.iloc[0]}' == g.imu_start and f'{raw.file.iloc[-1]}:{raw.row.iloc[-1]}' == g.imu_end)
    bad = []
    for p in pcs[pcs.stream == 'IMU'].itertuples():
        i0 = np.flatnonzero((m.file == p.file).to_numpy() & (m.row == p.first).to_numpy())
        i1 = np.flatnonzero((m.file == p.file).to_numpy() & (m.row == p.last).to_numpy())
        want = np.array(p.check_first.split() + p.check_last.split(), float)
        if not (len(i0) and len(i1) and np.allclose(np.r_[imu[i0[0], :3], imu[i1[0], :3]], want, rtol=0, atol=1e-9)):
            bad.append(f'piece {p.piece}')
    if not np.isclose(np.nansum(imu[:, :3]), float(g.check_imu_sum), rtol=1e-6, atol=1e-6):
        bad.append('check_imu_sum')
    out.update(imu_positions=result(ok), imu_values=result(not bad, bad))
    A = rs.alignment(rs.cgx_acc(eeg, C), imu[:, :3])
    if g.imu_usable == 'no':
        c3 = 'not usable (imu_usable = no)'
    elif g.eeg_samples < 40 * rs.FS_EEG:
        c3 = 'shorter than 40 s: reported only'
    else:
        c3 = result(abs(A['rho'] - float(g.rho)) <= 0.02 and abs(A['best_lag']) <= 1)
    return {**out, 'alignment': c3, 'rho': A['rho'], 'rho_table': g.rho, 'best_lag': A['best_lag']}


def run_recording(keys):
    """The segments of one recording; its raw files are loaded once."""
    cache = {}
    return [check(k, cache) for k in keys]


def init(args):
    global ARGS, T
    ARGS, T = args, rs.load_tables(args.tables)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Rebuild every segment from the raw recordings and check it.')
    ap.add_argument('--bids', required=True, help='root folder of the dataset')
    ap.add_argument('--tables', default=rs.TABLES, help='folder of the cut tables (default: tables/ of this repository)')
    ap.add_argument('--out', required=True, help='output folder; must not exist or be empty')
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()
    if os.path.isdir(args.out) and os.listdir(args.out):
        sys.exit(f'{args.out} is not empty')
    os.makedirs(args.out, exist_ok=True)
    S = rs.load_tables(args.tables)['segments']
    keys = list(zip(S['sub'], S.cond, S.seg))
    by_recording = {}
    for k, xdf in zip(keys, S.eeg_xdf):
        by_recording.setdefault(xdf, []).append(k)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init, initargs=(args,)) as ex:
        R = pd.DataFrame([r for rows in ex.map(run_recording, by_recording.values()) for r in rows])
    R = R.set_index(['sub', 'cond', 'seg']).loc[keys].reset_index()
    R.to_csv(os.path.join(args.out, 'verify_segments.csv'), index=False)
    has_imu = R.imu_positions != 'no IMU'
    eeg_ok = int(((R.eeg_positions == 'ok') & (R.eeg_values == 'ok')).sum())
    imu_ok = int(((R.imu_positions == 'ok') & (R.imu_values == 'ok')).sum())
    print(f'EEG: {eeg_ok}/{len(R)} segments pass checks 1 and 2')
    print(f'IMU: {imu_ok}/{int(has_imu.sum())} segments pass checks 1 and 2 '
          f'({int((~has_imu).sum())} segments have no IMU accelerometer)')
    print('check 3:', R.alignment.str.split(':').str[0].value_counts().to_dict())
