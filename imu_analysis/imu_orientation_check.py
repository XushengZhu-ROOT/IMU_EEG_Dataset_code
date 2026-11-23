"""IMU axes on the head (X right, Y anterior, Z superior) and tilt from gravity, gyroscope scale and clipping, and the
axes of the EEG headset accelerometer (CGX), estimated from the data.

    python imu_analysis/imu_orientation_check.py --bids /path/to/dataset --out /path/to/output [--workers 3]

Writes imu_orientation_segments.csv (per segment), imu_gyro_scale.csv (from head nodding and U-turns) and
imu_orientation_summary.csv (printed).
"""
import argparse
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sync import resync as rs  # noqa: E402
from imu_analysis import imu_alignment as al  # noqa: E402
from imu_analysis import imu_data as dat  # noqa: E402
from imu_analysis import imu_movement_rate as mr  # noqa: E402

FS = rs.FS_IMU
CLIP = 990.0
ARGS = None


def nod_scale(Y):
    th_acc = np.degrees(np.arctan2(Y[:, 1], Y[:, 2]))
    th_g = np.cumsum(Y[:, 3]) / FS
    a, b = mr.band(th_acc, 0.08, 3.0), mr.band(th_g, 0.08, 3.0)
    return float((a @ b) / (a @ a)), float(np.corrcoef(a, b)[0, 1])


def turn_scale(Y):
    acc, gyro = Y[:, :3], Y[:, 3:]
    gh = acc.mean(0) / np.linalg.norm(acc.mean(0))
    yaw = gyro @ gh
    out = []
    h = int(1.5 * FS)
    for t in mr.turn_times(gyro, gh, mr.GYRO_SCALE):
        i = int(t * FS)
        if i < h or i + h >= len(yaw):
            continue
        if (np.abs(gyro[i - h:i + h]) >= CLIP).any():
            continue                                               # a clipped turn adds up to too little
        out.append(abs(np.sum(yaw[i - h:i + h]) / FS) / 180.0)
    return out


def run_subject(sub):
    rows, scale = [], []
    for g, acc, _, after in al.segment_data(sub, ARGS.bids):
        ok = ~np.isnan(after[:, 0])
        G = after[ok, :3].mean(0)
        n = np.linalg.norm(G)
        C = acc.mean(0)
        r = dict(sub=g.sub, cond=g.cond, seg=g.seg, imu_usable=g.imu_usable, imu_g_x=G[0], imu_g_y=G[1], imu_g_z=G[2], imu_g_norm=n,
                 imu_tilt_deg=float(np.degrees(np.arccos(G[2] / n))), cgx_g_x=C[0], cgx_g_y=C[1], cgx_g_z=C[2],
                 gyro_clipped_samples=int((np.abs(after[ok, 3:]) >= CLIP).any(1).sum()), slots_with_data=int(ok.sum()),
                 gyro_x_rms=float(after[ok, 3].std()), gyro_y_rms=float(after[ok, 4].std()), gyro_z_rms=float(after[ok, 5].std()))
        if g.seg == 'Move' and g.imu_usable != 'no':
            A = rs.alignment(acc, after[:, :3])
            r['rho'] = A['rho']
            if A['rho'] >= 0.9:
                R = al.rotation(A['X'][~A['miss']], A['Y'][~A['miss']])
                r.update({f'R{i + 1}{j + 1}': R[i, j] for i in range(3) for j in range(3)})
            Yb = mr.bridged(after)
            if g.cond.startswith('movement3'):
                k, c = nod_scale(Yb)
                scale.append(dict(sub=g.sub, cond=g.cond, speed=g.speed, method='nodding: gyro integral / tilt from gravity', ratio=k, fit_r=c,
                                  clipped=r['gyro_clipped_samples']))
            if g.cond.startswith('movement0'):
                t = turn_scale(Yb)
                if t:
                    scale.append(dict(sub=g.sub, cond=g.cond, speed=g.speed, method='U-turn: gyro integral / 180 deg', ratio=float(np.median(t)),
                                      fit_r=np.nan, clipped=r['gyro_clipped_samples'], n_turns=len(t)))
            if g.cond.startswith('movement4'):
                y = Yb[:, 1]
                r.update(pickup_accY_p1=float(np.percentile(y, 1)), pickup_accY_p50=float(np.median(y)), pickup_accY_p99=float(np.percentile(y, 99)))
        rows.append(r)
    print(sub, 'done', flush=True)
    return rows, scale


def summary(T, K):
    S = rs.load_tables()['segments'][['sub', 'cond', 'seg', 'eeg_channels']]
    T = T.merge(S, on=['sub', 'cond', 'seg'], how='left')
    res = T[(T.seg == 'Rest') & (T.imu_usable != 'no')]
    mv = T[(T.seg == 'Move') & T.R11.notna() & (T.eeg_channels == '1-20')]    # the 30-channel headset has other CGX axes
    Rm = np.array([[mv[f'R{i + 1}{j + 1}'].mean() for j in range(3)] for i in range(3)])
    U, _, Vt = np.linalg.svd(Rm)
    Rp = U @ Vt
    R = [np.array([[r[f'R{i + 1}{j + 1}'] for j in range(3)] for i in range(3)]) for _, r in mv.iterrows()]
    ang = [float(np.degrees(np.arccos(np.clip((np.trace(Rp.T @ x) - 1) / 2, -1, 1)))) for x in R]
    pu = T[T.pickup_accY_p1.notna()]
    nods = K[K.method.str.startswith('nodding') & (K.speed == 'slow') & (K.fit_r >= 0.95) & (K.clipped == 0)]
    turns = K[K.method.str.startswith('U-turn')]
    tilt = res.imu_tilt_deg

    def iqr(x, d):
        return f'{x.median():.{d}f} [{x.quantile(.25):.{d}f}, {x.quantile(.75):.{d}f}]'

    def xyz(p, f='mean'):
        return ', '.join(f"{a} {getattr(res[p + a], f)():.3f}" for a in 'xyz')

    G = res[['imu_g_x', 'imu_g_y', 'imu_g_z']]
    rows = [('gravity at Rest, median over segments (g)', xyz('imu_g_', 'median'), len(res)),
            ('segments whose gravity lies closest to +Z',
             int(((G.abs().idxmax(axis=1) == 'imu_g_z') & (G.imu_g_z > 0)).sum()), len(res)),
            ('tilt of the IMU from gravity at Rest (deg): median [IQR], max', f'{iqr(tilt, 1)}, {tilt.max():.1f}', len(res)),
            ('picking up: accY p1 / median / p99 (g), median over segments',
             ' / '.join(f'{pu[c].median():.2f}' for c in ('pickup_accY_p1', 'pickup_accY_p50', 'pickup_accY_p99')), len(pu)),
            ('gyro scale from slow nods (file value / true deg): median [IQR]', iqr(nods.ratio, 2), len(nods)),
            ('gyro scale from U-turns of straight walking: median [IQR]', iqr(turns.ratio, 2), len(turns)),
            ('segments with clipped gyroscope samples (|value| >= 990)', int((T.gyro_clipped_samples > 0).sum()), len(T)),
            ('clipped samples / samples with data (all segments)',
             f'{int(T.gyro_clipped_samples.sum())} / {int(T.slots_with_data.sum())}', len(T)),
            ('CGX gravity at Rest, mean over segments (g)', xyz('cgx_g_'), len(res)),
            ('IMU-to-CGX rotation, mean over Move segments of the 20-channel headset (rows: IMU X, Y, Z; columns: CGX axes)',
             np.array2string(Rp, precision=2, suppress_small=True).replace('\n', ' '), len(mv)),
            ('spread of that rotation over segments (deg): median, p95', f'{np.median(ang):.1f}, {np.percentile(ang, 95):.1f}', len(mv))]
    return pd.DataFrame(rows, columns=['item', 'value', 'n'])


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
    T = pd.DataFrame([r for a, _ in res for r in a])
    K = pd.DataFrame([r for _, b in res for r in b])
    T.round(4).to_csv(out / 'imu_orientation_segments.csv', index=False)
    K.round(4).to_csv(out / 'imu_gyro_scale.csv', index=False)
    S = summary(T, K)
    S.to_csv(out / 'imu_orientation_summary.csv', index=False)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_colwidth', 120)
    print(S.to_string(index=False))
