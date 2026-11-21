"""Correlation of every EEG channel with the head IMU in the Move segments with usable IMU data: |r| with each IMU axis,
multiple correlation with the accelerometer (Acc*) and the gyroscope (Gyro*), and chance levels (IMU shifted by half the
segment).

    python imu_analysis/imu_eeg_correlation.py --bids /path/to/dataset --out /path/to/output [--workers 3]
    python imu_analysis/imu_eeg_correlation.py --bids /path/to/dataset --out /path/to/output --grid

Writes, for <setting> 40s_0.3-10Hz and 10s_0.3-10Hz (window length and band):
    imu_eeg_correlation_trials_<setting>.csv       per segment
    imu_eeg_correlation_group_<setting>.csv        per movement, speed and channel
    imu_eeg_correlation_axis_speed_<setting>.csv   per movement and speed, mean over channels
    imu_eeg_correlation_speed_test_<setting>.csv   fast vs slow, paired t-test
With --grid, only imu_eeg_correlation_grid.csv (one row per window length and band).
"""
import argparse
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sync import resync as rs  # noqa: E402
from imu_analysis import imu_data as dat  # noqa: E402

FS = rs.FS_IMU
BAND = (0.3, 10.0)
SETTINGS = [(40.0, BAND), (10.0, BAND)]
GRID = [(w, b) for w in (5.0, 10.0, 20.0, 30.0, 40.0) for b in ((0.3, 10.0), (0.5, 10.0), (1.0, 10.0))]
MOVEMENTS = list(range(6))
SHORT = {0: 'Straight walking', 1: 'Curved walking', 2: 'Head shaking', 3: 'Head nodding', 4: 'Picking up', 5: 'Stair climbing'}
EEG20 = ['F7', 'Fp1', 'Fp2', 'F8', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'P8', 'P7', 'Pz', 'P4', 'T3', 'P3', 'O1', 'O2', 'C4', 'T4', 'A2']
EEG_NAME_ALIASES = {'T3': ['T3', 'T7'], 'T4': ['T4', 'T8']}                  # names in the 30-channel headset
COLS = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'Acc*', 'Gyro*']
ARGS = None


def setting_name(s):
    return f'{s[0]:g}s_{s[1][0]:g}-{s[1][1]:g}Hz'


def resolve_eeg_index(names, target):
    for name in EEG_NAME_ALIASES.get(target, [target]):
        if name in names:
            return names.index(name)
    raise ValueError(f'no channel {target} in {names}')


def multiple_corr(x, Y):
    """For every column of x (n, k): the correlation with the best linear combination of the columns of Y (n, 3)."""
    x = x - x.mean(0)
    Y = Y - Y.mean(0)
    Q, _ = np.linalg.qr(Y)
    return np.sqrt(((Q.T @ x) ** 2).sum(0) / (x ** 2).sum(0))


def measures(E, Y):
    """(20, 8): |r| of every EEG channel with each IMU axis, then rho with the accelerometer and with the gyroscope."""
    Ec = E - E.mean(0)
    Yc = Y - Y.mean(0)
    r = np.abs((Yc.T @ Ec) / np.sqrt((Yc ** 2).sum(0)[:, None] * (Ec ** 2).sum(0)[None, :]))
    return np.column_stack([r.T, multiple_corr(E, Y[:, :3]), multiple_corr(E, Y[:, 3:])])


def windows(E, Y, bad, win_s):
    """Mean of measures() over the windows of win_s seconds (hop 1 s) without a bad sample; None if there is none."""
    W, H = int(round(win_s * FS)), int(round(FS))
    out = [measures(E[s:s + W], Y[s:s + W]) for s in range(0, len(E) - W + 1, H) if not bad[s:s + W].any()]
    return np.mean(out, axis=0) if out else None


def eeg_on_imu_grid(eeg, m):
    """EEG channels (n, k) low-passed at 20 Hz and interpolated at the times of IMU samples 1..m within the EEG."""
    c = rs._filter(np.asarray(eeg, float), rs.SOS_LP)
    t_c = np.arange(len(c)) / rs.FS_EEG
    t_i = np.arange(m) / FS
    t_i = t_i[t_i <= t_c[-1]]
    return np.column_stack([np.interp(t_i, t_c, c[:, k]) for k in range(c.shape[1])])


def bridge(Y, m):
    """IMU (slots, 6) cut or padded to m samples, empty samples bridged by straight lines; and the empty samples."""
    Y = np.vstack([Y, np.full((max(0, m - len(Y)), 6), np.nan)])[:m]
    miss = np.isnan(Y[:, 0])
    ok = np.flatnonzero(~miss)
    return np.column_stack([np.interp(np.arange(m), ok, Y[ok, k]) for k in range(6)]), miss


def segments(bids, sub, tables=dat.TABLES):
    """Per usable Move segment: row, EEG at the IMU sample times (m, 20), bridged IMU (m, 6) and its empty samples."""
    S = dat.move_segments(tables)
    for g in S[S['sub'] == sub].itertuples():
        data, names = dat.read_eeg(bids, g)
        imu, _ = dat.read_motion(dat.motion_path(bids, g))
        Ei = eeg_on_imu_grid(data[:, [resolve_eeg_index(names, c) for c in EEG20]], len(imu))
        Yb, miss = bridge(imu, len(Ei))
        yield g, Ei, Yb, miss


def first_seconds(Ei, Yb, miss, seconds=10.0):
    """EEG, IMU and whether no IMU sample is empty, in the first seconds of a segment band-passed at 0.3-10 Hz."""
    n = int(round(seconds * FS))
    sos = butter(4, BAND, 'bandpass', fs=FS, output='sos')
    return rs._filter(Ei, sos)[:n], rs._filter(Yb, sos)[:n], not miss[:n].any()


def run_subject(sub):
    out = []
    for g, Ei, Yb, miss in segments(ARGS.bids, sub):
        m = len(Ei)
        row = dict(sub=sub, movement=int(g.cond[8]), speed=int(g.cond[-1]))
        for s in ARGS.settings:
            sos = butter(4, s[1], 'bandpass', fs=FS, output='sos')
            E, Yf = rs._filter(Ei, sos), rs._filter(Yb, sos)
            Ys = np.roll(Yf, m // 2, axis=0)
            row[setting_name(s)] = (windows(E, Yf, miss, s[0]), windows(E, Ys, miss | np.roll(miss, m // 2), s[0]))
        out.append(row)
    print(sub, 'done', flush=True)
    return out


def valid(T, k):
    return T[[v[0] is not None for v in T[k]]]


def grid_table(T):
    rng = np.random.default_rng(0)
    subs = sorted(T['sub'].unique())
    splits = [rng.permutation(subs) for _ in range(200)]
    rows = []
    for s in GRID:
        k = setting_name(s)
        x = valid(T, k)

        def above(d):
            return d[0][:, 6] - d[1][:, 6]

        def mat(xx):
            return np.array([np.mean([above(d) for d in xx[xx.movement == mv][k]], axis=0) for mv in MOVEMENTS])

        rel = np.mean([np.corrcoef(mat(x[x['sub'].isin(p[:10])]).ravel(), mat(x[x['sub'].isin(p[10:])]).ravel())[0, 1] for p in splits])
        sig = 0
        for mv in MOVEMENTS:
            v = {sp: pd.Series([above(d).mean() for d in x[(x.movement == mv) & (x.speed == sp)][k]],
                               index=x[(x.movement == mv) & (x.speed == sp)]['sub']) for sp in (0, 1)}
            kk = v[0].index.intersection(v[1].index)
            t, p = stats.ttest_rel(v[1][kk], v[0][kk])
            sig += (t > 0) and (p < 0.05)
        A = np.mean([d[0] for d in x[k]], axis=0)
        C = np.mean([d[1] for d in x[k]], axis=0)
        rows.append(dict(setting=k, window_s=s[0], band_hz=f'{s[1][0]:g}-{s[1][1]:g}', acc_rho=round(A[:, 6].mean(), 3),
                         acc_chance=round(C[:, 6].mean(), 3), acc_above_chance=round((A - C)[:, 6].mean(), 3),
                         best_single_axis_r=round(A[:, :6].mean(0).max(), 3), split_half_reliability=round(rel, 3),
                         fast_gt_slow_movements=int(sig)))
    return pd.DataFrame(rows)


def trial_table(T, s):
    k = setting_name(s)
    rows = []
    for _, t in valid(T, k).iterrows():
        A, C = t[k]
        r = dict(sub=t['sub'], movement=t.movement, speed=t.speed)
        for j, e in enumerate(EEG20):
            r[f'acc_star_{e}'], r[f'gyro_star_{e}'], r[f'chance_acc_star_{e}'], r[f'chance_gyro_star_{e}'] = A[j, 6], A[j, 7], C[j, 6], C[j, 7]
        r.update({f'{c}_mean': A[:, i].mean() for i, c in enumerate(COLS)})
        rows.append(r)
    return pd.DataFrame(rows)


def final_tables(T, s):
    """Group, axis-speed and speed-test tables of one setting."""
    k = setting_name(s)
    x = valid(T, k)
    rows, axis, test = [], [], []
    for mv in MOVEMENTS:
        y = x[x.movement == mv]
        for tag, sp in (('both', None), ('slow', 0), ('fast', 1)):
            z = y if sp is None else y[y.speed == sp]
            A = np.stack([d[0] for d in z[k]])
            C = np.stack([d[1] for d in z[k]])
            for i, c in enumerate(COLS):
                for j, e in enumerate(EEG20):
                    rows.append(dict(movement=mv, task=SHORT[mv], speeds=tag, imu=c, eeg=e, mean=round(A[:, j, i].mean(), 4),
                                     sd=round(A[:, j, i].std(ddof=1), 4), chance=round(C[:, j, i].mean(), 4),
                                     n_trials=len(z), n_subjects=z['sub'].nunique()))
            if sp is not None:
                m = A.mean(axis=(0, 1))
                ch = C.mean(axis=(0, 1))
                b = int(np.argmax(m[:6]))
                axis.append(dict(task=SHORT[mv], speed=tag, best_single_axis=COLS[b], mean_r_best_axis=round(m[b], 3), rho_acc_star=round(m[6], 3),
                                 rho_gyro_star=round(m[7], 3), chance_acc_star=round(ch[6], 3), chance_gyro_star=round(ch[7], 3),
                                 n_subjects=z['sub'].nunique(), **{f'r_{c}': round(m[i], 3) for i, c in enumerate(COLS[:6])}))
        for i, c in ((6, 'Acc*'), (7, 'Gyro*')):
            for what, f in (('rho above chance', lambda d: d[0][:, i].mean() - d[1][:, i].mean()), ('rho', lambda d: d[0][:, i].mean())):
                v = {sp: pd.Series([f(d) for d in y[y.speed == sp][k]], index=y[y.speed == sp]['sub']) for sp in (0, 1)}
                kk = v[0].index.intersection(v[1].index)
                t_, p = stats.ttest_rel(v[1][kk], v[0][kk])
                dz = (v[1][kk] - v[0][kk]).mean() / (v[1][kk] - v[0][kk]).std(ddof=1)
                test.append(dict(task=SHORT[mv], imu=c, measure=what, slow=round(v[0][kk].mean(), 4), fast=round(v[1][kk].mean(), 4), t=round(t_, 2),
                                 p=float(f'{p:.2e}'), cohens_d=round(dz, 2), n_pairs=len(kk)))
    return pd.DataFrame(rows), pd.DataFrame(axis), pd.DataFrame(test)


def init(a):
    global ARGS
    ARGS = a


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids', required=True, help='dataset root')
    ap.add_argument('--out', required=True, help='output folder (new or empty)')
    ap.add_argument('--grid', action='store_true', help='every window length and band: imu_eeg_correlation_grid.csv')
    ap.add_argument('--workers', type=int, default=3)
    ARGS = ap.parse_args()
    ARGS.settings = GRID if ARGS.grid else SETTINGS
    out = dat.output_folder(ARGS.out)
    with ProcessPoolExecutor(max_workers=ARGS.workers, initializer=init, initargs=(ARGS,)) as ex:
        T = pd.DataFrame([r for rows in ex.map(run_subject, dat.SUBJECTS) for r in rows])
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 30)
    if ARGS.grid:
        G = grid_table(T).sort_values('acc_above_chance', ascending=False)
        G.to_csv(out / 'imu_eeg_correlation_grid.csv', index=False)
        print(G.to_string(index=False))
    else:
        for s in SETTINGS:
            k = setting_name(s)
            trial_table(T, s).to_csv(out / f'imu_eeg_correlation_trials_{k}.csv', index=False)
            group, axis, test = final_tables(T, s)
            group.to_csv(out / f'imu_eeg_correlation_group_{k}.csv', index=False)
            axis.to_csv(out / f'imu_eeg_correlation_axis_speed_{k}.csv', index=False)
            test.to_csv(out / f'imu_eeg_correlation_speed_test_{k}.csv', index=False)
            print(k)
            print(axis.drop(columns=[c for c in axis if c.startswith('r_')]).to_string(index=False))
            print(test[test.measure == 'rho above chance'].to_string(index=False))
