"""Lag between the head IMU and the EEG clock in every window of every segment, for the IMU as originally released
(before) and the IMU of the dataset (after). Reference: the accelerometer of the EEG headset (CGX), signals as in
sync/resync.prepare() (CGX low-passed at 20 Hz on the IMU sample times; both band-passed at 0.3-10 Hz).

Windows of 5 s, hop 2 s. In each window the lag d with the highest rotation-invariant rho (after: +/-1.5 s; before:
+/-6 s), refined below one sample by a parabola through the three points around the peak; d > 0: the IMU would have to
move later. A window counts when the head moves (CGX >= 0.02 g) and the best rho is >= 0.8. Per segment: median lag,
largest lag, share of windows within +/-1 sample, drift = Theil-Sen slope of the lag against time (ms per minute).

    python imu_analysis/imu_sync_validation.py --bids /path/to/dataset --out /path/to/output [--workers 3]

-> imu_lag_windows.csv (every window), imu_lag_segments.csv (every segment), imu_lag_summary.csv (printed)
"""
import argparse
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sync import resync as rs  # noqa: E402
from imu_analysis import imu_alignment as al  # noqa: E402
from imu_analysis import imu_data as dat  # noqa: E402

FS = rs.FS_IMU
MS = 1000.0 / FS
LIM = {'after': int(round(1.5 * FS)), 'before': int(round(6.0 * FS))}
ARGS = None


def track(acc, imu, lim, win_s=5.0, hop_s=2.0):
    X, Y, miss = rs.prepare(acc, imu[:, :3])
    m = len(X)
    W, H = int(round(win_s * FS)), int(round(hop_s * FS))
    rows = []
    for s in range(0, max(m - W, 0) + 1, H):
        w = np.arange(s, min(s + W, m))
        w = w[~miss[w]]
        if len(w) < W // 2:
            continue
        ds = np.arange(-lim, lim + 1)
        ds = ds[(w.min() - ds >= 0) & (w.max() - ds < m)]
        c = np.array([rs.kabsch_rho(X[w], Y[w - d]) for d in ds])
        i = int(np.argmax(c))
        sub = float(ds[i])
        if 0 < i < len(c) - 1:
            den = c[i - 1] - 2 * c[i] + c[i + 1]
            if den != 0:
                sub += 0.5 * (c[i - 1] - c[i + 1]) / den
        amp = float(np.sqrt((X[w] ** 2).sum(1).mean()))
        r0 = float(c[ds == 0][0]) if (ds == 0).any() else np.nan
        rows.append(dict(t_mid_s=round((s + W / 2) / FS, 2), amp_g=round(amp, 4), rho_lag0=round(r0, 3), best_lag=int(ds[i]),
                         best_lag_refined=round(sub, 3), rho_best=round(float(c[i]), 3), counts=bool(amp >= 0.02 and c[i] >= 0.8)))
    return rows


def summarize(g, when, rows):
    x = pd.DataFrame(rows)
    u = x[x.counts] if len(x) else x
    r = dict(sub=g.sub, cond=g.cond, seg=g.seg, when=when, imu_usable=g.imu_usable, correction=g.correction, windows=len(x),
             windows_counted=len(u))
    if len(u) >= 3:
        r.update(median_lag_ms=round(float(u.best_lag_refined.median() * MS), 2), max_abs_lag_ms=round(float(u.best_lag.abs().max() * MS), 1),
                 share_within_1_sample=round(float((u.best_lag.abs() <= 1).mean()), 4), share_lag_0=round(float((u.best_lag == 0).mean()), 4))
    if len(u) >= 10:
        r['drift_ms_per_min'] = round(float(stats.theilslopes(u.best_lag_refined * MS, u.t_mid_s / 60.0)[0]), 2)
        n = max(3, len(u) // 10)
        r['lag_change_first_last_ms'] = round(float((u.best_lag_refined.iloc[-n:].median() - u.best_lag_refined.iloc[:n].median()) * MS), 2)
    return r


def run_subject(sub):
    W, G = [], []
    for g, acc, before, after in al.segment_data(sub, ARGS.bids):
        for when, Y in (('before', before), ('after', after)):
            rows = track(acc, Y, LIM[when])
            G.append(summarize(g, when, rows))
            W += [dict(sub=g.sub, cond=g.cond, seg=g.seg, when=when, **r) for r in rows]
    print(sub, 'done', flush=True)
    return W, G


def summary(W, G):
    usable = set(map(tuple, G[G.imu_usable != 'no'][['sub', 'cond', 'seg']].values))
    Wu = W[[(a, b, c) in usable for a, b, c in zip(W['sub'], W.cond, W.seg)]]
    rows = []
    for when in ('before', 'after'):
        for seg in ('Move', 'Rest'):
            x = Wu[(Wu.when == when) & (Wu.seg == seg) & Wu.counts]
            g = G[(G.when == when) & (G.seg == seg) & (G.imu_usable != 'no')]
            d = g.drift_ms_per_min.dropna()
            rows.append(dict(when=when, part=seg, segments=len(g), windows_counted=len(x), share_lag_0=round(float((x.best_lag == 0).mean()), 4),
                             share_within_1_sample_20ms=round(float((x.best_lag.abs() <= 1).mean()), 4),
                             median_abs_lag_ms=round(float(x.best_lag_refined.abs().median() * MS), 2),
                             p95_abs_lag_ms=round(float(x.best_lag_refined.abs().quantile(.95) * MS), 1),
                             p99_abs_lag_ms=round(float(x.best_lag_refined.abs().quantile(.99) * MS), 1),
                             max_abs_lag_ms=round(float(x.best_lag.abs().max() * MS), 1), segments_with_drift_estimate=len(d),
                             median_abs_drift_ms_per_min=round(float(d.abs().median()), 2),
                             p95_abs_drift_ms_per_min=round(float(d.abs().quantile(.95)), 2),
                             max_abs_drift_ms_per_min=round(float(d.abs().max()), 2)))
    return pd.DataFrame(rows)


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
    W = pd.DataFrame([r for a, _ in res for r in a])
    G = pd.DataFrame([r for _, b in res for r in b])
    W.to_csv(out / 'imu_lag_windows.csv', index=False)
    G.to_csv(out / 'imu_lag_segments.csv', index=False)
    M = summary(W, G)
    M.to_csv(out / 'imu_lag_summary.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print(M.T.to_string())
