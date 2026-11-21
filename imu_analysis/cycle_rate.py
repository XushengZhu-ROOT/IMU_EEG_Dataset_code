"""Rate of head shaking, head nodding and picking up in every Move segment: the median of three estimates (events of the
angular velocity or head height, events of the head angle or tilt, and the autocorrelation period), also for the first
and last 60 s.

    python imu_analysis/cycle_rate.py --bids /path/to/dataset --out /path/to/output

Writes cycle_rate_segments.csv (per segment) and cycle_rate_summary.csv (per movement: slow vs fast, and first vs last
60 s with first in slow_median and last in fast_median).
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from imu_analysis import imu_data as dat  # noqa: E402
from imu_analysis import imu_movement_rate as mr  # noqa: E402

FS = mr.FS
MOVEMENTS = (2, 3, 4)
UNIT = {2: 'cycles per minute', 3: 'cycles per minute', 4: 'pick-ups per minute'}
ACF_MIN_LAG = {2: 0.4, 3: 0.4, 4: 1.0}


def armed_events(x, frac=0.35):
    """Times (s) at which x rises past +th after falling below -th (th = frac x 90th percentile of |x|)."""
    th = frac * np.percentile(np.abs(x), 90)
    t, armed = [], False
    for i, v in enumerate(x):
        if v < -th:
            armed = True
        elif v > th and armed:
            t.append(i / FS)
            armed = False
    return np.array(t)


def head_angle(gyro, scale=mr.GYRO_SCALE):
    """Angle (deg) about the main rotation axis of the head."""
    x = mr.band(gyro - gyro.mean(0), 0.05, 3.0)
    vt = np.linalg.svd(x, full_matrices=False)[2]
    return mr.band(np.cumsum(x @ vt[0] / scale) / FS, 0.05, 3.0)


def head_tilt(acc):
    """Slow change of the acceleration along its main direction (g): the tilt of the head relative to gravity."""
    x = mr.band(acc - acc.mean(0), 0.05, 1.5)
    vt = np.linalg.svd(x, full_matrices=False)[2]
    return x @ vt[0]


def tilt_from_upright(acc, rest_acc):
    """Angle (deg) between the head orientation (acceleration low-passed at 1 Hz) and the upright orientation at rest."""
    up = np.nanmean(rest_acc, axis=0)
    up = up / np.linalg.norm(up)
    lp = mr.band(acc, 0, 1.0)
    u = lp / np.linalg.norm(lp, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(u @ up, -1, 1)))


def acf_rate(x, lo_s=0.4, hi_s=30.0):
    """Events per minute from the autocorrelation period, and the height of that peak."""
    x = x - x.mean()
    n = len(x)
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * n)) ** 2)[:n]
    ac = ac / ac[0]
    a, b = int(lo_s * FS), min(int(hi_s * FS), n - 2)
    p, _ = find_peaks(ac[a:b])
    if not len(p):
        return np.nan, np.nan
    p = p + a
    q = p[ac[p] >= 0.6 * ac[p].max()][0]
    den = ac[q - 1] - 2 * ac[q] + ac[q + 1]
    lag = q + (0.5 * (ac[q - 1] - ac[q + 1]) / den if den else 0)
    return 60 * FS / lag, float(ac[q])


def segment_rates(g, Y, rest):
    """Rate estimates of one Move segment g from its IMU Y and the IMU of the Rest segment before it (slots, 6)."""
    mv = int(g.cond[8])
    e1 = mr.one_segment(g, Y)
    Y = mr.bridged(Y)
    dur = len(Y) / FS
    if mv == 4:
        s2 = head_tilt(Y[:, :3])
        s3 = tilt_from_upright(Y[:, :3], mr.bridged(rest)[:, :3])
    else:
        s2 = s3 = head_angle(Y[:, 3:])
    t = armed_events(s2)
    e2 = len(t) / dur * 60
    e3, q3 = acf_rate(s3, ACF_MIN_LAG[mv])
    n60 = int(60 * FS)
    f2, l2 = float(((t >= 0) & (t < 60)).sum()), float(((t >= dur - 60) & (t < dur)).sum())
    f3 = acf_rate(s3[:n60], ACF_MIN_LAG[mv])[0]
    l3 = acf_rate(s3[-n60:], ACF_MIN_LAG[mv])[0]
    return dict(sub=g.sub, movement=mv, task=mr.TASK[mv], speed=g.speed, duration_s=round(dur, 2), unit=UNIT[mv],
                est1_events=e1['main_measure'], est2_count=round(e2, 1), est3_period=round(e3, 1), est3_peak=round(q3, 3),
                rate=round(float(np.median([e1['main_measure'], e2, e3])), 1),
                first_60s=round(float(np.median([e1['first_60s'], f2, f3])), 1),
                last_60s=round(float(np.median([e1['last_60s'], l2, l3])), 1))


def segment_table(bids, tables=dat.TABLES):
    """cycle_rate_segments.csv: one row per Move segment of head shaking, head nodding and picking up."""
    S = dat.load_tables(tables)[0]
    M = dat.move_segments(tables)
    rows = []
    for g in M[M.cond.str[8].astype(int).isin(MOVEMENTS)].itertuples():
        rest = S[(S['sub'] == g.sub) & (S.cond == g.cond) & (S.seg == 'Rest')].iloc[0]
        Y = dat.read_motion(dat.motion_path(bids, g))[0]
        rows.append(segment_rates(g, Y, dat.read_motion(dat.motion_path(bids, rest))[0]))
    return pd.DataFrame(rows)


def summary(D):
    rows = []
    for mv in MOVEMENTS:
        d = D[D.movement == mv]
        s = d[d.speed == 'slow'].set_index('sub')['rate']
        f = d[d.speed == 'fast'].set_index('sub')['rate']
        k = s.index.intersection(f.index)
        a, b = s[k], f[k]
        rows.append(dict(task=mr.TASK[mv], measure='rate (median of three estimates)', unit=UNIT[mv], n_pairs=len(k),
                         slow_median=round(a.median(), 2), slow_q1=round(a.quantile(.25), 2), slow_q3=round(a.quantile(.75), 2),
                         fast_median=round(b.median(), 2), fast_q1=round(b.quantile(.25), 2), fast_q3=round(b.quantile(.75), 2),
                         median_ratio_fast_over_slow=round(float((b / a).median()), 2), fast_is_faster=int((b > a).sum()),
                         wilcoxon_p=float(f'{stats.wilcoxon(b, a).pvalue:.2e}')))
        rows.append(dict(task=mr.TASK[mv], measure='rate, first vs last 60 s (both speeds)', unit=UNIT[mv], n_pairs=len(d),
                         slow_median=d.first_60s.median(), fast_median=d.last_60s.median(),
                         wilcoxon_p=float(f'{stats.wilcoxon(d.last_60s, d.first_60s).pvalue:.3f}')))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids', required=True, help='dataset root')
    ap.add_argument('--out', required=True, help='output folder (new or empty)')
    A = ap.parse_args()
    out = dat.output_folder(A.out)
    D = segment_table(A.bids)
    S = summary(D)
    D.to_csv(out / 'cycle_rate_segments.csv', index=False)
    S.to_csv(out / 'cycle_rate_summary.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print(S.to_string(index=False))
