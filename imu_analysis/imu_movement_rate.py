"""Speed and intensity of every Move segment with usable IMU data, from the head IMU of the dataset.

Main measure: straight walking, one-way time across the room (s; median time between U-turns, a U-turn = the heading,
the integral of the angular rate about gravity with the gyroscope / 4, changes by at least 130 deg within 3 s); curved
walking and stairs, steps per minute (autocorrelation of the vertical acceleration at 0.7-4 Hz; for walking not reported
below a peak height of 0.4); head shaking and nodding, cycles per minute (first principal component of the gyroscope at
0.08-3 Hz, one cycle = past -threshold, then past +threshold); picking up, pick-ups per minute (dips of the head height,
the double integral of the vertical acceleration high-passed at 0.15 Hz). Intensity: rms of the three-axis acceleration
band-passed at 0.3-10 Hz (g). The same main measure for the first and the last 60 s of each segment.

    python imu_analysis/imu_movement_rate.py --bids /path/to/dataset --out /path/to/output

-> imu_movement_rate.csv (one row per segment); imu_movement_rate_summary.csv (per task: slow vs fast; first vs last
   60 s, with slow_median and fast_median = first and last)
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, find_peaks, sosfiltfilt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from imu_analysis import imu_data as dat  # noqa: E402

FS = 49.647537
GYRO_SCALE = 4.0
TASK = {0: 'straight walking', 1: 'curved walking', 2: 'head shaking', 3: 'head nodding', 4: 'picking up', 5: 'stairs'}


def band(x, lo, hi, order=4):
    sos = butter(order, [lo, hi], 'bandpass', fs=FS, output='sos') if lo else butter(order, hi, 'low', fs=FS, output='sos')
    return sosfiltfilt(sos, x, axis=0)


def highpass(x, f):
    return sosfiltfilt(butter(2, f, 'high', fs=FS, output='sos'), x, axis=0)


def bridged(Y):
    """Empty samples bridged by straight lines, every column."""
    Y = Y.copy()
    ok = np.flatnonzero(~np.isnan(Y[:, 0]))
    for k in range(6):
        Y[:, k] = np.interp(np.arange(len(Y)), ok, Y[ok, k])
    return Y


def intensity(acc):
    """Rms of the three-axis acceleration band-passed at 0.3-10 Hz (g)."""
    return round(float(np.sqrt((band(acc, 0.3, 10.0) ** 2).sum(1).mean())), 4)


def turn_times(gyro, ghat, scale, half_s=1.5, deg=130):
    """Times (s) of the U-turns of straight walking: the heading (integral of the angular rate about gravity) changes by
    at least `deg` degrees between half_s before and after."""
    th = np.cumsum((gyro @ ghat) / scale) / FS
    h = int(half_s * FS)
    d = np.zeros(len(th))
    d[h:-h] = th[2 * h:] - th[:-2 * h]
    p, _ = find_peaks(np.abs(d), height=deg, distance=int(2.0 * FS))
    return p / FS


def cadence(a_v, lo_s=0.33, hi_s=2.2):
    """Steps per minute and the height of the autocorrelation peak: the first peak that reaches 60 % of the highest,
    its lag refined by a parabola."""
    x = band(a_v, 0.7, 4.0)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(np.abs(f) ** 2)[:n]
    ac = ac / ac[0]
    a, b = int(lo_s * FS), min(int(hi_s * FS), n - 1)
    p, _ = find_peaks(ac[a:b])
    if not len(p):
        return np.nan, np.nan
    p = p + a
    top = ac[p].max()
    q = p[ac[p] >= 0.6 * top][0]
    den = ac[q - 1] - 2 * ac[q] + ac[q + 1]
    lag = q + (0.5 * (ac[q - 1] - ac[q + 1]) / den if den else 0)
    return 60 * FS / lag, float(ac[q])


def cycles(gyro):
    """Times (s) of the cycles of the main rotation (first principal component of the gyroscope, 0.08-3 Hz). One cycle =
    the rotation goes past +threshold after having been past -threshold (threshold: 35 % of the 90th percentile)."""
    x = band(gyro - gyro.mean(0), 0.08, 3.0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    pc = u[:, 0] * s[0]
    th = 0.35 * np.percentile(np.abs(pc), 90)
    t, armed = [], False
    for i, v in enumerate(pc):
        if v < -th:
            armed = True
        elif v > th and armed:
            t.append(i / FS)
            armed = False
    return np.array(t), vt[0]


def pickups(acc, ghat):
    """Times (s) of the head dips and their depth (m): double integral of the vertical acceleration."""
    a = (acc @ ghat - np.linalg.norm(acc.mean(0))) * 9.81
    v = highpass(np.cumsum(highpass(a, 0.15)) / FS, 0.15)
    h = highpass(np.cumsum(v) / FS, 0.15)
    p0, pr = find_peaks(-h, prominence=0.02, distance=int(1.0 * FS))
    if not len(p0):
        return np.array([]), np.array([])
    ref = np.percentile(pr['prominences'], 90)
    keep = pr['prominences'] >= 0.4 * ref
    return p0[keep] / FS, pr['prominences'][keep]


def in_window(t, a, b):
    return ((t >= a) & (t < b)).sum()


def one_segment(g, Y0, scale=GYRO_SCALE):
    """All measures of one Move segment (row g of imu_segments.csv, IMU Y0 (slots, 6))."""
    Y = bridged(Y0)
    acc, gyro = Y[:, :3], Y[:, 3:]
    dur = len(Y) / FS
    mv = int(g.cond[8])
    ghat = acc.mean(0) / np.linalg.norm(acc.mean(0))
    a_v = acc @ ghat
    r = dict(sub=g.sub, cond=g.cond, movement=mv, task=TASK[mv], speed=g.speed, duration_s=round(dur, 2), imu_usable=g.imu_usable,
             intensity_acc_g=intensity(acc), gyro_rms=round(float(np.sqrt((band(gyro, 0.3, 10.0) ** 2).sum(1).mean())), 2))
    cad, q = cadence(a_v)
    first, last = None, None
    if mv in (0, 1, 5):
        r.update(cadence_steps_per_min=round(cad, 1) if q >= 0.4 else np.nan, cadence_quality=round(q, 3))
    if mv == 0:
        t = turn_times(gyro, ghat, scale)
        d = np.diff(t)
        d = d[d > 2.0]
        ok = len(d) >= 3
        r.update(n_turns=len(t), one_way_time_s=round(float(np.median(d)), 2) if ok else np.nan,
                 one_way_iqr_s=round(float(np.subtract(*np.percentile(d, [75, 25]))), 2) if ok else np.nan)
        val, unit = r['one_way_time_s'], 's per one-way crossing'
        if ok:
            f = np.diff(t[t < 60])
            l = np.diff(t[t >= dur - 60])
            first, last = (float(np.median(f)) if len(f) else np.nan), (float(np.median(l)) if len(l) else np.nan)
    elif mv == 1:
        val, unit = r['cadence_steps_per_min'], 'steps per minute'
        n60 = int(60 * FS)
        cf, qf = cadence(a_v[:n60])
        cl, ql = cadence(a_v[-n60:])
        first, last = (cf if qf >= 0.4 else np.nan), (cl if ql >= 0.4 else np.nan)
    elif mv in (2, 3):
        t, axis = cycles(gyro)
        val, unit = round(len(t) / dur * 60, 1), 'cycles per minute'
        r.update(cycles_per_min=val, main_gyro_axis=['GyroX', 'GyroY', 'GyroZ'][int(np.argmax(np.abs(axis)))],
                 main_gyro_axis_share=round(float(axis[np.argmax(np.abs(axis))] ** 2), 3))
        first, last = in_window(t, 0, 60) * 1.0, in_window(t, dur - 60, dur) * 1.0
    elif mv == 4:
        t, depth = pickups(acc, ghat)
        val, unit = round(len(t) / dur * 60, 1), 'pick-ups per minute'
        r.update(pickups_per_min=val, median_head_drop_m=round(float(np.median(depth)), 3) if len(depth) else np.nan)
        first, last = in_window(t, 0, 60) * 1.0, in_window(t, dur - 60, dur) * 1.0
    else:
        val, unit = round(cad, 1), 'steps per minute'
        n60 = int(60 * FS)
        first, last = cadence(a_v[:n60])[0], cadence(a_v[-n60:])[0]
    r.update(main_measure=val, main_unit=unit, first_60s=round(first, 2) if first is not None else np.nan,
             last_60s=round(last, 2) if last is not None else np.nan)
    return r


def summary(T):
    rows = []
    for mv, task in TASK.items():
        x = T[T.movement == mv]
        for c, lab in (('main_measure', x.main_unit.iloc[0]), ('intensity_acc_g', 'g (0.3-10 Hz rms)')) + (
                (('cadence_steps_per_min', 'steps per minute'),) if mv == 0 else ()):
            s = x[x.speed == 'slow'].set_index('sub')[c].astype(float)
            f = x[x.speed == 'fast'].set_index('sub')[c].astype(float)
            k = s.dropna().index.intersection(f.dropna().index)
            a, b = s[k], f[k]
            lower = c == 'main_measure' and mv == 0                   # a shorter one-way time is faster
            p = stats.wilcoxon(b, a).pvalue if len(k) >= 5 else np.nan
            rows.append(dict(task=task, measure=c, unit=lab, n_pairs=len(k),
                             slow_median=round(a.median(), 2), slow_q1=round(a.quantile(.25), 2), slow_q3=round(a.quantile(.75), 2),
                             fast_median=round(b.median(), 2), fast_q1=round(b.quantile(.25), 2), fast_q3=round(b.quantile(.75), 2),
                             median_ratio_fast_over_slow=round(float((a / b if lower else b / a).median()), 2),
                             fast_is_faster=int(((b < a) if lower else (b > a)).sum()), wilcoxon_p=float(f'{p:.2e}') if p == p else np.nan))
        fl = x[['first_60s', 'last_60s']].astype(float).dropna()
        if len(fl) >= 5:
            rows.append(dict(task=task, measure='first 60 s vs last 60 s (both speeds)', unit=x.main_unit.iloc[0], n_pairs=len(fl),
                             slow_median=round(fl.first_60s.median(), 2), fast_median=round(fl.last_60s.median(), 2),
                             wilcoxon_p=float(f'{stats.wilcoxon(fl.last_60s, fl.first_60s).pvalue:.2e}')))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids', required=True, help='dataset root')
    ap.add_argument('--out', required=True, help='output folder (new or empty)')
    A = ap.parse_args()
    out = dat.output_folder(A.out)
    T = pd.DataFrame([one_segment(g, dat.read_motion(dat.motion_path(A.bids, g))[0]) for g in dat.move_segments().itertuples()])
    T.to_csv(out / 'imu_movement_rate.csv', index=False)
    M = summary(T)
    M.to_csv(out / 'imu_movement_rate_summary.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print(M.to_string(index=False))
    print('segments without the main measure:')
    print(T.groupby('task').main_measure.apply(lambda v: int(v.isna().sum())).to_string())
