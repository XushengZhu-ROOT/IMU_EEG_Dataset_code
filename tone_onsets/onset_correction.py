"""One value of c = (LSL marker time) - (sound onset) per audio set-up.

audio.py pushes the LSL marker only after sd.wait() returns, so the marker follows the sound by a delay set by the
laptop's audio output. c places the grand-average N1 to standard tones at rest (mean of Fz and Cz) N1_LATENCY = 100 ms
after the corrected onset: c = N1_LATENCY - (N1 peak time relative to the marker). Subject averages (trimmed means)
come first, then the mean over subjects; the 95% CI resamples subjects (trials for a set-up with one subject).
"""
import numpy as np
from scipy.stats import trim_mean

from tone_onsets import config as C
from tone_onsets.extract import GRID


def n1_peak(erp, lo, hi, t=GRID):
    """Time and value of the most negative point of erp (sampled on t) in [lo, hi], refined by a parabola."""
    i = np.flatnonzero((t >= lo) & (t <= hi))
    j = i[np.argmin(erp[i])]
    a, b, c = erp[j - 1], erp[j], erp[j + 1]
    d = a - 2 * b + c
    return t[j] + (0.5 * (a - c) / d / C.FS if d else 0.0), b


def estimate(fc, meta):
    """c per audio set-up from the rest standard epochs fc (one row per row of meta); one table row per set-up."""
    rng = np.random.default_rng(C.SEED)
    rows = []
    for setup in C.SETUPS:
        g = meta['setup'] == setup
        subs = sorted(set(meta['subject'][g]))
        if not subs:
            continue
        S = np.array([trim_mean(fc[g & (meta['subject'] == s)], C.TRIM, axis=0) for s in subs])
        lat, amp = n1_peak(S.mean(0), *C.N1_SEARCH)
        near = (lat - C.PEAK_CI_HALF, lat + C.PEAK_CI_HALF)
        if len(subs) > 1:                               # resample subjects
            boot = [n1_peak(S[rng.integers(0, len(S), len(S))].mean(0), *near)[0] for _ in range(C.BOOTSTRAP)]
        else:                                           # a single subject (Sub02): resample its trials
            X = fc[g]
            boot = [n1_peak(trim_mean(X[rng.integers(0, len(X), len(X))], C.TRIM, axis=0), *near)[0]
                    for _ in range(C.BOOTSTRAP)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        c = round((C.N1_LATENCY - lat) * C.FS) / C.FS   # whole samples
        rows.append(dict(setup=setup, subjects=' '.join(subs), n_subjects=len(subs), n_standard_trials=int(g.sum()),
                         n1_vs_marker_ms=round(1e3 * lat, 1), n1_uV=round(float(amp), 2), c_ms=round(1e3 * c),
                         c_ci_low_ms=round(1e3 * (C.N1_LATENCY - hi)), c_ci_high_ms=round(1e3 * (C.N1_LATENCY - lo))))
    return rows


def baseline_check(fc, meta, table):
    """c again, from the trials of the sitting / standing baseline files only (no key press decides rest there)."""
    rows = estimate(fc[meta['baseline_file']], meta[meta['baseline_file']])
    c_all = {r['setup']: r['c_ms'] for r in table}
    return ['c from the sitting / standing baseline files only:'] + \
        [f"  {r['setup']:10s} {r['n_subjects']:2d} subjects, {r['n_standard_trials']:5d} trials: c = {r['c_ms']:3d} ms "
         f"[95% CI {r['c_ci_low_ms']}, {r['c_ci_high_ms']}] (all rest trials: {c_all[r['setup']]} ms)"
         for r in rows]
