"""Sample times of the EEG from the LSL arrival times and the CGX packet counter.

LSL stamps the EEG samples when they reach the laptop, in chunks, so the raw time stamps lag the sampling times by a
variable delivery delay. The packet counter (+1 per sample, modulo 128) numbers the samples with the lost ones
included. Within a stretch without long arrival gaps the sampling times are a straight line in this number, and every
arrival lies on or above that line plus the minimum delivery delay, so a trimmed line fit through the earliest arrival
in each 1-s window gives the clock. The minimum delay is a constant that ends up in c (onset_correction.py). Where the
earliest arrival stays above the line for a whole neighbourhood of windows, the samples are flagged as off-clock.
"""
import numpy as np
from scipy.ndimage import grey_opening


def lower_line(x, y, window, keep):
    """Straight line through the per-window minima of y(x), refitted on the lowest `keep` % of them.

    Returns the line (np.polyval coefficients), the window number of each sample, and the window numbers and
    residuals of the window minima.
    """
    w = np.floor((x - x[0]) / window).astype(int)
    if w[-1] < 2:
        return np.array([0.0, y.min()]), w, np.zeros(1, int), np.zeros(1)
    order = np.lexsort((y, w))                          # by window, then by y: the first of each window is its minimum
    first = order[np.r_[0, np.flatnonzero(np.diff(w[order])) + 1]]
    xm, ym = x[first], y[first]
    use = np.ones(len(xm), bool)
    for _ in range(5):
        p = np.polyfit(xm[use], ym[use], 1)
        r = ym - np.polyval(p, xm)
        use = r <= np.percentile(r[use], keep) + 1e-3
    return p, w, w[first], r


def lost_before(counter, mod, known=None):
    """Number of samples lost before each received sample.

    The counter gives it modulo `mod`, so a gap of `mod` or more samples is counted short by a multiple of `mod`.
    `known` maps the 0-based index of the first sample after a gap to the full length of that gap.
    """
    lost = np.r_[0, (np.diff(counter) - 1) % mod]
    for i, n in (known or {}).items():
        if n < lost[i] or (n - lost[i]) % mod:
            raise ValueError(f'gap before sample {i + 1}: {n} lost samples do not match the counter '
                             f'({lost[i]} modulo {mod})')
        lost[i] = n
    return lost


def sample_times(ts, counter, fs, gap, window, keep, mod, off_tol, off_span, known=None):
    """Time (s, LSL clock) of every received sample.

    Returns the times, the segment number of each sample, the number of samples lost before each sample
    (lost_before) and a flag for samples the clock line does not describe.
    """
    lost = lost_before(counter, mod, known)
    seg = np.r_[0, np.cumsum(np.diff(ts) > gap)]
    t = np.empty(len(ts))
    off_clock = np.zeros(len(ts), bool)
    for s in range(seg[-1] + 1):
        i = np.flatnonzero(seg == s)
        k = np.r_[0, np.cumsum(lost[i[1:]] + 1)]        # sample number within the segment, lost samples included
        nominal = k / fs
        p, w, wm, r = lower_line(nominal, ts[i] - nominal, window, keep)
        t[i] = nominal + np.polyval(p, nominal)
        bad = np.zeros(w[-1] + 1, bool)
        bad[wm] = grey_opening(r, size=off_span, mode='nearest') > off_tol   # runs shorter than off_span drop out
        off_clock[i] = bad[w]
    return t, seg, lost, off_clock
