"""Per raw XDF file: the EEG sample of every audio marker, its status, and the rest epochs for the N1.

The mean of Fz and Cz as recorded is band-passed per gap-free segment and read off at the clock's sample times
(eeg_clock.py) by linear interpolation. A trial's status refers to MARKER_EPOCH around its marker: 'outside' (runs
past the recording), 'dropout' (lost samples), 'clock' (off-clock samples) or 'ok'.
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.signal import butter, sosfiltfilt

from tone_onsets import config as C
from tone_onsets import eeg_clock, xdf_reader

GRID = np.arange(round(C.MARKER_EPOCH[0] * C.FS), round(C.MARKER_EPOCH[1] * C.FS) + 1) / C.FS
META = [('file', 'U40'), ('subject', 'U5'), ('setup', 'U12'), ('tone', 'U8'), ('trial', 'i4'), ('rest', '?'),
        ('baseline_file', '?'), ('marker_time', 'f8'), ('marker_sample', 'f8'), ('status', 'U7')]


def table_gaps(tables):
    """EEG gaps listed in the cut tables (segments.csv, column eeg_gaps).

    Returns {xdf file: {0-based index of the first sample after the gap: lost samples}}. The packet counter counts a
    gap of 128 or more samples short by a multiple of 128; these lengths are the full ones.
    """
    gaps = defaultdict(dict)
    with open(os.path.join(tables, 'segments.csv'), newline='') as f:
        for r in csv.DictReader(f):
            for fname, sample, lost in json.loads(r['eeg_gaps'] or '[]'):
                gaps[fname][sample - 1] = lost
    return dict(gaps)


def is_baseline(fname):
    """Sitting / standing baseline recording (*_Sit, *_sit, *_Stand)."""
    return re.search(r'_(sit|stand)', fname, re.I) is not None


def rest_trials(fname, marker_ts, key_ts, keys, actions):
    """Trials at rest. Baseline files: all trials. Other files: trials inside the experimenter's S segments
    (S pressed -> S released; a repeated press while S is held is ignored)."""
    if is_baseline(fname):
        return np.ones(len(marker_ts), bool)
    rest, start = np.zeros(len(marker_ts), bool), None
    for t, k, a in zip(key_ts, keys, actions):
        if k == 'S' and a == 'pressed' and start is None:
            start = t
        elif k == 'S' and a == 'released' and start is not None:
            rest |= (marker_ts >= start) & (marker_ts <= t)
            start = None
    return rest


def bandpass(x, seg):
    """Zero-phase Butterworth band-pass of each gap-free segment; segments shorter than 10 s are left NaN."""
    sos = butter(C.FILTER_ORDER, C.BAND, btype='band', fs=C.FS, output='sos')
    out = np.full_like(x, np.nan)
    for s in np.unique(seg):
        i = seg == s
        if i.sum() >= 10 * C.FS:
            v = x[:, i]
            out[:, i] = sosfiltfilt(sos, v - v.mean(axis=1, keepdims=True), axis=1)
    return out


def process_file(path, gaps=None):
    """Markers of one raw XDF file (META), the rest standard epochs and their rows, and the file's sample numbering
    (see numbering()). `gaps`: table_gaps()."""
    r = xdf_reader.load(path, C.CHANNELS)
    t, seg, lost, off_clock = eeg_clock.sample_times(
        r['eeg_ts'], r['counter'], C.FS, C.SEGMENT_GAP, C.ENVELOPE_WINDOW, C.ENVELOPE_KEEP, C.COUNTER_MOD,
        C.OFF_CLOCK_TOL, C.OFF_CLOCK_SPAN, known=(gaps or {}).get(r['file']))
    after_loss = lost > 0
    fc = bandpass(r['eeg'], seg).mean(0)                # mean of Fz and Cz
    m = r['marker_ts']
    n = len(m)
    subject = r['file'][:5]
    setup = C.SEPARATE_SUBJECTS.get(subject, C.LAPTOPS[r['host']])
    rest = rest_trials(r['file'], m, r['key_ts'], r['key'], r['action'])

    edges = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(t)]
    loss_t, off_t = t[after_loss], t[off_clock]
    status = np.full(n, 'ok', dtype='U7')
    seg_of = np.full(n, -1)
    for j in range(n):
        lo, hi = m[j] + GRID[0], m[j] + GRID[-1]
        s = np.searchsorted(t[edges[:-1]], lo, side='right') - 1
        if s < 0 or hi > t[edges[s + 1] - 1] or np.isnan(fc[edges[s]]):
            status[j] = 'outside'
        elif np.any((loss_t > lo) & (loss_t <= hi)):
            status[j] = 'dropout'
        elif np.any((off_t >= lo) & (off_t <= hi)):
            status[j] = 'clock'
        else:
            seg_of[j] = s

    rows = np.flatnonzero(rest & (r['tone'] == 'standard') & (status == 'ok'))
    ep = np.empty((len(rows), len(GRID)))
    for q, j in enumerate(rows):
        i = slice(edges[seg_of[j]], edges[seg_of[j] + 1])
        ep[q] = np.interp(m[j] + GRID, t[i], fc[i])

    # EEG sample of each marker: 1-based sample number of the EEG stream in the file, fractional
    sample = np.interp(m, t, np.arange(1, len(t) + 1))
    sample[(m < t[0]) | (m > t[-1])] = np.nan
    meta = np.array(list(zip([r['file']] * n, [subject] * n, [setup] * n, r['tone'], r['trial'], rest,
                             [is_baseline(r['file'])] * n, m, sample, status)), dtype=META)
    print(f"{r['file']:36s} {setup:10s} markers {n:4d}  ok {int((status == 'ok').sum()):4d}  "
          f"rest standards {len(rows):4d}  lost samples {int(lost.sum()):4d}", file=sys.stderr, flush=True)
    return dict(meta=meta, ep=ep, rows=rows, numbering=numbering(lost))


def numbering(lost):
    """Sample numbers with and without the lost samples counted (0-based, 1-based) at the first and last sample and on
    both sides of every gap; np.interp between them converts one into the other."""
    g = np.flatnonzero(lost)
    at = np.unique(np.r_[0, g - 1, g, len(lost) - 1])
    return np.cumsum(lost + 1)[at] - 1, at + 1


def onset_sample(x, d, num):
    """Sample number (1-based, fractional) d samples before sample number x, lost samples counted; num: numbering()."""
    kb, ib = num
    k = np.interp(x, ib, kb) - d
    return np.interp(k, kb, ib) + min(k - kb[0], 0)     # before the first sample: one sample per step


def in_gap(x, d, num):
    """True if the place d samples before sample number x (lost samples counted) lies inside an EEG gap, more than one
    sample from every recorded sample; num: numbering()."""
    kb, ib = num
    k = np.interp(x, ib, kb) - d
    for i in np.flatnonzero((np.diff(ib) == 1) & (np.diff(kb) > 1)):
        if kb[i] + 1 < k < kb[i + 1] - 1:
            return True
    return False
