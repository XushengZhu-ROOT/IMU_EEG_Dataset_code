"""IMU of one segment from the raw IMU rows, with slot u at (u - 1) / 49.647537 s after the segment's first EEG sample.

One function per value of `correction` in tables/imu_segments.csv; process_segment() picks it. The parameters come
from imu_segments.csv, imu_pieces.csv and imu_time_base_runs.csv.
"""
import numpy as np


def rows_to_values(raw, txt, rows):
    """Values (n, 6) of 1-based rows of an IMU file; rows outside the file give NaN."""
    rows = np.asarray(rows)
    out = np.full((len(rows), 6), np.nan)
    ok = (rows >= 1) & (rows <= len(raw[txt]))
    out[ok] = raw[txt][rows[ok] - 1, :6]
    return out


def shift_offset(raw, pieces, L):
    """Constant offset: every piece moved by L rows (L > 0 moves the IMU later)."""
    return np.vstack([rows_to_values(raw, p.txt, np.arange(p.released_first_row, p.released_last_row + 1) - L)
                      for p in pieces.itertuples()])


def extend_to_eeg_end(Y, raw, last_piece, L, n_add):
    """Append the n_add rows that follow the last piece (NaN past the end of the file)."""
    if n_add <= 0:
        return Y
    nxt = last_piece.released_last_row - L + 1 + np.arange(n_add)
    return np.vstack([Y, rows_to_values(raw, last_piece.txt, nxt)])


def interpolate_lost_samples(Y, lost):
    """Fill the slots of lost IMU samples by linear interpolation between their neighbours."""
    e = np.diff(np.r_[0, lost.astype(int), 0])
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        assert a > 0 and b < len(Y) and not np.isnan(Y[a - 1, 0]) and not np.isnan(Y[b, 0]), 'lost samples need raw neighbours'
        for c in range(Y.shape[1]):
            Y[a:b, c] = np.interp(np.arange(a, b), [a - 1, b], [Y[a - 1, c], Y[b, c]])
    return Y


def place_runs(raw, runs, n_slots):
    """Place each run of imu_time_base_runs.csv on its slots; lost samples interpolated, no-data runs NaN."""
    Y = np.full((n_slots, 6), np.nan)
    lost = np.zeros(n_slots, bool)
    for r in runs.itertuples():
        a, b = int(r.slot_first) - 1, int(r.slot_last)
        if r.kind == 'rows':
            Y[a:b] = rows_to_values(raw, r.txt, np.arange(int(r.row_first), int(r.row_last) + 1))
        elif r.kind.startswith('lost'):
            lost[a:b] = True
    return interpolate_lost_samples(Y, lost)


def correct_drift(raw, runs, n_slots):
    """Drift: every pause-delimited piece moved by its own shift."""
    assert (runs.kind == 'rows').all()
    return place_runs(raw, runs, n_slots)


def correct_lost_samples(raw, runs, n_slots):
    """Lost IMU samples: every raw row on its own slot, the lost samples interpolated."""
    return place_runs(raw, runs, n_slots)


def correct_eeg_packet_loss(raw, runs, n_slots):
    """EEG packet loss: the IMU rows of the lost EEG time are left out."""
    return place_runs(raw, runs, n_slots)


def correct_fast_imu_clock(raw, runs, n_slots):
    """IMU clock faster than 49.647537 Hz: rows dropped evenly."""
    return place_runs(raw, runs, n_slots)


CORRECT = {'drift': correct_drift, 'lost samples': correct_lost_samples, 'eeg packet loss': correct_eeg_packet_loss,
           'fast imu clock': correct_fast_imu_clock}


def process_segment(g, pieces, runs, raw):
    """IMU (slots, 6) of one segment. g: its row of imu_segments.csv; pieces, runs: its rows of imu_pieces.csv
    (tail pieces left out) and imu_time_base_runs.csv; raw: {IMU file name: (rows, 6)}."""
    if g.correction in CORRECT:
        Y = CORRECT[g.correction](raw, runs, int(g.imu_slots))
    else:
        L = int(g.offset_samples) if g.correction == 'offset' else 0
        Y = extend_to_eeg_end(shift_offset(raw, pieces, L), raw, pieces.iloc[-1], L, int(g.end_slots_added))
    assert len(Y) == int(g.imu_slots), (g.sub, g.cond, g.seg, len(Y), g.imu_slots)
    return Y
