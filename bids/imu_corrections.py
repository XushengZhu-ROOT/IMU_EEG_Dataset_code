"""IMU of one segment from the raw IMU rows, placed so that IMU slot u lies (u - 1) / 49.647537 s after the segment's
first EEG sample. One function per value of the column `correction` of imu_segments.csv; the parameters come from
imu_segments.csv, imu_pieces.csv and imu_time_base_runs.csv.

  none             shift_offset(L = 0)        the rows of the original segmentation
  offset           shift_offset()             every piece moved by the same number of rows L
  drift            correct_drift()            every pause-delimited piece moved by its own shift
  lost samples     correct_lost_samples()     every raw row on its own slot; lost samples interpolated, longer gaps n/a
  eeg packet loss  correct_eeg_packet_loss()  the IMU rows of the time the EEG lost are left out
  fast imu clock   correct_fast_imu_clock()   rows dropped evenly
For none and offset, extend_to_eeg_end() lets the IMU reach the end of its EEG.
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
    """Constant offset: every piece takes the rows released_first_row - L onwards (as many slots as released).
    L > 0 moves the IMU later (right), L < 0 earlier (left); rows past the end of the file are NaN."""
    return np.vstack([rows_to_values(raw, p.txt, np.arange(p.released_first_row, p.released_last_row + 1) - L)
                      for p in pieces.itertuples()])


def extend_to_eeg_end(Y, raw, last_piece, L, n_add):
    """The IMU ended before its EEG: append n_add slots, the rows that follow the last row of the last piece, NaN where
    the file has no more rows."""
    if n_add <= 0:
        return Y
    nxt = last_piece.released_last_row - L + 1 + np.arange(n_add)
    return np.vstack([Y, rows_to_values(raw, last_piece.txt, nxt)])


def interpolate_lost_samples(Y, lost):
    """Slots of samples the IMU lost: straight line between the raw slots on either side (each column)."""
    e = np.diff(np.r_[0, lost.astype(int), 0])
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        assert a > 0 and b < len(Y) and not np.isnan(Y[a - 1, 0]) and not np.isnan(Y[b, 0]), 'lost samples need raw neighbours'
        for c in range(Y.shape[1]):
            Y[a:b, c] = np.interp(np.arange(a, b), [a - 1, b], [Y[a - 1, c], Y[b, c]])
    return Y


def place_runs(raw, runs, n_slots):
    """Build the segment run by run (imu_time_base_runs.csv). A `rows` run puts consecutive rows on consecutive slots,
    `lost sample (interpolated)` runs are filled by interpolate_lost_samples(), `no data (n/a)` runs stay empty. Between
    two runs of rows the file may skip rows (lost EEG time, fast IMU clock) or the slots may skip ahead (IMU sample loss)."""
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
    """Drift between pieces: the IMU clock ran slightly off, which built up across the pauses between the pieces. Every
    pause-delimited piece is moved as a block by its own shift (one run of rows per piece)."""
    assert (runs.kind == 'rows').all()
    return place_runs(raw, runs, n_slots)


def correct_lost_samples(raw, runs, n_slots):
    """IMU sample loss (Bluetooth): every raw row sits on its own slot, found by dynamic programming against the headset
    accelerometer, so the shift grows by one at every lost sample. Lost samples are interpolated; gaps longer than 0.5 s
    stay NaN."""
    return place_runs(raw, runs, n_slots)


def correct_eeg_packet_loss(raw, runs, n_slots):
    """The EEG amplifier lost samples (packet counter): the IMU rows recorded during the lost EEG time are left out, so
    the rows skip ahead between runs."""
    return place_runs(raw, runs, n_slots)


def correct_fast_imu_clock(raw, runs, n_slots):
    """The IMU delivered more rows per second than 49.647537: the rows between two anchors (the sync nod before, the
    first nod after) are put on the line through them, so rows are dropped evenly."""
    return place_runs(raw, runs, n_slots)


CORRECT = {'drift': correct_drift, 'lost samples': correct_lost_samples, 'eeg packet loss': correct_eeg_packet_loss,
           'fast imu clock': correct_fast_imu_clock}


def process_segment(g, pieces, runs, raw):
    """IMU of the segment (slots, 6). g = row of imu_segments.csv, pieces and runs = its rows of imu_pieces.csv (tail
    pieces left out) and imu_time_base_runs.csv, raw = {IMU file name: (rows, 6)}."""
    if g.correction in CORRECT:
        Y = CORRECT[g.correction](raw, runs, int(g.imu_slots))
    else:
        L = int(g.offset_samples) if g.correction == 'offset' else 0
        Y = extend_to_eeg_end(shift_offset(raw, pieces, L), raw, pieces.iloc[-1], L, int(g.end_slots_added))
    assert len(Y) == int(g.imu_slots), (g.sub, g.cond, g.seg, len(Y), g.imu_slots)
    return Y
