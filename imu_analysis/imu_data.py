"""Helpers for the IMU scripts: read the IMU tables, the EEG and IMU of a segment, and the IMU as originally released.
"""
import os
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sync import resync as rs  # noqa: E402

TABLES = pathlib.Path(__file__).resolve().parents[1] / 'tables'
SUBJECTS = [f'Sub{i:02d}' for i in range(1, 21)]


def load_tables(tables=TABLES):
    """imu_segments.csv (one row per segment) and imu_pieces.csv (the IMU rows of every piece)."""
    S = pd.read_csv(os.path.join(tables, 'imu_segments.csv'), keep_default_na=False)
    P = pd.read_csv(os.path.join(tables, 'imu_pieces.csv'), keep_default_na=False)
    return S, P


def move_segments(tables=TABLES):
    """Rows of imu_segments.csv: the Move segments with usable IMU data (imu_usable = yes or partly)."""
    S = load_tables(tables)[0]
    return S[(S.seg == 'Move') & (S.imu_slots.astype(int) > 0) & (S.imu_usable != 'no')]


def output_folder(path):
    """Create the output folder; stop if it already holds files."""
    p = pathlib.Path(path)
    if p.exists() and any(p.iterdir()):
        sys.exit(f'{p} is not empty')
    p.mkdir(parents=True, exist_ok=True)
    return p


def motion_path(bids, g):
    return os.path.join(bids, g.bids_file)


def eeg_path(bids, g):
    d, f = os.path.split(g.bids_file)
    return os.path.join(bids, os.path.dirname(d), 'eeg', f.replace('_tracksys-imu', '').replace('_motion.tsv', '_eeg.vhdr'))


def read_motion(path):
    """_motion.tsv -> IMU (slots, 6): accX accY accZ (g), gyroX gyroY gyroZ; n/a = NaN. Also the latency column (s)."""
    D = pd.read_csv(path, sep='\t', header=None, na_values='n/a', float_precision='round_trip')
    return D.iloc[:, :6].to_numpy(float), D.iloc[:, 6].to_numpy(float)


def read_brainvision(vhdr):
    """Channel names, sampling rate (Hz) and data (samples x channels, float32, in the units of the header)."""
    info, names, res, section = {}, [], [], ''
    with open(vhdr, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('['):
                section = line
            elif '=' in line and not line.startswith(';'):
                key, val = line.split('=', 1)
                if section == '[Channel Infos]':
                    parts = val.split(',')
                    names.append(parts[0])
                    res.append(float(parts[2]) if len(parts) > 2 and parts[2] else 1.0)
                else:
                    info[key] = val
    assert info['DataFormat'] == 'BINARY' and info['DataOrientation'] == 'MULTIPLEXED', vhdr
    assert info['BinaryFormat'] == 'IEEE_FLOAT_32' and int(info['NumberOfChannels']) == len(names), vhdr
    data = np.fromfile(os.path.join(os.path.dirname(vhdr), info['DataFile']), dtype='<f4').reshape(-1, len(names))
    if any(r != 1.0 for r in res):
        data = data * np.asarray(res, dtype=np.float32)
    return names, 1e6 / float(info['SamplingInterval']), data


def read_eeg(bids, g):
    """EEG file of a segment: data (samples x channels, float32) and channel names (EEG channels, ACC1-3, PacketCounter)."""
    names, _, data = read_brainvision(eeg_path(bids, g))
    return data, names


def cgx_acc(data, names):
    """Accelerometer of the EEG headset (n, 3) in g: channels ACC1-ACC3 (micro-g)."""
    return data[:, [names.index(c) for c in ('ACC1', 'ACC2', 'ACC3')]].astype(float) / 1e6


def read_raw_imu(bids, sub, S):
    """{txt name: (rows, 6)} of the raw IMU files used by the segments S of one subject."""
    names = {t for x in S.imu_txt for t in x.split('|') if t}
    return {t: rs.read_txt(rs.source_file(bids, sub, t)) for t in sorted(names)}


def rows_to_values(raw, txt, rows):
    """Values (n, 6) of 1-based txt rows; rows outside the txt give NaN."""
    rows = np.asarray(rows)
    out = np.full((len(rows), 6), np.nan)
    ok = (rows >= 1) & (rows <= len(raw[txt]))
    out[ok] = raw[txt][rows[ok] - 1, :6]
    return out


def released_imu(raw, P, g):
    """IMU of a segment as originally released: the released rows of its pieces, without tail pieces."""
    pcs = P[(P['sub'] == g.sub) & (P.cond == g.cond) & (P.seg == g.seg) & (P['tail_piece'] != 'yes')]
    return np.vstack([rows_to_values(raw, p.txt, np.arange(p.released_first_row, p.released_last_row + 1))
                      for p in pcs.itertuples()])
