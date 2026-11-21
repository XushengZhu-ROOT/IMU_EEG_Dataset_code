"""Raw EEG of every recording of the dataset as tab-separated text (derivatives/eeg_text).

    python bids/make_eeg_text.py --bids /path/to/dataset --out /path/to/dataset/derivatives/eeg_text [--workers 3]

For every sub-XX/eeg/*_eeg.vhdr (BINARY, MULTIPLEXED, IEEE_FLOAT_32, 2000 us, resolution 1, channel names as in
_channels.tsv) one file <out>/sub-XX/eeg/<same name>_eeg.tsv: a header row, latency_s (sample index / 500), then the
channels of the .vhdr in the same order, values with 9 significant digits, which give back the same float32. Every file
is read back and compared with the .eeg value by value; the script stops at the first difference. Also writes
dataset_description.json and README.
"""
import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd

FS = 500
FORMAT = {'DataFormat': 'BINARY', 'DataOrientation': 'MULTIPLEXED', 'BinaryFormat': 'IEEE_FLOAT_32',
          'SamplingInterval': '2000'}
DESCRIPTION = {
    'Name': 'Raw EEG as tab-separated text',
    'BIDSVersion': '1.10.0',
    'DatasetType': 'derivative',
    'GeneratedBy': [{'Name': 'make_eeg_text.py',
                     'Description': 'Every raw EEG recording (sub-XX/eeg/*_eeg.eeg, BrainVision, float32) written as '
                                    'tab-separated text: a header row, latency_s (sample index / 500 Hz), then the '
                                    'channels of the .vhdr in the same order, values as stored with 9 significant '
                                    'digits. Each file was read back and is identical to the .eeg value by value.'}],
    'DatasetLinks': {'raw': '../../'},
}
README = """Raw EEG as tab-separated text
=============================

The raw EEG of every recording of this dataset as plain text, for use without a BrainVision reader. The values are
the same as in the BrainVision files (sub-XX/eeg/*_eeg.eeg): each text file was read back and compared with its .eeg
file value by value.

Files
-----
sub-XX/eeg/<name>_eeg.tsv, one per recording, with the same <name> as the raw files (for example
sub-01/eeg/sub-01_task-walkstraight_acq-slowmove_eeg.tsv belongs to
sub-01/eeg/sub-01_task-walkstraight_acq-slowmove_eeg.vhdr).

Columns
-------
- latency_s: time in seconds from the first sample of the recording (sample index / 500 Hz). This is the time base
  of the onset column of the raw _events.tsv and of the latency column of the IMU _motion.tsv of the same recording.
- then the channels of the BrainVision file in the same order: the EEG channels (uV, reference A1), ACC1-3 (headset
  accelerometer, micro-g) and PacketCounter. Names, types, units and descriptions are in the raw _channels.tsv;
  sampling rate, reference and recording details in the raw _eeg.json.

Values are float32 as stored, written with 9 significant digits, which gives back exactly the same float32 numbers.
The events (tones and boundaries) are in the raw _events.tsv.

Made with make_eeg_text.py (see dataset_description.json).
"""


def read_vhdr(vhdr):
    """Channel names of a BrainVision header; stops unless it has the format above."""
    with open(vhdr, encoding='utf-8') as fh:
        h = fh.read()
    for k, v in FORMAT.items():
        m = re.search(rf'(?m)^{k}=(.*)$', h)
        if not m or m.group(1).strip() != v:
            raise SystemExit(f'{vhdr}: {k} is not {v}')
    ch = re.findall(r'(?m)^Ch\d+=([^,]*),([^,]*),([^,]*),(.*)$', h)
    n = int(re.search(r'(?m)^NumberOfChannels=(\d+)', h).group(1))
    if len(ch) != n or any(float(c[2]) != 1 for c in ch):
        raise SystemExit(f'{vhdr}: {len(ch)} channels for {n}, or a resolution other than 1')
    return [c[0] for c in ch]


def text_copy(vhdr, bids, out):
    """Write the text file of one recording and read it back. -> (samples, bytes)"""
    rel = os.path.relpath(vhdr, bids)
    names = read_vhdr(vhdr)
    if list(pd.read_csv(vhdr[:-len('_eeg.vhdr')] + '_channels.tsv', sep='\t')['name']) != names:
        raise SystemExit(f'{rel}: channel names differ from _channels.tsv')
    raw = np.fromfile(vhdr[:-len('.vhdr')] + '.eeg', dtype='<f4')
    if raw.size % len(names):
        raise SystemExit(f'{rel}: .eeg size is not a multiple of the channel count')
    x = raw.reshape(-1, len(names))
    dst = os.path.join(out, rel[:-len('.vhdr')] + '.tsv')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    lat = np.arange(len(x)) / FS
    with open(dst, 'w', encoding='utf-8', newline='\n') as fh:
        np.savetxt(fh, np.column_stack([lat, x.astype(np.float64)]), fmt=['%.3f'] + ['%.9g'] * len(names),
                   delimiter='\t', header='\t'.join(['latency_s'] + names), comments='')
    back = pd.read_csv(dst, sep='\t', float_precision='round_trip')
    if not (list(back.columns) == ['latency_s'] + names
            and np.array_equal(back[names].to_numpy(np.float64).astype(np.float32), x, equal_nan=True)
            and np.allclose(back['latency_s'].to_numpy(), lat, atol=1e-9, rtol=0)):
        raise SystemExit(f'{rel}: text read back differs from the .eeg')
    return len(x), os.path.getsize(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids', required=True, help='dataset root')
    ap.add_argument('--out', required=True, help='output folder (new or empty), normally <dataset>/derivatives/eeg_text')
    ap.add_argument('--workers', type=int, default=3, help='files written in parallel')
    a = ap.parse_args()
    bids, out = os.path.abspath(a.bids), os.path.abspath(a.out)
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f'{out} is not empty')
    files = sorted(glob.glob(os.path.join(bids, 'sub-*', 'eeg', '*_eeg.vhdr')))
    os.makedirs(out, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(partial(text_copy, bids=bids, out=out), files))
    with open(os.path.join(out, 'dataset_description.json'), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(DESCRIPTION, fh, indent=2)
    with open(os.path.join(out, 'README'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(README)
    print(f'{len(rows)} files, each read back identical to its .eeg; {sum(r[0] for r in rows):,} samples; '
          f'{sum(r[1] for r in rows) / 1e9:.2f} GB in {out}')


if __name__ == '__main__':
    main()
