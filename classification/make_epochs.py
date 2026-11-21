"""Cut the cleaned movement segments into 1-s epochs for CBraMod.

Input: one .mat file (EEGLAB structure EEG) per segment, e.g. sub-01_task-walkstraight_acq-slowmove.mat.
Output: <out>/<epoch_id>.pickle with signal, label and epoch_id.

    python classification/make_epochs.py --segments /path/to/clean_segments --out /path/to/epochs
"""
import argparse
import os
import pickle
import re

import mne
import numpy as np
import scipy.io as sio

LABELS = {'walkstraight': 0, 'walkcurved': 1, 'headshake': 2, 'headnod': 3, 'pickup': 4, 'stairs': 5}
SEGMENT = re.compile(r'(sub-\d+)_task-([a-z]+)_acq-(slow|fast)move(?=[_.])')
SFREQ = 200
EPOCH_S = 1.0


def load_mat_to_raw(mat_path):
    EEG = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)['EEG']
    ch_names = [str(ch.labels) for ch in EEG.chanlocs]
    info = mne.create_info(ch_names=ch_names, sfreq=float(EEG.srate), ch_types='eeg')
    return mne.io.RawArray(np.array(EEG.data), info, verbose=False)


def segment_epochs(mat_path):
    """Band-pass 0.3-75 Hz, resample to 200 Hz, cut into non-overlapping 1-s epochs."""
    raw = load_mat_to_raw(mat_path)
    raw.filter(0.3, 75.0, verbose=False)
    if raw.info['sfreq'] != SFREQ:
        raw.resample(SFREQ, npad='auto', verbose=False)
    n = int(EPOCH_S * raw.info['sfreq'])
    signals = raw.get_data()
    return [signals[:, start:start + n] for start in range(0, signals.shape[1] - n + 1, n)]


def find_segments(folder):
    segments = {}
    for name in sorted(os.listdir(folder)):
        m = SEGMENT.match(name)
        if not name.endswith('.mat') or m is None:
            continue
        sub, task, speed = m.groups()
        if task not in LABELS:
            raise ValueError(f'{name}: unknown task {task}; expected one of {sorted(LABELS)}')
        segment = f'{sub}_task-{task}_acq-{speed}move'
        if segment in segments:
            raise ValueError(f'two files for {segment}: {segments[segment][0]} and {name}')
        segments[segment] = (name, LABELS[task])
    if not segments:
        raise SystemExit(f'no movement segment (sub-XX_task-<task>_acq-<slow|fast>move*.mat) in {folder}')
    return segments


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--segments', required=True,
                    help='folder with the cleaned movement segments, sub-XX_task-<task>_acq-<slow|fast>move*.mat')
    ap.add_argument('--out', required=True, help='new or empty folder for the epoch pickles')
    args = ap.parse_args()
    if os.path.exists(args.out) and (not os.path.isdir(args.out) or os.listdir(args.out)):
        raise SystemExit(f'{args.out} exists and is not an empty folder')
    segments = find_segments(args.segments)
    os.makedirs(args.out, exist_ok=True)

    total = 0
    for segment, (name, label) in segments.items():
        epochs = segment_epochs(os.path.join(args.segments, name))
        for i, signal in enumerate(epochs, start=1):
            epoch_id = f'{segment}_epoch{i:03d}'
            with open(os.path.join(args.out, f'{epoch_id}.pickle'), 'wb') as f:
                pickle.dump({'signal': signal, 'label': label, 'epoch_id': epoch_id}, f)
        print(f'{segment}: {len(epochs)} epochs')
        total += len(epochs)
    print(f'{total} epochs from {len(segments)} segments written to {args.out}')


if __name__ == '__main__':
    main()
