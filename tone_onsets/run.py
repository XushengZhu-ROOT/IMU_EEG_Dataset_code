"""Corrected onset time (LSL marker time - c) of every oddball tone, one c per audio set-up.

    python tone_onsets/run.py --data <dataset>/sourcedata --out <folder> [--tables <folder>] [--jobs 3]

Writes to --out: tone_onsets.csv (one row per tone), onset_correction.csv (c per set-up) and summary.txt.
"""
import argparse
import csv
import functools
import glob
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tone_onsets import extract  # noqa: E402
from tone_onsets import onset_correction as OC  # noqa: E402
from tone_onsets import timestamps  # noqa: E402


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='folder with sub-*/eeg/*.xdf (sourcedata of the dataset)')
    ap.add_argument('--out', required=True, help='output folder; must not exist or be empty')
    ap.add_argument('--tables', default=str(timestamps.TABLES), help='folder with the cut tables (segments.csv)')
    ap.add_argument('--jobs', type=int, default=3, help='files read in parallel (default 3)')
    a = ap.parse_args()
    if os.path.isdir(a.out) and os.listdir(a.out):
        raise SystemExit(f'{a.out} is not empty')
    files = sorted(glob.glob(os.path.join(a.data, 'sub-*', 'eeg', '*.xdf')))
    if not files:
        raise SystemExit(f'no sub-*/eeg/*.xdf under {a.data}')
    os.makedirs(a.out, exist_ok=True)

    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        parts = list(ex.map(functools.partial(extract.process_file, gaps=extract.table_gaps(a.tables)), files))
    meta = np.concatenate([p['meta'] for p in parts])
    offset = np.cumsum([0] + [len(p['meta']) for p in parts])
    rest_std = meta[np.concatenate([p['rows'] + o for p, o in zip(parts, offset)])]   # the trials behind fc
    fc = np.concatenate([p['ep'] for p in parts])

    table = OC.estimate(fc, rest_std)
    write_csv(os.path.join(a.out, 'onset_correction.csv'), list(table[0]), [list(r.values()) for r in table])
    write_csv(os.path.join(a.out, 'tone_onsets.csv'), timestamps.COLUMNS,
              timestamps.rows(meta, {r['setup']: r['c_ms'] for r in table},
                              {os.path.basename(f): p['numbering'] for f, p in zip(files, parts)}))

    status = ', '.join(f'{k} {v}' for k, v in zip(*np.unique(meta['status'], return_counts=True)))
    lines = ['Tone onset correction (corrected time = marker time - c), from the standard tones at rest, mean of Fz '
             'and Cz as recorded (reference A1):']
    for r in table:
        lines.append(f"  {r['setup']:10s} {r['n_subjects']:2d} subjects, {r['n_standard_trials']:5d} trials: N1 at "
                     f"marker {r['n1_vs_marker_ms']:+6.1f} ms -> c = {r['c_ms']:3d} ms "
                     f"[95% CI {r['c_ci_low_ms']}, {r['c_ci_high_ms']}]")
    lines += ['', f"Markers: {len(meta)} in {len(parts)} files from {len(set(meta['subject']))} subjects; rest "
              f"{int(meta['rest'].sum())}. Status (-600..900 ms around the marker): {status}."]
    lines += [''] + OC.baseline_check(fc, rest_std, table)
    with open(os.path.join(a.out, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
