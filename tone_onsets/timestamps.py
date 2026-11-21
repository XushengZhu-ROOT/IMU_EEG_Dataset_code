"""Corrected time and EEG sample of every audio marker of one raw XDF file.

    python tone_onsets/timestamps.py <file.xdf> [--c onset_correction.csv] [--tables folder]

Prints the file's rows of tone_onsets.csv. corrected_eeg_sample (1-based, EEG stream of the XDF file) is empty for
markers outside the recording or onsets inside an EEG gap.
"""
import argparse
import csv
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tone_onsets import config as C  # noqa: E402
from tone_onsets import extract  # noqa: E402

TABLES = pathlib.Path(__file__).resolve().parents[1] / 'tables'
COLUMNS = ['file', 'trial', 'label', 'setup', 'marker_time', 'correction_ms', 'corrected_time',
           'corrected_eeg_sample', 'status']


def read_c(path=TABLES / 'onset_correction.csv'):
    """c in ms per audio set-up."""
    with open(path, newline='') as f:
        return {r['setup']: int(r['c_ms']) for r in csv.DictReader(f)}


def rows(meta, c_ms, numbering):
    """Rows of tone_onsets.csv for the markers in meta (extract.META); numbering: {file: extract.numbering()}."""
    for m in meta:
        c = c_ms[m['setup']] / 1e3
        num = numbering[m['file']]
        have = not np.isnan(m['marker_sample']) and not extract.in_gap(m['marker_sample'], c * C.FS, num)
        onset = extract.onset_sample(m['marker_sample'], c * C.FS, num) if have else None
        yield [m['file'], m['trial'], m['tone'], m['setup'], f"{m['marker_time']:.6f}", c_ms[m['setup']],
               f"{m['marker_time'] - c:.6f}", int(round(onset)) if have else '', m['status']]


def correct_file(path, c_ms=None, tables=TABLES):
    """Table rows of one raw XDF file."""
    part = extract.process_file(path, extract.table_gaps(tables))
    return list(rows(part['meta'], c_ms or read_c(), {os.path.basename(path): part['numbering']}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('xdf', help='raw XDF file (sourcedata/sub-XX/eeg/*.xdf of the dataset)')
    ap.add_argument('--c', default=str(TABLES / 'onset_correction.csv'), help='c per audio set-up')
    ap.add_argument('--tables', default=str(TABLES), help='folder with the cut tables (segments.csv)')
    a = ap.parse_args()
    w = csv.writer(sys.stdout, lineterminator='\n')
    w.writerow(COLUMNS)
    w.writerows(correct_file(a.xdf, read_c(a.c), a.tables))


if __name__ == '__main__':
    main()
