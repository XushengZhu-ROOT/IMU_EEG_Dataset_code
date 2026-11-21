"""Heatmaps of the group-mean IMU-EEG correlation (|r| per IMU axis, rho per sensor) of each movement (40-s windows,
0.3-10 Hz, both speeds).

    python figures/make_fig_group.py --results /path/to/imu_eeg_correlation/output --out /path/to/output

Reads imu_eeg_correlation_group_40s_0.3-10Hz.csv, written by imu_analysis/imu_eeg_correlation.py. Writes fig_group.png,
fig_group.pdf and fig_group_values.csv.
"""
import argparse
import os
import pathlib
import sys

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from imu_analysis import imu_data as dat  # noqa: E402
from imu_analysis.imu_eeg_correlation import COLS, EEG20  # noqa: E402

SOURCE = 'imu_eeg_correlation_group_40s_0.3-10Hz.csv'
NAMES = {0: 'Straight walking', 1: 'Curved walking', 2: 'Head shaking', 3: 'Head nodding', 4: 'Picking up an object',
         5: 'Stair climbing'}
VMAX = 0.6
ABS_R = '∣r∣'


def load(results):
    G = pd.read_csv(os.path.join(results, SOURCE))
    G = G[G.speeds == 'both']
    P = []
    for mv in range(6):
        g = G[G.movement == mv]
        M = g.pivot(index='eeg', columns='imu', values='mean').loc[EEG20, COLS].to_numpy()
        P.append(dict(movement=mv, M=M, n_trials=int(g.n_trials.iloc[0]), n_subjects=int(g.n_subjects.iloc[0])))
    return P


def draw(P, png, pdf):
    fig, axs = plt.subplots(2, 3, figsize=(11, 10), sharex=True, sharey=True, layout='constrained')
    fig.suptitle('IMU–EEG correlation, group mean')
    for p, ax in zip(P, axs.flat):
        mv = p['movement']
        im = ax.imshow(p['M'], vmin=0, vmax=VMAX, aspect='auto')
        ax.axvline(5.5, color='w')
        ax.set_xticks(range(8), COLS, rotation=90)
        ax.set_yticks(range(20), EEG20)
        ax.set_title(f"({'abcdef'[mv]}) {NAMES[mv]}\n{p['n_subjects']} participants, {p['n_trials']} trials")
    fig.colorbar(im, ax=axs, label=f'{ABS_R} or ρ')
    fig.savefig(png)
    fig.savefig(pdf)


def values(P):
    out = []
    for p in P:
        d = pd.DataFrame(p['M'], columns=COLS)
        d.insert(0, 'eeg', EEG20)
        d.insert(0, 'task', NAMES[p['movement']])
        d.insert(0, 'movement', p['movement'])
        d['n_subjects'], d['n_trials'] = p['n_subjects'], p['n_trials']
        out.append(d)
    return pd.concat(out, ignore_index=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True, help='output folder of imu_analysis/imu_eeg_correlation.py')
    ap.add_argument('--out', required=True, help='output folder (new or empty)')
    A = ap.parse_args()
    P = load(A.results)
    assert max(p['M'].max() for p in P) <= VMAX
    out = dat.output_folder(A.out)
    draw(P, out / 'fig_group.png', out / 'fig_group.pdf')
    values(P).to_csv(out / 'fig_group_values.csv', index=False)
    print(f'largest cell {max(p["M"].max() for p in P):.4f} (colour scale 0-{VMAX})')
