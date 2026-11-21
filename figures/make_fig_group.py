"""Figure of the group-mean IMU-EEG correlation of each movement (40-s windows, 0.3-10 Hz, both speeds).

    python figures/make_fig_group.py --results /path/to/imu_eeg_correlation/output --out /path/to/output

Reads imu_eeg_correlation_group_40s_0.3-10Hz.csv (imu_analysis/imu_eeg_correlation.py, speeds = both). Per movement:
rows = the 20 EEG channels, columns = |r| with each IMU axis and rho with the three accelerometer (Acc*) and the three
gyroscope axes (Gyro*), each the mean over all Move trials of both speeds; one colour scale (0-0.6) for all panels.

-> fig_group.png (300 dpi), fig_group.pdf, fig_group_values.csv
"""
import argparse
import os
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures import figstyle  # noqa: E402
from imu_analysis import imu_data as dat  # noqa: E402
from imu_analysis.imu_eeg_correlation import COLS, EEG20  # noqa: E402

SOURCE = 'imu_eeg_correlation_group_40s_0.3-10Hz.csv'
NAMES = {0: 'Straight walking', 1: 'Curved walking', 2: 'Head shaking', 3: 'Head nodding', 4: 'Picking up an object',
         5: 'Stair climbing'}
VMAX = 0.6
ABS_R = '∣r∣'                                    # |r| with a bar that cannot be read as 'l'


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
    W = 18.0
    x0, pw, gap, hh = 0.85, 4.95, 0.32, 5.8           # heatmaps: left edge, width, gap, height (20 rows of 0.29 cm)
    y1 = 1.55
    y2 = y1 + hh + 1.9
    H = y2 + hh + 1.12
    pg = figstyle.Page(W, H)
    pg.title('IMU–EEG correlation, group mean')
    im = None
    for p in P:
        mv = p['movement']
        r_, c_ = divmod(mv, 3)
        ax = pg.axes(x0 + c_ * (pw + gap), y1 if r_ == 0 else y2, pw, hh)
        im = ax.pcolormesh(np.arange(9) - 0.5, np.arange(21) - 0.5, p['M'], cmap='viridis', vmin=0, vmax=VMAX,
                           shading='flat', edgecolors='none', antialiased=False, rasterized=False)
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(19.5, -0.5)
        ax.axvline(5.5, color='white', lw=1.2)
        ax.set_xticks(range(8))
        ax.set_xticklabels(COLS, rotation=90)
        ax.set_yticks(range(20))
        ax.set_yticklabels(EEG20 if c_ == 0 else [])
        if c_:
            ax.tick_params(axis='y', length=0)
        ax.set_title(f"({'abcdef'[mv]}) {NAMES[mv]}\n{p['n_subjects']} participants, {p['n_trials']} trials", linespacing=1.15)
    cax = pg.axes(x0 + 3 * pw + 2 * gap + 0.3, y1, 0.22, y2 + hh - y1)
    cb = pg.fig.colorbar(im, cax=cax, ticks=np.arange(0, VMAX + 0.01, 0.1))
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, length=2.5, direction='out', colors='k')
    cb.set_label(f'{ABS_R} or ρ', color='k', labelpad=4)
    figstyle.save(pg.fig, png, pdf)
    return H


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
    with figstyle.paper_style():
        H = draw(P, out / 'fig_group.png', out / 'fig_group.pdf')
    values(P).to_csv(out / 'fig_group_values.csv', index=False)
    print(f'figure 18 x {H:.1f} cm; largest cell {max(p["M"].max() for p in P):.4f} (colour scale 0-{VMAX})')
