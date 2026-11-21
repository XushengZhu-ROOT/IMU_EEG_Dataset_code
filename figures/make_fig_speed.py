"""Slow versus fast condition of each movement, from the head IMU; also prints the table of movement intensity and rate.

    python figures/make_fig_speed.py --bids /path/to/dataset --out /path/to/output

Intensity (rms of the 0.3-10 Hz acceleration): straight walking, curved walking, stair climbing. Rate: head shaking,
head nodding, picking up an object. Table: medians, quartiles, fast/slow ratio, Wilcoxon p.

Writes fig_speed.png, fig_speed.pdf and fig_speed_values.csv.
"""
import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from imu_analysis import cycle_rate as cr  # noqa: E402
from imu_analysis import imu_data as dat  # noqa: E402
from imu_analysis import imu_movement_rate as mr  # noqa: E402

NAMES = {0: 'Straight walking', 1: 'Curved walking', 2: 'Head shaking', 3: 'Head nodding', 4: 'Picking up an object',
         5: 'Stair climbing'}
COLUMN = {0: 'intensity_acc_g', 1: 'intensity_acc_g', 2: 'rate', 3: 'rate', 4: 'rate', 5: 'intensity_acc_g'}
YLABEL = {0: 'Intensity (g)', 1: 'Intensity (g)', 2: 'Rate (cycles/min)', 3: 'Rate (cycles/min)',
          4: 'Rate (pick-ups/min)', 5: 'Intensity (g)'}
INTENSITY = [mv for mv in range(6) if COLUMN[mv] == 'intensity_acc_g']


def load(bids):
    """Slow and fast value per movement and participant with a value in both blocks."""
    S = dat.move_segments()
    S = S[S.cond.str[8].astype(int).isin(INTENSITY)]
    T = pd.DataFrame([dict(sub=g.sub, movement=int(g.cond[8]), speed=g.speed,
                           intensity_acc_g=mr.intensity(mr.bridged(dat.read_motion(dat.motion_path(bids, g))[0])[:, :3]))
                      for g in S.itertuples()])
    R = cr.segment_table(bids)
    rows = []
    for mv in range(6):
        x = T[T.movement == mv] if mv in INTENSITY else R[R.movement == mv]
        s = x[x.speed == 'slow'].set_index('sub')[COLUMN[mv]].astype(float)
        f = x[x.speed == 'fast'].set_index('sub')[COLUMN[mv]].astype(float)
        for sub in s.dropna().index.intersection(f.dropna().index):
            rows.append(dict(movement=mv, task=NAMES[mv], measure=COLUMN[mv], unit=YLABEL[mv], sub=sub, slow=s[sub], fast=f[sub]))
    return pd.DataFrame(rows)


def rate_table(V):
    rows = []
    for mv in range(6):
        v = V[V.movement == mv]
        a, b = v.slow, v.fast
        rows.append(dict(movement=NAMES[mv], measure=YLABEL[mv], n_pairs=len(v),
                         slow_median=round(a.median(), 2), slow_q1=round(a.quantile(.25), 2), slow_q3=round(a.quantile(.75), 2),
                         fast_median=round(b.median(), 2), fast_q1=round(b.quantile(.25), 2), fast_q3=round(b.quantile(.75), 2),
                         median_ratio_fast_over_slow=round(float((b / a).median()), 2), fast_is_faster=int((b > a).sum()),
                         wilcoxon_p=float(f'{stats.wilcoxon(b, a).pvalue:.2e}')))
    return pd.DataFrame(rows)


def draw(V, png, pdf):
    fig, axs = plt.subplots(2, 3, figsize=(11, 7), layout='constrained')
    fig.suptitle('Slow and fast conditions')
    for mv, ax in enumerate(axs.flat):
        v = V[V.movement == mv]
        lo, hi = v.slow.min(), v.fast.max()
        if v.fast.min() < lo or v.slow.max() > hi:
            sys.exit(f'{NAMES[mv]}: a value lies outside [lowest slow, highest fast]')
        pad = 0.05 * (hi - lo)
        ax.plot([0, 1], v[['slow', 'fast']].to_numpy().T, color='k', lw=0.5)    # one line per participant
        ax.plot(np.zeros(len(v)), v.slow, 'o', color='C0')
        ax.plot(np.ones(len(v)), v.fast, 'o', color='C1')
        ax.set_xlim(-0.4, 1.4)
        ax.set_xticks([0, 1], ['slow', 'fast'])
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(YLABEL[mv])
        ax.set_title(f'{NAMES[mv]} ({len(v)} participants)')
    fig.savefig(png)
    fig.savefig(pdf)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids', required=True, help='dataset root')
    ap.add_argument('--out', required=True, help='output folder (new or empty)')
    A = ap.parse_args()
    out = dat.output_folder(A.out)
    V = load(A.bids)
    draw(V, out / 'fig_speed.png', out / 'fig_speed.pdf')
    V.to_csv(out / 'fig_speed_values.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print(rate_table(V).to_string(index=False, formatters={'wilcoxon_p': '{:.2g}'.format}))
