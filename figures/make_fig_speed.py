"""Figure of the slow and the fast condition of each movement, measured from the head IMU, and the numbers of the table
of movement intensity and rate.

    python figures/make_fig_speed.py --bids /path/to/dataset --out /path/to/output

One line per participant with a value in both blocks: the intensity (rms of the three-axis acceleration band-passed at
0.3-10 Hz, imu_movement_rate.intensity) of straight walking, curved walking and stair climbing, and the rate
(cycle_rate.py) of head shaking, head nodding and picking up an object. Each y axis spans the lowest slow to the highest
fast value of its panel and 5 % of that range on each side. Printed per movement: median [quartiles] of the slow and the
fast block, median fast/slow ratio, participants with fast > slow, Wilcoxon signed-rank p.

-> fig_speed.png (300 dpi), fig_speed.pdf, fig_speed_values.csv
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures import figstyle  # noqa: E402
from imu_analysis import cycle_rate as cr  # noqa: E402
from imu_analysis import imu_data as dat  # noqa: E402
from imu_analysis import imu_movement_rate as mr  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

NAMES = {0: 'Straight walking', 1: 'Curved walking', 2: 'Head shaking', 3: 'Head nodding', 4: 'Picking up an object',
         5: 'Stair climbing'}
COLUMN = {0: 'intensity_acc_g', 1: 'intensity_acc_g', 2: 'rate', 3: 'rate', 4: 'rate', 5: 'intensity_acc_g'}
YLABEL = {0: 'Intensity (g)', 1: 'Intensity (g)', 2: 'Rate (cycles/min)', 3: 'Rate (cycles/min)',
          4: 'Rate (pick-ups/min)', 5: 'Intensity (g)'}
INTENSITY = [mv for mv in range(6) if COLUMN[mv] == 'intensity_acc_g']


def load(bids):
    """Per movement and participant with a value in both blocks: the slow and the fast value."""
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
    W = 18.0
    lm, pw, gap = 1.3, 4.4, 0.35                         # y labels, plot width, gap after a plot
    ph = 3.7
    y1 = 1.8
    y2 = y1 + ph + 1.25
    H = y2 + ph + 0.62
    pg = figstyle.Page(W, H)
    pg.title('Slow and fast conditions')
    pg.legend([Line2D([], [], ls='none', marker='o', ms=4, mfc=figstyle.BLUE, mec='none'),
               Line2D([], [], ls='none', marker='o', ms=4, mfc=figstyle.ORANGE, mec='none'),
               Line2D([], [], color='k', lw=0.5)],
              ['slow', 'fast', 'one participant'])
    for mv in range(6):
        v = V[V.movement == mv]
        lo, hi = v.slow.min(), v.fast.max()
        if v.fast.min() < lo or v.slow.max() > hi:
            sys.exit(f'{NAMES[mv]}: a value lies outside [lowest slow, highest fast]')
        pad = 0.05 * (hi - lo)
        r_, c_ = divmod(mv, 3)
        ax = pg.axes(lm + c_ * (pw + gap + lm), y1 if r_ == 0 else y2, pw, ph)
        for row in v.itertuples():
            ax.plot([0, 1], [row.slow, row.fast], color='k', lw=0.5, zorder=2)
        ax.scatter(np.zeros(len(v)), v.slow, s=9, color=figstyle.BLUE, edgecolors='none', zorder=3)
        ax.scatter(np.ones(len(v)), v.fast, s=9, color=figstyle.ORANGE, edgecolors='none', zorder=3)
        ax.set_xlim(-0.4, 1.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['slow', 'fast'])
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))
        ax.set_ylabel(YLABEL[mv])
        ax.set_title(f'{NAMES[mv]} ({len(v)} participants)')
    figstyle.save(pg.fig, png, pdf)
    return H


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids', required=True, help='dataset root')
    ap.add_argument('--out', required=True, help='output folder (new or empty)')
    A = ap.parse_args()
    out = dat.output_folder(A.out)
    V = load(A.bids)
    with figstyle.paper_style():
        H = draw(V, out / 'fig_speed.png', out / 'fig_speed.pdf')
    V.to_csv(out / 'fig_speed_values.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print(f'figure 18 x {H:.1f} cm')
    print(rate_table(V).to_string(index=False, formatters={'wilcoxon_p': '{:.2g}'.format}))
