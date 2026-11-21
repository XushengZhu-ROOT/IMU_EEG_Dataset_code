"""Figure of the single trials with the highest IMU-EEG correlation: for each movement, the trial whose first 10 s give
the highest rho (imu_analysis/imu_eeg_correlation.py, 0.3-10 Hz).

    python figures/make_fig_single_trial.py --bids /path/to/dataset --out /path/to/output [--workers 3]

Candidates: Move segments with imu_usable = yes whose first 10 s hold no empty IMU sample. Heatmaps (a)-(f): |r| and rho
of every channel in the first 10 s; waveforms (a1)-(f1): the EEG channel and sensor of the highest rho, the least-squares
combination of the three axes (blue), the single axis with the highest |r| (orange, sign matched to the EEG) and the EEG
(black), each z-scored over the 10 s.

-> fig_single_trial.png (300 dpi), fig_single_trial.pdf, fig_single_trial_values_heatmap.csv,
   fig_single_trial_values_waveform.csv
"""
import argparse
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures import figstyle  # noqa: E402
from imu_analysis import imu_data as dat  # noqa: E402
from imu_analysis import imu_eeg_correlation as ec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FS = ec.FS
EEG20, COLS = ec.EEG20, ec.COLS
NAMES = {0: 'Straight walking', 1: 'Curved walking', 2: 'Head shaking', 3: 'Head nodding', 4: 'Picking up an object',
         5: 'Stair climbing'}
ABS_R = '∣r∣'                                    # |r| with a bar that cannot be read as 'l'
ARGS = None


def run_subject(sub):
    """First 10 s of every Move segment of the subject: the measures, the band-passed EEG and IMU."""
    out = []
    for g, Ei, Yb, miss in ec.segments(ARGS.bids, sub):
        E, Y, clean = ec.first_seconds(Ei, Yb, miss)
        out.append(dict(sub=sub, movement=int(g.cond[8]), speed=int(g.cond[-1]), usable=g.imu_usable,
                        M=ec.measures(E, Y), E=E, Y=Y, clean=clean))
    print(sub, 'done', flush=True)
    return out


def init(a):
    global ARGS
    ARGS = a


def select(T):
    """Per movement: the trial with the highest rho of any channel, its EEG channel, sensor and waveforms."""
    P = []
    for mv in ec.MOVEMENTS:
        best = None
        for _, t in T[(T.movement == mv) & (T.usable == 'yes')].iterrows():
            if t.clean and (best is None or t.M[:, 6:].max() > best[0]):
                best = (t.M[:, 6:].max(), t)
        _, t = best
        M, E, Y = t.M, t.E, t.Y
        j, c = np.unravel_index(np.argmax(M[:, 6:]), (20, 2))
        sens = slice(0, 3) if c == 0 else slice(3, 6)
        x = E[:, j] - E[:, j].mean()
        Yc = Y[:, sens] - Y[:, sens].mean(0)
        w = np.linalg.lstsq(Yc, x, rcond=None)[0]
        sname = 'Acc' if c == 0 else 'Gyro'
        ba = int(np.argmax(M[j, sens.start:sens.stop]))

        def z(v):
            return (v - v.mean()) / v.std()

        P.append(dict(movement=mv, sub=t['sub'], speed='slow' if t.speed == 0 else 'fast', eeg=EEG20[j], sensor=sname, rho=M[j, 6 + c],
                      best_single_axis=f"{sname}{'XYZ'[ba]}", M=M, t=np.arange(len(x)) / FS, eeg_z=z(x), fit_z=z(Yc @ w),
                      axis_z=z(Yc[:, ba]) * np.sign(np.corrcoef(Yc[:, ba], x)[0, 1])))
    return P


def draw(P, png, pdf):
    W = 18.0
    x0, pw, gap = 0.85, 4.95, 0.32                     # heatmaps: left edge, width, gap
    hh = 5.4                                            # heatmap height: 20 rows of 0.27 cm
    wx0, wpw, wgap = 1.05, 5.2, 0.5                     # waveforms
    wh = 1.9
    y_hm1 = 1.8                                         # top of the first heatmap row
    y_hm2 = y_hm1 + hh + 0.65
    y_w1 = y_hm2 + hh + 1.65
    y_w2 = y_w1 + wh + 1.3
    H = y_w2 + wh + 0.85
    pg = figstyle.Page(W, H)
    pg.title('Single-trial IMU–EEG correlation')
    pg.legend([Line2D([], [], color='k', lw=1.0), Line2D([], [], color=figstyle.BLUE, lw=1.0),
               Line2D([], [], color=figstyle.ORANGE, lw=1.0)],
              ['EEG', 'IMU, best combination of the three axes', 'IMU, best single axis'])
    im = None
    for p in P:
        mv = p['movement']
        r_, c_ = divmod(mv, 3)
        ax = pg.axes(x0 + c_ * (pw + gap), y_hm1 if r_ == 0 else y_hm2, pw, hh)
        im = ax.pcolormesh(np.arange(9) - 0.5, np.arange(21) - 0.5, p['M'], cmap='viridis', vmin=0, vmax=1,
                           shading='flat', edgecolors='none', antialiased=False, rasterized=False)
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(19.5, -0.5)
        ax.axvline(5.5, color='white', lw=1.2)
        ax.set_xticks(range(8))
        ax.set_xticklabels(COLS if r_ == 1 else [], rotation=90)
        ax.set_yticks(range(20))
        ax.set_yticklabels(EEG20 if c_ == 0 else [])
        if c_:
            ax.tick_params(axis='y', length=0)
        ax.set_title(f"({'abcdef'[mv]}) {NAMES[mv]}, {p['sub']}, {p['speed']}")
    cax = pg.axes(x0 + 3 * pw + 2 * gap + 0.3, y_hm1, 0.22, y_hm2 + hh - y_hm1)
    cb = pg.fig.colorbar(im, cax=cax, ticks=np.arange(0, 1.01, 0.2))
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, length=2.5, direction='out', colors='k')
    cb.set_label(f'{ABS_R} or ρ', color='k', labelpad=4)
    lim = max(np.abs(np.r_[p['eeg_z'], p['fit_z'], p['axis_z']]).max() for p in P)
    lim = np.ceil(lim * 2) / 2
    for p in P:
        mv = p['movement']
        r_, c_ = divmod(mv, 3)
        ax = pg.axes(wx0 + c_ * (wpw + wgap), y_w1 if r_ == 0 else y_w2, wpw, wh)
        ax.plot(p['t'], p['axis_z'], color=figstyle.ORANGE, lw=0.6)
        ax.plot(p['t'], p['fit_z'], color=figstyle.BLUE, lw=0.7)
        ax.plot(p['t'], p['eeg_z'], color='k', lw=0.7)
        ax.set_xlim(0, 10)
        ax.set_xticks(range(0, 11, 2))
        ax.set_ylim(-lim, lim)
        ax.set_yticks(np.arange(-np.floor(lim), np.floor(lim) + 1, 2))
        if c_ == 0:
            ax.set_ylabel('Amplitude (z)')
        else:
            ax.set_yticklabels([])
        figstyle.x_unit(ax, '(s)')
        ax.set_title(f"({'abcdef'[mv]}1) {p['eeg']} and {p['sensor']}*, ρ = {p['rho']:.2f}")
    figstyle.save(pg.fig, png, pdf)
    return H


def values(P):
    hm = pd.concat([pd.DataFrame(p['M'], columns=COLS).assign(movement=p['movement'], task=NAMES[p['movement']], sub=p['sub'],
                                                                 speed=p['speed'], eeg=EEG20)
                    for p in P])[['movement', 'task', 'sub', 'speed', 'eeg'] + COLS]
    wf = pd.concat([pd.DataFrame(dict(movement=p['movement'], task=NAMES[p['movement']], sub=p['sub'], speed=p['speed'],
                                      eeg_channel=p['eeg'], sensor=p['sensor'], best_single_axis=p['best_single_axis'],
                                      time_s=p['t'], eeg_z=p['eeg_z'], imu_best_combination_z=p['fit_z'],
                                      imu_best_single_axis_z=p['axis_z'])) for p in P])
    return hm, wf


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids', required=True, help='dataset root')
    ap.add_argument('--out', required=True, help='output folder (new or empty)')
    ap.add_argument('--workers', type=int, default=3)
    ARGS = ap.parse_args()
    out = dat.output_folder(ARGS.out)
    with ProcessPoolExecutor(max_workers=ARGS.workers, initializer=init, initargs=(ARGS,)) as ex:
        T = pd.DataFrame([r for rows in ex.map(run_subject, dat.SUBJECTS) for r in rows])
    P = select(T)
    with figstyle.paper_style():
        H = draw(P, out / 'fig_single_trial.png', out / 'fig_single_trial.pdf')
    hm, wf = values(P)
    hm.to_csv(out / 'fig_single_trial_values_heatmap.csv', index=False)
    wf.to_csv(out / 'fig_single_trial_values_waveform.csv', index=False)
    print(f'Move segments: {len(T)}; figure 18 x {H:.1f} cm')
    for p in P:
        print(f"  {NAMES[p['movement']]:21s} {p['sub']} {p['speed']}: {p['eeg']} {p['sensor']}* rho {p['rho']:.4f}")
