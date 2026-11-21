"""For each movement, the trial whose first 10 s give the highest IMU-EEG rho (0.3-10 Hz; usable trials without empty
IMU samples): |r| and rho of every EEG channel, and z-scored waveforms of the best channel and sensor.

    python figures/make_fig_single_trial.py --bids /path/to/dataset --out /path/to/output [--workers 3]

Writes fig_single_trial.png, fig_single_trial.pdf, fig_single_trial_values_heatmap.csv and
fig_single_trial_values_waveform.csv.
"""
import argparse
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from imu_analysis import imu_data as dat  # noqa: E402
from imu_analysis import imu_eeg_correlation as ec  # noqa: E402

FS = ec.FS
EEG20, COLS = ec.EEG20, ec.COLS
NAMES = {0: 'Straight walking', 1: 'Curved walking', 2: 'Head shaking', 3: 'Head nodding', 4: 'Picking up an object',
         5: 'Stair climbing'}
ABS_R = '∣r∣'
ARGS = None


def run_subject(sub):
    """Measures, band-passed EEG and IMU of the first 10 s of every Move segment of the subject."""
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
    """Per movement: the trial with the highest rho, its EEG channel, sensor and waveforms."""
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
    """Rows 1-2: heatmaps (a)-(f); rows 3-4: waveforms (a1)-(f1)."""
    fig, axs = plt.subplots(4, 3, figsize=(12, 15), sharey='row', height_ratios=[3, 3, 1, 1], layout='constrained')
    fig.suptitle('Single-trial IMU–EEG correlation')
    lim = max(np.abs(np.r_[p['eeg_z'], p['fit_z'], p['axis_z']]).max() for p in P)
    lim = np.ceil(lim * 2) / 2
    for p in P:
        mv = p['movement']
        r_, c_ = divmod(mv, 3)
        ax = axs[r_, c_]
        im = ax.imshow(p['M'], vmin=0, vmax=1, aspect='auto')
        ax.axvline(5.5, color='w')
        ax.set_xticks(range(8), COLS, rotation=90)
        ax.set_yticks(range(20), EEG20)
        ax.set_title(f"({'abcdef'[mv]}) {NAMES[mv]}, {p['sub']}, {p['speed']}")
        ax = axs[r_ + 2, c_]
        ax.plot(p['t'], p['axis_z'], color='C1', lw=0.8, label='IMU, best single axis')
        ax.plot(p['t'], p['fit_z'], color='C0', lw=0.8, label='IMU, best combination of the three axes')
        ax.plot(p['t'], p['eeg_z'], color='k', lw=0.8, label='EEG')
        ax.set_xlim(0, 10)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel('Time (s)')
        ax.set_title(f"({'abcdef'[mv]}1) {p['eeg']} and {p['sensor']}*, ρ = {p['rho']:.2f}")
    for ax in axs[2:, 0]:
        ax.set_ylabel('Amplitude (z)')
    fig.colorbar(im, ax=axs[:2], label=f'{ABS_R} or ρ')
    handles, labels = axs[2, 0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='outside lower center', ncols=3)
    fig.savefig(png)
    fig.savefig(pdf)


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
    draw(P, out / 'fig_single_trial.png', out / 'fig_single_trial.pdf')
    hm, wf = values(P)
    hm.to_csv(out / 'fig_single_trial_values_heatmap.csv', index=False)
    wf.to_csv(out / 'fig_single_trial_values_waveform.csv', index=False)
    print(f'Move segments: {len(T)}')
    for p in P:
        print(f"  {NAMES[p['movement']]:21s} {p['sub']} {p['speed']}: {p['eeg']} {p['sensor']}* rho {p['rho']:.4f}")
