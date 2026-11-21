"""Figure of 5 s of EEG and head IMU from the sitting baseline and the six movements (fast condition) of one participant.

    python figures/make_fig_5s_example.py --bids /path/to/dataset --out /path/to/output

Participant: the first in ID order (participants.tsv) with a sitting baseline, all six fast movement segments, IMUUsable
= yes in the _motion.json of all seven segments, and a clean window in each of them.
Window: the 5 s from 60 s after the first EEG sample of the segment; if it holds an n/a IMU value, a clipped gyroscope
value (|value| >= 990 deg/s), a saturated accelerometer value (|value| >= 1.98 g) or a boundary event, the next 5 s
(65 s, 70 s, ...) until a window is clean.
EEG: Fz Cz Pz O1 T3 T4 in uV (reference A1, as recorded), for display the first filters of
eeg_cleaning/clean_paper_versions.m: a 4th-order Butterworth high-pass at 0.5 Hz and Butterworth band-stops at 60, 120,
180 and 240 Hz (butter with N = 2, f0 +/- 1 Hz), each run forward and backward over the continuous stretch between
boundary events that holds the window; channel baselines 1000 uV apart. IMU: the values of the _motion.tsv; the axes
span the sensor ranges (+/-2 g, +/-1000 deg/s).

-> fig_5s_example.png (300 dpi), fig_5s_example.pdf, fig_5s_example_values.csv (the plotted values)
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures import figstyle  # noqa: E402
from imu_analysis import imu_data as dat  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

MOVES = [('walkstraight', 'straight walking'), ('walkcurved', 'curved walking'), ('headshake', 'head shaking'),
         ('headnod', 'head nodding'), ('pickup', 'picking up'), ('stairs', 'stairs')]
CHANNELS = ['Fz', 'Cz', 'Pz', 'O1', 'T3', 'T4']
IMU_COLS = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']
WIN, START, STEP = 5.0, 60.0, 5.0          # s
GYRO_CLIP = 990.0                           # deg/s (device units); the gyroscope stops at about +992 / -1000
ACC_CLIP = 1.98                             # g; the accelerometer stops at +1.984 to +1.992 and -2.000 to -2.008
HP_HZ, HP_ORDER = 0.5, 4
LINE_HZ, LINE_HALF_BW, LINE_ORDER = (60, 120, 180, 240), 1.0, 2
SPACING = 1000.0                            # uV between EEG channel baselines = length of the scale bar
ACC_LIM, GYRO_LIM = 2.0, 1000.0             # sensor ranges
C_REF = (0.82, 0.82, 0.82)                  # zero lines
STYLE = {'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5, 'lines.solid_joinstyle': 'round'}


def load_segment(bids, sub, task, acq):
    eeg_stem = os.path.join(sub, 'eeg', f'{sub}_task-{task}_acq-{acq}')
    mot_stem = os.path.join(sub, 'motion', f'{sub}_task-{task}_tracksys-imu_acq-{acq}')
    need = [eeg_stem + '_eeg.vhdr', eeg_stem + '_events.tsv', mot_stem + '_motion.tsv', mot_stem + '_motion.json']
    if not all(os.path.exists(os.path.join(bids, p)) for p in need):
        return None
    with open(os.path.join(bids, mot_stem + '_motion.json'), encoding='utf-8') as f:
        mjson = json.load(f)
    ev = pd.read_csv(os.path.join(bids, eeg_stem + '_events.tsv'), sep='\t')
    return {
        'task': task, 'acq': acq, 'vhdr': os.path.join(bids, eeg_stem + '_eeg.vhdr'), 'mjson': mjson,
        'boundaries': np.sort(ev.loc[ev['trial_type'] == 'boundary', 'onset'].to_numpy(float)),
        'imu': pd.read_csv(os.path.join(bids, mot_stem + '_motion.tsv'), sep='\t', header=None,
                           na_values='n/a').to_numpy(float),
    }


def find_window(seg, n_eeg, fs_eeg):
    """First clean 5-s window from 60 s on, in 5-s steps: (t0, IMU row indices, rejected windows)."""
    imu, lat, bnd = seg['imu'], seg['imu'][:, 6], seg['boundaries']
    n_min = int(np.floor(WIN * float(seg['mjson']['SamplingFrequencyEffective'])))
    rejected, t0 = [], START
    while t0 + WIN <= n_eeg / fs_eeg + 1e-9:
        rows = np.flatnonzero((lat >= t0 - 1e-9) & (lat < t0 + WIN - 1e-9))
        w = imu[rows, :6]
        why = []
        if len(rows) < n_min:
            why.append('IMU ends')
        if np.isnan(w).any():
            why.append(f'{int(np.isnan(w).any(1).sum())} n/a IMU rows')
        with np.errstate(invalid='ignore'):
            g = (np.abs(w[:, 3:6]) >= GYRO_CLIP).any(1)
            a = (np.abs(w[:, 0:3]) >= ACC_CLIP).any(1)
        if g.any():
            why.append(f'{int(g.sum())} clipped gyro rows')
        if a.any():
            why.append(f'{int(a.sum())} saturated acc rows')
        if ((bnd >= t0) & (bnd <= t0 + WIN)).any():
            why.append('boundary')
        if not why:
            return t0, rows, rejected
        rejected.append(f'{t0:g}-{t0 + WIN:g} s: ' + ', '.join(why))
        t0 += STEP
    return None, None, rejected


def stretch_filter(x, k0, k1, boundaries, fs, sos_chain):
    """The filters of sos_chain one after the other, each forward and backward, over the continuous stretch (between
    boundary events) that holds samples k0..k1-1; returns the window."""
    edges = np.unique(np.r_[0, np.round(boundaries * fs).astype(int), len(x)])
    p = np.searchsorted(edges, k0, side='right') - 1
    a, b = int(edges[p]), int(edges[p + 1])
    assert a <= k0 and k1 <= b, 'window crosses a boundary'
    y = x[a:b].astype(np.float64)
    for sos in sos_chain:
        y = sosfiltfilt(sos, y, axis=0)
    return y[k0 - a:k1 - a]


def display_filters(fs):
    """0.5-Hz high-pass (butter with N = 4) and band-stops at 60/120/180/240 Hz (butter with N = 2, f0 +/- 1 Hz)."""
    chain = [butter(HP_ORDER, HP_HZ, btype='highpass', fs=fs, output='sos')]
    for f0 in LINE_HZ:
        chain.append(butter(LINE_ORDER, [f0 - LINE_HALF_BW, f0 + LINE_HALF_BW], btype='bandstop', fs=fs, output='sos'))
    return chain


def select(bids):
    parts = pd.read_csv(os.path.join(bids, 'participants.tsv'), sep='\t', dtype=str).sort_values('participant_id')
    for _, r in parts.iterrows():
        sub = r['participant_id']
        if r['baseline_posture'] != 'sit':
            print(f'{sub}: baseline {r["baseline_posture"]} -> next participant')
            continue
        wanted = [('rest', 'sit', 'sitting baseline')] + [(t, 'fastmove', title) for t, title in MOVES]
        panels, reason = [], ''
        for task, acq, title in wanted:
            seg = load_segment(bids, sub, task, acq)
            if seg is None:
                reason = f'{task} {acq}: file missing'
                break
            if seg['mjson']['Quality'].get('IMUUsable') != 'yes':
                reason = f'{task} {acq}: IMUUsable = {seg["mjson"]["Quality"].get("IMUUsable")}'
                break
            names, fs, data = dat.read_brainvision(seg['vhdr'])
            t0, rows, rejected = find_window(seg, data.shape[0], fs)
            if t0 is None:
                reason = f'{task} {acq}: no clean 5-s window'
                break
            seg.update(title=title, names=names, fs=fs, data=data, t0=t0, rows=rows, rejected=rejected)
            panels.append(seg)
        if reason:
            print(f'{sub}: {reason} -> next participant')
            continue
        print(f'{sub}: chosen')
        return sub, panels
    sys.exit('no participant meets the rule')


def draw(panels, sub_paper, eeg_lim, png, pdf):
    """18 cm wide; one column per task; rows EEG, accelerometer, gyroscope (sizes in cm)."""
    W, LEFT, RIGHT, GAP_X = 18.0, 1.5, 0.2, 0.32
    EEG_TOP, EEG_H, GAP_Y, IMU_H, BOTTOM = 1.55, 6.4, 0.3, 2.3, 0.72
    H = EEG_TOP + EEG_H + GAP_Y + IMU_H + GAP_Y + IMU_H + BOTTOM
    w = (W - LEFT - RIGHT - 6 * GAP_X) / 7
    rows_b = [H - EEG_TOP - EEG_H, H - EEG_TOP - EEG_H - GAP_Y - IMU_H, BOTTOM]
    rows_h = [EEG_H, IMU_H, IMU_H]
    ylab_x = -1.08 / w                                          # right edge of the y labels, in axes units
    fig = plt.figure(figsize=(W * figstyle.CM, H * figstyle.CM), facecolor='w')
    for j, seg in enumerate(panels):
        x0 = LEFT + j * (w + GAP_X)
        ax_e, ax_a, ax_g = [fig.add_axes([x0 / W, b / H, w / W, h / H]) for b, h in zip(rows_b, rows_h)]
        for ax in (ax_e, ax_a, ax_g):
            ax.set_xlim(0, WIN)
            ax.set_xticks(np.arange(0, WIN + 0.01, 1))
            ax.tick_params(labelbottom=ax is ax_g, labelleft=j == 0)
        for i in range(len(CHANNELS)):
            ax_e.plot(seg['t_eeg'], seg['eeg'][:, i] - i * SPACING, color='k', lw=0.35)
        ax_e.set_ylim(eeg_lim)
        ax_e.set_yticks([-i * SPACING for i in range(len(CHANNELS))])
        ax_e.set_yticklabels(CHANNELS)
        ax_e.set_title(seg['title'], pad=4)
        # zero line, then Z, Y, X so that the smaller X and Y stay visible
        for ax, cols, lim, ticks in ((ax_a, (0, 1, 2), ACC_LIM, [-2, -1, 0, 1, 2]),
                                     (ax_g, (3, 4, 5), GYRO_LIM, [-1000, -500, 0, 500, 1000])):
            ax.axhline(0, color=C_REF, lw=0.5, zorder=0)
            for c, col in zip(cols[::-1], (figstyle.GREEN, figstyle.ORANGE, figstyle.BLUE)):
                ax.plot(seg['t_imu'], seg['imu_win'][:, c], color=col, lw=0.7)
            ax.set_ylim(-lim, lim)
            ax.set_yticks(ticks)
        if j == 0:
            ax_a.set_ylabel('acc (g)')
            ax_g.set_ylabel('gyro (deg/s)')
            for ax in (ax_a, ax_g):
                ax.yaxis.set_label_coords(ylab_x, 0.5)
            # EEG scale bar of SPACING uV between the baselines of the 3rd and 4th channel, left of the names
            tr = blended_transform_factory(ax_e.transAxes, ax_e.transData)
            y0 = -(len(CHANNELS) / 2) * SPACING
            ax_e.plot([-0.86 / w] * 2, [y0, y0 + SPACING], color='k', lw=1.0, transform=tr, clip_on=False,
                      solid_capstyle='butt')
            ax_e.text(ylab_x, y0 + SPACING / 2, f'{SPACING:.0f} µV', rotation=90, ha='right', va='center',
                      transform=tr)
        figstyle.x_unit(ax_g, '(s)')
    fig.text(0.5, 1 - 0.34 / H, f'EEG and head IMU in each task ({sub_paper})', ha='center', va='center',
             fontsize=11, fontweight='bold')
    handles = [Line2D([], [], color=c, lw=1.4) for c in (figstyle.BLUE, figstyle.ORANGE, figstyle.GREEN)]
    fig.legend(handles, ['X', 'Y', 'Z'], loc='center left', bbox_to_anchor=(0.004, 1 - 0.88 / H), ncol=3,
               frameon=False, handlelength=2.2, handletextpad=0.5, columnspacing=1.6, borderaxespad=0,
               borderpad=0)
    figstyle.save(fig, png, pdf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids', required=True, help='dataset root')
    ap.add_argument('--out', required=True, help='output folder (new or empty)')
    args = ap.parse_args()
    out = dat.output_folder(args.out)
    sub, panels = select(args.bids)
    sub_paper = 'Sub' + sub.split('-')[1]

    val_rows = []
    for seg in panels:
        fs, t0 = seg['fs'], seg['t0']
        k0 = int(round(t0 * fs))
        k1 = k0 + int(round(WIN * fs))
        x = seg['data'][:, [seg['names'].index(c) for c in CHANNELS]]
        eeg = stretch_filter(x, k0, k1, seg['boundaries'], fs, display_filters(fs))
        assert np.isfinite(eeg).all()
        imu = seg['imu'][seg['rows']]
        t_eeg = np.arange(k0, k1) / fs
        seg.update(eeg=eeg, t_eeg=t_eeg - t0, imu_win=imu, t_imu=imu[:, 6] - t0)
        for j in range(k1 - k0):
            val_rows.append([seg['task'], 'EEG', k0 + j + 1, t_eeg[j], seg['t_eeg'][j]] + list(eeg[j]) + [np.nan] * 6)
        for j, r in enumerate(seg['rows']):
            val_rows.append([seg['task'], 'IMU', int(r) + 1, imu[j, 6], seg['t_imu'][j]] + [np.nan] * 6 + list(imu[j, :6]))

    # one EEG scale for all panels: the limits hold every plotted sample
    lo = min((seg['eeg'][:, i] - i * SPACING).min() for seg in panels for i in range(len(CHANNELS)))
    hi = max((seg['eeg'][:, i] - i * SPACING).max() for seg in panels for i in range(len(CHANNELS)))
    pad = 0.02 * (hi - lo)
    eeg_lim = (lo - pad, hi + pad)
    imu_all = np.vstack([seg['imu_win'][:, :6] for seg in panels])
    assert np.abs(imu_all[:, :3]).max() < ACC_LIM and np.abs(imu_all[:, 3:]).max() < GYRO_LIM

    with figstyle.paper_style(**STYLE):
        draw(panels, sub_paper, eeg_lim, out / 'fig_5s_example.png', out / 'fig_5s_example.pdf')
    cols = (['task', 'signal', 'source_index_1based', 'latency_s', 'time_in_figure_s'] + [f'{c}_uV' for c in CHANNELS]
            + [f'{c}_{"g" if c.startswith("acc") else "dps"}' for c in IMU_COLS])
    pd.DataFrame(val_rows, columns=cols).to_csv(out / 'fig_5s_example_values.csv', index=False, float_format='%.6g')
    print(f'EEG scale: baselines {SPACING:.0f} uV apart, limits {eeg_lim[0]:.0f} to {eeg_lim[1]:.0f} uV')
    for seg in panels:
        print(f'{seg["task"]:12s} {seg["acq"]:8s} window {seg["t0"]:g}-{seg["t0"] + WIN:g} s '
              f'(windows rejected before: {len(seg["rejected"])})')


if __name__ == '__main__':
    main()
