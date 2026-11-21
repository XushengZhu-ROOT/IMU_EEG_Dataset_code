"""Agreement of the IMU with the headset accelerometer in every segment, as released and after the correction.

    python sync/alignment_report.py --bids /path/to/dataset --out alignment_out [--tables tables] [--workers 3]

Headset (CGX) accelerometer: rows eeg_acc_rows of the xdf's EEG stream / 1e6 (g) at the segment's samples, i.e. the
channels ACC1-3 of the dataset's EEG; as released, with the tail pieces of pieces.csv. IMU as released: the released
rows of every piece of imu_pieces.csv (tail pieces included) from the raw txt. IMU after the correction: the dataset's
_motion.tsv. rho: CGX low-passed at 20 Hz and interpolated at the IMU slot times, both band-passed 0.3-10 Hz (zero
phase, scipy's default padding), IMU turned into the CGX axes by the best proper rotation; 2 s left out at both ends and
1 s around empty IMU slots; best lag within +/-0.5 s (> 0: IMU moved later).
-> <out>/alignment_release_vs_final.csv (one row per segment), <out>/alignment_summary.txt
"""
import os, sys, argparse, pathlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy.signal import sosfiltfilt
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sync import resync as rs

ARGS = T = IMU = None


def on_imu_grid(acc, m):
    """CGX (n x 3, 500 Hz) low-passed at 20 Hz and interpolated at the times of IMU slots 1..m within the EEG."""
    c = sosfiltfilt(rs.SOS_LP, acc, axis=0)
    t_c = np.arange(len(c)) / rs.FS_EEG
    t_i = np.arange(m) / rs.FS_IMU
    t_i = t_i[t_i <= t_c[-1]]
    return np.column_stack([np.interp(t_i, t_c, c[:, k]) for k in range(3)])


def whole_segment(acc, imu):
    """rho at lag 0 and the best lag of an IMU segment (m x 3, NaN = no data) against the CGX (n x 3)."""
    X = sosfiltfilt(rs.SOS_BP, on_imu_grid(acc, len(imu)), axis=0)
    m = len(X)
    Y = np.array(imu[:m, :3], float)
    miss = np.isnan(Y).any(1)
    if miss.any():
        g = np.flatnonzero(~miss)
        for k in range(3):
            Y[miss, k] = np.interp(np.flatnonzero(miss), g, Y[g, k])
    Y = sosfiltfilt(rs.SOS_BP, Y, axis=0)
    edge = min(int(2 * rs.FS_IMU), max(m // 8, rs.LAG + 1))
    keep = np.zeros(m, bool)
    keep[edge:m - edge] = True
    if miss.any():
        keep &= ~binary_dilation(miss, iterations=int(rs.FS_IMU))
    idx = np.flatnonzero(keep)
    curve = {d: rs.kabsch_rho(X[idx], Y[idx - d]) for d in range(-rs.LAG, rs.LAG + 1)}
    best = max(curve, key=curve.get)
    return round(curve[0], 4), best


def cgx(cache, sub, g, xdf):
    """CGX accelerometer (g) of a whole xdf."""
    if xdf not in cache:
        eeg, _ = rs.load_eeg_stream(rs.source_file(ARGS.bids, sub, xdf), g.eeg_stream)
        cache[xdf] = np.asarray(eeg[:, rs.rows_of(g.eeg_acc_rows)], float) / 1e6
    return cache[xdf]


def txt_rows(cache, sub, txt, first, last):
    """accX accY accZ of the txt rows first..last (1-based); rows outside the txt give NaN."""
    if txt not in cache:
        cache[txt] = rs.read_txt(rs.source_file(ARGS.bids, sub, txt))
    r = np.arange(first, last + 1)
    v = np.full((len(r), 3), np.nan)
    ok = (r >= 1) & (r <= len(cache[txt]))
    v[ok] = cache[txt][r[ok] - 1, :3]
    return v


def of_segment(df, key):
    return df[(df['sub'] == key[0]) & (df.cond == key[1]) & (df.seg == key[2])]


def segment(key, cache):
    sub = key[0]
    g = of_segment(T['segments'], key).iloc[0]
    i = of_segment(IMU['segments'], key).iloc[0]
    out = dict(sub=sub, cond=key[1], seg=key[2], move=g['move'], speed=g['speed'], imu_usable=g['imu_usable'],
               correction=i['correction'])
    if not g['imu_slots']:
        return out
    P = of_segment(T['pieces'], key)
    acc = np.vstack([cgx(cache, sub, g, p.file)[p.first - 1:p.last]
                     for p in P[P.stream == 'EEG'].sort_values('piece').itertuples()])
    released = np.vstack([txt_rows(cache, sub, p.txt, p.released_first_row, p.released_last_row)
                          for p in of_segment(IMU['pieces'], key).sort_values('piece').itertuples()])
    final = pd.read_csv(os.path.join(ARGS.bids, i['bids_file']), sep='\t', header=None, na_values='n/a',
                        float_precision='round_trip').iloc[:, :3].to_numpy(float)
    out['rho_release'], out['best_lag_release'] = whole_segment(acc, released)
    out['rho_final'], out['best_lag_final'] = whole_segment(acc[:g['eeg_samples']], final)
    return out


def run_recording(keys):
    """The segments of one recording; its raw files are loaded once."""
    cache = {}
    return [segment(k, cache) for k in keys]


def summary(R):
    lines = []
    for seg, need in (('Move', 0.8), ('Rest', 0.5)):
        d = R[(R.seg == seg) & R.rho_release.notna()]
        ok = d[d.imu_usable.isin(['yes', 'partly'])]
        lines.append(f'{seg}: {len(d)} segments with an IMU accelerometer, {len(ok)} of them with usable IMU '
                     f'(imu_usable yes or partly)')
        for tag, name in (('release', 'as released'), ('final', 'corrected')):
            r, lag = ok[f'rho_{tag}'], ok[f'best_lag_{tag}']
            lines.append(f'  {name:11s} rho median {r.median():.3f} (IQR {r.quantile(.25):.3f}-{r.quantile(.75):.3f}), '
                         f'rho >= {need}: {int((r >= need).sum())}/{len(ok)}; best lag 0: {int((lag == 0).sum())}/{len(ok)}, '
                         f'|best lag| <= 1 sample (20 ms): {int((lag.abs() <= 1).sum())}/{len(ok)}')
    return '\n'.join(lines)


def init(args):
    global ARGS, T, IMU
    ARGS, T = args, rs.load_tables(args.tables)
    IMU = dict(segments=pd.read_csv(os.path.join(args.tables, 'imu_segments.csv'), keep_default_na=False),
               pieces=pd.read_csv(os.path.join(args.tables, 'imu_pieces.csv'), keep_default_na=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='IMU against the headset accelerometer, as released and corrected.')
    ap.add_argument('--bids', required=True, help='root folder of the dataset')
    ap.add_argument('--tables', default=rs.TABLES, help='folder of the cut tables (default: tables/ of this repository)')
    ap.add_argument('--out', required=True, help='output folder; must not exist or be empty')
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()
    if os.path.isdir(args.out) and os.listdir(args.out):
        sys.exit(f'{args.out} is not empty')
    os.makedirs(args.out, exist_ok=True)
    S = rs.load_tables(args.tables)['segments']
    keys = list(zip(S['sub'], S.cond, S.seg))
    by_recording = {}
    for k, xdf in zip(keys, S.eeg_xdf):
        by_recording.setdefault(xdf, []).append(k)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init, initargs=(args,)) as ex:
        R = pd.DataFrame([r for rows in ex.map(run_recording, by_recording.values()) for r in rows])
    R = R.set_index(['sub', 'cond', 'seg']).loc[keys].reset_index()
    R.to_csv(os.path.join(args.out, 'alignment_release_vs_final.csv'), index=False)
    s = summary(R)
    with open(os.path.join(args.out, 'alignment_summary.txt'), 'w') as fh:
        fh.write(s + '\n')
    print(s)
