"""Cut a segment from the raw recordings and measure how well its IMU follows the headset accelerometer.

    from sync import resync as rs
    T = rs.load_tables()
    C = rs.cut(T, 'Sub01', 'movement3_speed0', 'Move')
    eeg, imu = rs.cut_from_raw(C, '/path/to/dataset')         # (samples x rows), (slots x 6)
    A = rs.alignment(rs.cgx_acc(eeg, C), imu[:, :3])          # rho, best lag

Positions are 1-based. Both eeg and imu start at the segment's first EEG sample.
"""
import os, re, warnings
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt

TABLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tables')
FS_EEG = 500.0
FS_IMU = 49.647537
SOS_LP = butter(4, 20.0, 'low', fs=FS_EEG, output='sos')               # CGX before it goes onto the IMU slots
SOS_BP = butter(4, (0.3, 10.0), 'bandpass', fs=FS_IMU, output='sos')   # both accelerometers
PAD = int(round(3 * FS_IMU))                                            # odd reflection at both ends before filtering
LAG = int(round(0.5 * FS_IMU))                                          # whole-segment lag search: +/-0.5 s
WIDE = int(round(1.5 * FS_IMU))                                         # lag search inside a low window: +/-1.5 s
MOVING = 0.02                                                           # g, CGX rms of a window with movement
_num = re.compile(r'^\s*-?\d')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*matmul.*')   # spurious on macOS Accelerate BLAS


def load_tables(tables=TABLES):
    """Read segments.csv and pieces.csv."""
    S = pd.read_csv(os.path.join(tables, 'segments.csv'), keep_default_na=False, dtype=str)
    P = pd.read_csv(os.path.join(tables, 'pieces.csv'), keep_default_na=False, dtype=str)
    for c in ('piece', 'first', 'last', 'n', 'seg_first', 'seg_last'):
        P[c] = P[c].astype(int)
    for c in ('eeg_samples', 'imu_slots', 'imu_empty'):
        S[c] = S[c].astype(int)
    return dict(tables=tables, segments=S, pieces=P)


def source_file(bids, sub, name):
    """Path of a raw xdf (sourcedata/sub-XX/eeg) or txt (sourcedata/sub-XX/motion) in the dataset; sub = 'SubXX'."""
    return os.path.join(bids, 'sourcedata', 'sub-' + sub[3:], 'eeg' if name.endswith('.xdf') else 'motion', name)


def read_txt(path):
    """Data rows of a raw IMU txt as (N x 6): accX accY accZ (g), gyroX gyroY gyroZ."""
    rows = []
    with open(path, errors='replace') as fh:
        for line in fh:
            if not _num.match(line):
                continue
            p = line.split(',')
            if len(p) < 6:
                continue
            try:
                rows.append([float(x) for x in p[:6]])
            except ValueError:
                continue
    return np.asarray(rows)


def load_eeg_stream(path, name):
    """EEG stream `name` of an xdf as (samples x rows) in file order, and its row labels."""
    import pyxdf
    streams, _ = pyxdf.load_xdf(path, dejitter_timestamps=False, synchronize_clocks=False)
    for s in streams:
        if s['info']['type'][0] == 'EEG' and s['info']['name'][0] == name:
            return s['time_series'], [c['label'][0] for c in s['info']['desc'][0]['channels'][0]['channel']]
    raise ValueError(f'no EEG stream named {name} in {path}')


def cut(T, sub, cond, seg):
    """Positions of one segment: EEG pieces (xdf, first, last), IMU slots (txt, row, status), the segment row."""
    S, P = T['segments'], T['pieces']
    g = S[(S['sub'] == sub) & (S.cond == cond) & (S.seg == seg)]
    if len(g) != 1:
        raise KeyError(f'{sub} {cond} {seg} not in segments.csv')
    g = g.iloc[0]
    p = P[(P['sub'] == sub) & (P.cond == cond) & (P.seg == seg) & ~P.note.str.startswith('tail')].sort_values('piece')
    eeg = [(r.file, r['first'], r['last']) for _, r in p[p.stream == 'EEG'].iterrows()]
    files, rows, status = [], [], []
    if g.imu_method == 'slot map':
        m = pd.read_csv(os.path.join(T['tables'], g.imu_slot_map), keep_default_na=False, dtype=str)
        m = m.iloc[:int(g.imu_slots)]                                   # (tail slots are after the segment)
        files = list(m.txt_file)
        rows = [int(r) if r else 0 for r in m['row']]
        status = list(m.status)
    elif g.imu_method == 'rows':
        for _, r in p[p.stream == 'IMU'].iterrows():
            for s in range(r.n):                                        # slot s + 1 of the piece = row first + s
                row = r['first'] + s
                ok = row <= r['last']                                   # past the end of the IMU file: empty
                files.append(r.file); rows.append(row if ok else 0); status.append('raw' if ok else 'missing')
    return dict(sub=sub, cond=cond, seg=seg, info=g, eeg=eeg,
                imu=pd.DataFrame(dict(file=files, row=rows, status=status)))


def apply(C, eeg_src, imu_src):
    """Cut a segment from eeg_src {xdf: (samples x rows)} and imu_src {txt: (N x 6)}; missing IMU slots are NaN.
    eeg_src may be processed, as long as each line is still one xdf sample."""
    eeg = np.vstack([eeg_src[f][a - 1:b] for f, a, b in C['eeg']]) if eeg_src is not None else None
    imu = None
    if imu_src is not None and len(C['imu']):
        m = C['imu']
        imu = np.full((len(m), 6), np.nan)
        st = m.status.to_numpy()
        raw = st == 'raw'
        for f in pd.unique(m.file[raw]):
            k = raw & (m.file == f).to_numpy()
            imu[k] = imu_src[f][m.row[k].to_numpy() - 1, :6]
        e = np.diff(np.r_[0, (st == 'interpolated').astype(int), 0])
        for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            if a == 0 or b == len(st) or st[a - 1] != 'raw' or st[b] != 'raw':
                raise ValueError(f'interpolated slots {a + 1}-{b} need a raw slot on both sides')
            for c in range(6):
                imu[a:b, c] = np.interp(np.arange(a, b), [a - 1, b], [imu[a - 1, c], imu[b, c]])
    return eeg, imu


def cut_from_raw(C, bids, cache=None):
    """Load the raw files of a segment from the dataset's sourcedata (once per file with `cache`) and cut them."""
    cache = {} if cache is None else cache
    for f in {f for f, _, _ in C['eeg']}:
        if f not in cache:
            cache[f] = load_eeg_stream(source_file(bids, C['sub'], f), C['info'].eeg_stream)[0]
    for f in set(C['imu'].file):
        if f and f not in cache:
            cache[f] = read_txt(source_file(bids, C['sub'], f))
    return apply(C, cache, cache if len(C['imu']) else None)


def rows_of(spec):
    a, b = (int(x) for x in spec.split('-'))
    return slice(a - 1, b)


def cgx_acc(eeg, C):
    """Headset (CGX) accelerometer of the cut EEG, (n x 3) in g."""
    return np.asarray(eeg[:, rows_of(C['info'].eeg_acc_rows)], float) / 1e6


def _filter(x, sos):
    """Zero-phase filter after odd reflection of PAD samples at both ends."""
    n = min(PAD, len(x) - 1)
    xp = np.vstack([2 * x[0] - x[n:0:-1], x, 2 * x[-1] - x[-2:-n - 2:-1]])
    return sosfiltfilt(sos, xp, axis=0)[n:n + len(x)]


def on_imu_grid(acc, m):
    """CGX (n x 3, 500 Hz) low-passed at 20 Hz and interpolated at the IMU slot times (only within the EEG)."""
    c = _filter(np.asarray(acc, float), SOS_LP)
    t_c = np.arange(len(c)) / FS_EEG
    t_i = np.arange(m) / FS_IMU
    t_i = t_i[t_i <= t_c[-1]]
    return np.column_stack([np.interp(t_i, t_c, c[:, k]) for k in range(3)])


def kabsch_rho(X, Y):
    """Rotation-invariant Procrustes correlation of two (n, 3) signals (1 = the same movement)."""
    X = X - X.mean(0); Y = Y - Y.mean(0)
    U, S, Vt = np.linalg.svd(X.T @ Y)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return float((S[0] + S[1] + d * S[2]) / np.sqrt((X ** 2).sum() * (Y ** 2).sum()))


def prepare(acc, imu3):
    """band-passed CGX and IMU on the IMU slots, and the slots without IMU data."""
    X = _filter(on_imu_grid(acc, len(imu3)), SOS_BP)
    m = len(X)
    Y = np.array(imu3[:m, :3], float)
    miss = np.isnan(Y[:, 0])
    if miss.any():
        g = np.flatnonzero(~miss)
        for k in range(3):
            Y[miss, k] = np.interp(np.flatnonzero(miss), g, Y[g, k])
    return X, _filter(Y, SOS_BP), miss


def alignment(acc, imu3, win_s=5.0, hop_s=1.0, low=0.5):
    """Whole-segment rho and best lag (> 0: the IMU fits best shifted later), and rho per win_s window, with a best
    lag and a class (quiet, offset or no match) for windows below `low`."""
    X, Y, miss = prepare(acc, imu3)
    m = len(X)
    edge = min(int(2 * FS_IMU), max(m // 8, LAG + 1))
    bad = np.convolve(miss.astype(float), np.ones(2 * int(FS_IMU) + 1), 'same') > 0   # 1 s around empty slots
    idx = np.flatnonzero(~bad & (np.arange(m) >= edge) & (np.arange(m) < m - edge))
    curve = {d: kabsch_rho(X[idx], Y[idx - d]) for d in range(-LAG, LAG + 1) if (idx - d).min() >= 0 and (idx - d).max() < m}
    best = max(curve, key=curve.get)
    W, H = int(round(win_s * FS_IMU)), int(round(hop_s * FS_IMU))
    wins = []
    starts = list(range(0, max(m - W, 0) + 1, H)) or [0]
    if starts[-1] + W < m:
        starts.append(m - W)                                             # the last window ends with the segment
    for s in starts:
        w = np.arange(s, min(s + W, m))
        w = w[~miss[w]]
        if len(w) < W // 2:
            continue
        r0 = kabsch_rho(X[w], Y[w])
        amp = float(np.sqrt((X[w] ** 2).sum(1).mean()))
        row = dict(t_start_s=round(s / FS_IMU, 2), t_end_s=round(min(s + W, m) / FS_IMU, 2), rho=round(r0, 3),
                   amp_g=round(amp, 4), best_lag='', rho_best='', cls='')
        if r0 < low:
            c = {d: kabsch_rho(X[w], Y[w - d]) for d in range(-WIDE, WIDE + 1) if (w - d).min() >= 0 and (w - d).max() < m}
            d = max(c, key=c.get)
            row.update(best_lag=d, rho_best=round(c[d], 3),
                       cls='quiet' if amp < MOVING else 'offset' if d != 0 and c[d] >= r0 + 0.2 else 'no match')
        wins.append(row)
    return dict(rho=round(curve[0], 4), best_lag=best, rho_best=round(curve[best], 4), windows=pd.DataFrame(wins),
                X=X, Y=Y, miss=miss)
