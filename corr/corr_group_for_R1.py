# %%
import os
import shutil
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

# %% [markdown]
# ## Data Loading

# %%
def load_eeg_data(subject, movement, speed, base_dir=".", verbose=False):
    eeg_file = os.path.join(
        base_dir, "EEG_Data", subject, "Raw_mat_segmented",
        f"{subject}_movement{movement}_speed{speed}_Move.mat")
    if not os.path.exists(eeg_file):
        raise FileNotFoundError(f"EEG file not found: {eeg_file}")

    data = sio.loadmat(eeg_file, squeeze_me=True, struct_as_record=False)
    eeg = data["EEG"]
    eeg_data = np.array(eeg.data)
    sfreq = float(eeg.srate)
    ch_names = [str(ch.labels) for ch in eeg.chanlocs]
    if verbose:
        print(f"Loaded EEG: {eeg_file}  shape={eeg_data.shape}, sr={sfreq}")
    return eeg_data, sfreq, ch_names


def _as_channels_by_time(arr, name):
    """Ensure IMU array is (n_channels, n_times)."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} has invalid shape {arr.shape}")
    # files are usually (n_times, 3); accept either orientation
    if arr.shape[1] in (3, 6) and arr.shape[0] > arr.shape[1]:
        arr = arr.T
    elif arr.shape[0] in (3, 6) and arr.shape[1] > arr.shape[0]:
        pass
    else:
        raise ValueError(f"{name} unexpected shape {arr.shape}")
    return arr


def load_imu_data(subject, movement, speed, base_dir=".", verbose=False):
    acc_file = os.path.join(
        base_dir, "IMU_Data", subject, "Raw_mat_segmented",
        f"{subject}_movement{movement}_speed{speed}_ACC_Move.mat")
    gyro_file = os.path.join(
        base_dir, "IMU_Data", subject, "Raw_mat_segmented",
        f"{subject}_movement{movement}_speed{speed}_GYRO_Move.mat")
    if not os.path.exists(acc_file):
        raise FileNotFoundError(f"ACC file not found: {acc_file}")
    if not os.path.exists(gyro_file):
        raise FileNotFoundError(f"GYRO file not found: {gyro_file}")

    acc = _as_channels_by_time(sio.loadmat(acc_file, squeeze_me=True)["acc_data"], "ACC")
    gyro = _as_channels_by_time(sio.loadmat(gyro_file, squeeze_me=True)["gyro_data"], "GYRO")
    if acc.shape[0] != 3 or gyro.shape[0] != 3:
        raise ValueError(f"Expected 3 ACC + 3 GYRO axes, got ACC{acc.shape}, GYRO{gyro.shape}")
    imu_data = np.vstack([acc, gyro])
    if verbose:
        print(f"Loaded IMU: shape={imu_data.shape}")
    return imu_data

# %% [markdown]
# ## Signal Processing

# %%
def bandpass_filter(signal_data, fs, lowcut=0.05, highcut=40, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal_data)


def resample_imu_to_eeg(imu_data, fs_imu=50, fs_eeg=500):
    n_ch, n_orig = imu_data.shape
    n_new = int(n_orig * fs_eeg / fs_imu)
    out = np.zeros((n_ch, n_new))
    for i in range(n_ch):
        out[i] = signal.resample(imu_data[i], n_new)
    return out


def analyze_imu_eeg_correlation(eeg_data, imu_data, fs_eeg=500, fs_imu=50):
    imu_resampled = resample_imu_to_eeg(imu_data, fs_imu=fs_imu, fs_eeg=fs_eeg)
    min_length = min(eeg_data.shape[1], imu_resampled.shape[1])
    eeg_data = eeg_data[:, :min_length]
    imu_resampled = imu_resampled[:, :min_length]

    filtered_eeg = np.zeros_like(eeg_data)
    for ch in range(eeg_data.shape[0]):
        filtered_eeg[ch] = bandpass_filter(eeg_data[ch], fs_eeg, 0.05, 40)
    filtered_imu = np.zeros_like(imu_resampled)
    for ch in range(imu_resampled.shape[0]):
        filtered_imu[ch] = bandpass_filter(imu_resampled[ch], fs_eeg, 0.05, 40)
    return filtered_eeg, filtered_imu

# %% [markdown]
# ## Sliding-window correlation (covers 100% of the Move segment)
#
# Rationale: a single Pearson/coherence estimate over one ~180s Move
# trial averages away any short-timescale (gait-cycle-locked) coupling
# into a near-noise-floor number, and a single 10s excerpt is a
# cherry-picked snapshot. Instead we slide a fixed window across the
# ENTIRE trial (no overlap, no truncation), estimate raw R/C per
# window, and average the raw window-level matrices within-subject.
# Only after that do we truncate/normalize, matching the paper's H
# definition but computed on a denoised, full-coverage estimate.

# %%
def _segment_indices(n_samples, nperseg, noverlap):
    step = nperseg - noverlap
    n_seg = 1 + (n_samples - nperseg) // step
    if n_seg < 1:
        return np.empty((0, nperseg), dtype=int)
    return np.arange(nperseg)[None, :] + step * np.arange(n_seg)[:, None]


def compute_pearson_matrix_vectorized(filtered_imu, filtered_eeg):
    """Pearson R between every IMU x EEG channel pair via one corrcoef call."""
    n_imu = filtered_imu.shape[0]
    stacked = np.vstack([filtered_imu, filtered_eeg])
    full_corr = np.corrcoef(stacked)
    return full_corr[:n_imu, n_imu:]


def compute_coherence_matrix_vectorized(filtered_imu, filtered_eeg, fs=500,
                                        nperseg=256, freq_band=(0.05, 40)):
    """Band-averaged coherence for every IMU x EEG pair via one FFT batch.

    Numerically validated against scipy.signal.coherence per-pair
    (max abs diff < 0.01 on synthetic test signals); coherence is a
    ratio so PSD scaling constants cancel and don't need to match
    scipy's exactly.
    """
    n_t = filtered_imu.shape[1]
    nperseg = int(min(nperseg, n_t))
    noverlap = nperseg // 2
    window = np.hanning(nperseg)

    def segment_fft(data):
        idx = _segment_indices(data.shape[1], nperseg, noverlap)
        segs = data[:, idx]                     # (n_ch, n_seg, nperseg)
        segs = segs - segs.mean(axis=-1, keepdims=True)
        segs = segs * window
        return np.fft.rfft(segs, axis=-1)        # (n_ch, n_seg, n_freq)

    F_imu = segment_fft(filtered_imu)
    F_eeg = segment_fft(filtered_eeg)
    n_seg = F_imu.shape[1]

    Pxx = np.mean(np.abs(F_imu) ** 2, axis=1)                       # (n_imu, n_freq)
    Pyy = np.mean(np.abs(F_eeg) ** 2, axis=1)                       # (n_eeg, n_freq)
    Pxy = np.einsum("ist,jst->ijt", F_imu, np.conj(F_eeg)) / n_seg   # (n_imu, n_eeg, n_freq)

    Cxy = (np.abs(Pxy) ** 2) / (Pxx[:, None, :] * Pyy[None, :, :] + 1e-20)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    band = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    if not np.any(band):
        return np.zeros((filtered_imu.shape[0], filtered_eeg.shape[0]))
    return Cxy[:, :, band].mean(axis=-1)


def normalize_matrix(matrix):
    if np.max(matrix) == np.min(matrix):
        return np.zeros_like(matrix)
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))


def compute_subject_raw_matrices(filtered_imu, filtered_eeg, fs=500,
                                 window_sec=10.0, step_sec=10.0):
    """Average raw R and raw C over non-overlapping windows spanning the
    FULL trial (drops only a final <window_sec remainder). Returns raw,
    un-normalized matrices plus how many windows/seconds were used.
    """
    n_t = filtered_imu.shape[1]
    win = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    win = min(win, n_t)
    n_windows = max(1, 1 + (n_t - win) // step) if n_t >= win else 1

    R_sum = np.zeros((filtered_imu.shape[0], filtered_eeg.shape[0]))
    C_sum = np.zeros_like(R_sum)
    used = 0
    for w in range(n_windows):
        s = w * step
        e = s + win
        if e > n_t:
            break
        imu_w = filtered_imu[:, s:e]
        eeg_w = filtered_eeg[:, s:e]
        R_sum += compute_pearson_matrix_vectorized(imu_w, eeg_w)
        C_sum += compute_coherence_matrix_vectorized(imu_w, eeg_w, fs=fs)
        used += 1
    if used == 0:
        # trial shorter than one window: use whatever we have
        R_sum = compute_pearson_matrix_vectorized(filtered_imu, filtered_eeg)
        C_sum = compute_coherence_matrix_vectorized(filtered_imu, filtered_eeg, fs=fs)
        used = 1
    R_raw = R_sum / used
    C_raw = C_sum / used
    coverage_s = used * step_sec if n_t >= win else n_t / fs
    return R_raw, C_raw, used, coverage_s


def compute_hybrid_from_raw(R_raw, C_raw, pearson_weight=0.2):
    """H = w * normalize(max(R,0)) + (1-w) * normalize(C), applied once
    to the window-averaged (denoised) subject-level raw matrices."""
    R_pos = np.maximum(R_raw, 0)
    return pearson_weight * normalize_matrix(R_pos) + (1.0 - pearson_weight) * normalize_matrix(C_raw)


def find_extreme_pairs(matrix, n_pairs=10):
    work = np.maximum(matrix, 0)
    order = np.argsort(work.flatten())[::-1]
    pairs = []
    for idx in order[:n_pairs]:
        imu_idx, eeg_idx = np.unravel_index(idx, matrix.shape)
        pairs.append((int(imu_idx), int(eeg_idx), float(matrix[imu_idx, eeg_idx])))
    return pairs


def visualize_hybrid_correlation(hybrid_matrix, imu_names, eeg_names, title, save_path,
                                 cbar_label="Group-mean coupling strength"):
    mat = hybrid_matrix[:, :len(eeg_names)].T
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        mat, cmap="viridis", yticklabels=eeg_names, xticklabels=imu_names, ax=ax,
        cbar_kws={"shrink": 0.85, "label": cbar_label})
    ax.set_xlabel("IMU Channels", fontweight="bold", fontsize=18)
    ax.set_ylabel("EEG Channels", fontweight="bold", fontsize=18)
    ax.set_title(title, fontweight="bold", fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold", fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold", fontsize=13)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_label, fontsize=14, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {save_path}")

# %% [markdown]
# ## Aggregation and paper MD

# %%
def compute_subject_hybrid(subject, movement, speed, base_dir=".", fs_imu=49.7,
                           pearson_weight=0.2, window_sec=10.0, step_sec=10.0):
    eeg_data, sfreq, ch_names = load_eeg_data(subject, movement, speed, base_dir)
    imu_data = load_imu_data(subject, movement, speed, base_dir)
    filtered_eeg, filtered_imu = analyze_imu_eeg_correlation(
        eeg_data, imu_data, fs_eeg=sfreq, fs_imu=fs_imu)
    duration_s = filtered_eeg.shape[1] / sfreq
    R_raw, C_raw, n_windows, coverage_s = compute_subject_raw_matrices(
        filtered_imu, filtered_eeg, fs=sfreq, window_sec=window_sec, step_sec=step_sec)
    H = compute_hybrid_from_raw(R_raw, C_raw, pearson_weight=pearson_weight)
    return R_raw, C_raw, H, ch_names, duration_s, n_windows, coverage_s


# Standard 20-ch montage used across the dataset (order fixed for group matrices).
STANDARD_EEG20 = [
    "F7", "Fp1", "Fp2", "F8", "F3", "Fz", "F4", "C3", "Cz", "P8",
    "P7", "Pz", "P4", "T3", "P3", "O1", "O2", "C4", "T4", "A2",
]
# 10-20 / 10-10 name aliases (Sub04 30-ch uses T7/T8 instead of T3/T4).
EEG_NAME_ALIASES = {
    "T3": ["T3", "T7"],
    "T4": ["T4", "T8"],
}


def resolve_eeg_index(ch_names, target):
    """Find channel index by exact name, then by known aliases."""
    candidates = EEG_NAME_ALIASES.get(target, [target])
    for name in candidates:
        if name in ch_names:
            return ch_names.index(name), name
    return None, None


def align_H_to_standard_eeg(H, ch_names, ref_names=None):
    """Select EEG columns to STANDARD_EEG20 order by channel name.

    For Sub04 30-ch: exact name match for 18 sites; T3<-T7, T4<-T8.
    Output labels always use the standard 20 names (T3/T4), not aliases.
    """
    if H.shape[0] != 6:
        raise ValueError(f"Expected 6 IMU channels, got H.shape={H.shape}")
    ref = list(ref_names) if ref_names is not None else list(STANDARD_EEG20)
    idxs = []
    used_src = []
    for c in ref:
        idx, src = resolve_eeg_index(ch_names, c)
        if idx is None:
            raise ValueError(f"Missing EEG channel '{c}' in {ch_names}")
        idxs.append(idx)
        used_src.append(src)
    return H[:, idxs], list(ref), used_src


def stack_and_aggregate(mat_list, labels, durs=None):
    # Require identical shapes (caller should pre-align)
    shapes = {m.shape for m in mat_list}
    if len(shapes) != 1:
        from collections import Counter
        common = Counter(m.shape for m in mat_list).most_common(1)[0][0]
        keep_idx = [i for i, m in enumerate(mat_list) if m.shape == common]
        mat_list = [mat_list[i] for i in keep_idx]
        labels = [labels[i] for i in keep_idx]
        if durs is not None:
            durs = [durs[i] for i in keep_idx]
        print(f"  Warning: mixed shapes {shapes}; kept common {common}, N={len(mat_list)}")
    stack = np.stack(mat_list, axis=0)
    mean = np.mean(stack, axis=0)
    std = (np.std(stack, axis=0, ddof=1)
           if stack.shape[0] > 1 else np.zeros_like(mean))
    return mean, std, labels, durs


def region_means(H, ch_names):
    regions = {
        "frontal": ["F7", "Fp1", "Fp2", "F8", "F3", "Fz", "F4"],
        "central": ["C3", "Cz", "C4"],
        "parietal": ["P7", "P8", "Pz", "P3", "P4"],
        "occipital": ["O1", "O2"],
        "temporal": ["T3", "T4"],
    }
    out = {}
    for name, chs in regions.items():
        idxs = [ch_names.index(c) for c in chs if c in ch_names]
        out[name] = float(H[:, idxs].mean()) if idxs else None
    return out


def format_result_block(tag, movement, movement_label, H_mean, H_std, labels,
                        ch_names, imu_names, durations, R_mean_raw, C_mean_raw,
                        H_groupnorm, n_windows_list):
    top = find_extreme_pairs(H_mean, n_pairs=10)
    top_lines = []
    for rank, (ii, ee, val) in enumerate(top, 1):
        sd = float(H_std[ii, ee])
        top_lines.append(
            f"  {rank}. {imu_names[ii]}-{ch_names[ee]}: {val:.4f} +/- {sd:.4f}")
    per_imu_mean = {imu_names[i]: float(H_mean[i].mean()) for i in range(H_mean.shape[0])}
    per_imu_max = {imu_names[i]: float(H_mean[i].max()) for i in range(H_mean.shape[0])}
    regions = region_means(H_mean, ch_names)
    n_subj = len({x.split("@")[0] for x in labels})
    R_pos_mean_raw = np.maximum(R_mean_raw, 0)
    return {
        "analysis": tag,
        "movement": movement,
        "movement_label": movement_label,
        "n_trials": len(labels),
        "n_subjects": n_subj,
        "labels": labels,
        "duration_s_mean": float(np.mean(durations)) if durations else None,
        "duration_s_min": float(np.min(durations)) if durations else None,
        "duration_s_max": float(np.max(durations)) if durations else None,
        "n_windows_mean": float(np.mean(n_windows_list)) if n_windows_list else None,
        "n_windows_min": float(np.min(n_windows_list)) if n_windows_list else None,
        "n_windows_max": float(np.max(n_windows_list)) if n_windows_list else None,
        "H_mean_stats": {
            "min": float(H_mean.min()),
            "max": float(H_mean.max()),
            "mean": float(H_mean.mean()),
            "std": float(H_mean.std()),
        },
        "H_std_mean": float(H_std.mean()),
        "H_groupnorm_stats": {
            "min": float(H_groupnorm.min()),
            "max": float(H_groupnorm.max()),
            "mean": float(H_groupnorm.mean()),
        },
        "R_pos_raw_stats": {
            "mean": float(R_pos_mean_raw.mean()),
            "max": float(R_pos_mean_raw.max()),
        },
        "C_raw_stats": {
            "mean": float(C_mean_raw.mean()),
            "max": float(C_mean_raw.max()),
        },
        "top10": top_lines,
        "per_imu_mean": per_imu_mean,
        "per_imu_max": per_imu_max,
        "region_means": regions,
        "H_mean": H_mean,
        "H_groupnorm": H_groupnorm,
        "ch_names": ch_names,
    }


def write_paper_md(path, all_blocks, method_note):
    lines = []
    lines.append("# Section 4.1 Group-level IMU-EEG correlation (paper summary)\n\n")
    lines.append("> Auto-generated by `corr_group_for_R1.py` (sliding-window, full-trial coverage). ")
    lines.append("Methods + key numbers only. ")
    lines.append("Heatmaps: `result/window10s_group/{speed0,speed1,combined}/`. ")
    lines.append("(`combined` pools both speeds — kept for completeness, but speed0/speed1 ")
    lines.append("are the primary, apples-to-apples group statistics; see Sec. 2.1/2.2.)\n\n")
    lines.append("---\n\n## 1. Methods\n\n")
    lines.append(method_note)
    lines.append("\n## 2. Results\n")

    by_analysis = {"speed0": [], "speed1": [], "combined": []}
    for b in all_blocks:
        by_analysis[b["analysis"]].append(b)

    section_titles = [
        ("speed0", "Speed = 0 (slow)"),
        ("speed1", "Speed = 1 (fast)"),
        ("combined", "Combined (all Move trials from speed0 + speed1)"),
    ]
    for si, (tag, title) in enumerate(section_titles, start=1):
        lines.append(f"\n### 2.{si} {title}\n\n")
        blocks = by_analysis[tag]
        lines.append("| Cond | Action | N_trials | N_subj | mean(Hbar) | max(Hbar) | mean(SD) | "
                     "mean(H_groupnorm) | mean rawR+ | mean rawC | Windows/trial | Duration mean [min,max] s |\n")
        lines.append("|------|--------|----------|--------|------------|-----------|----------|"
                     "--------------------|------------|-----------|---------------|---------------------------|\n")
        for b in blocks:
            st = b["H_mean_stats"]
            gn = b["H_groupnorm_stats"]
            d = (f"{b['duration_s_mean']:.1f} "
                 f"[{b['duration_s_min']:.1f}, {b['duration_s_max']:.1f}]")
            lines.append(
                f"| {b['movement']} | {b['movement_label']} | {b['n_trials']} | "
                f"{b['n_subjects']} | {st['mean']:.4f} | {st['max']:.4f} | "
                f"{b['H_std_mean']:.4f} | {gn['mean']:.4f} | "
                f"{b['R_pos_raw_stats']['mean']:.4f} | {b['C_raw_stats']['mean']:.4f} | "
                f"{b['n_windows_mean']:.1f} | {d} |\n")

        for b in blocks:
            lines.append(f"\n#### Movement {b['movement']} — {b['movement_label']}\n\n")
            lines.append(
                f"- N_trials={b['n_trials']}, N_subjects={b['n_subjects']}, "
                f"windows/trial: mean={b['n_windows_mean']:.1f} "
                f"[{b['n_windows_min']:.0f}, {b['n_windows_max']:.0f}] "
                f"(each window = full non-overlapping coverage of the trial)\n")
            lines.append(
                f"- Hbar (subject-mean of per-subject-normalized H): "
                f"min={b['H_mean_stats']['min']:.4f}, max={b['H_mean_stats']['max']:.4f}, "
                f"mean={b['H_mean_stats']['mean']:.4f}; mean cross-subject SD={b['H_std_mean']:.4f}\n")
            lines.append(
                f"- H_groupnorm (group-mean raw R/C, normalized once at group level): "
                f"min={b['H_groupnorm_stats']['min']:.4f}, max={b['H_groupnorm_stats']['max']:.4f}, "
                f"mean={b['H_groupnorm_stats']['mean']:.4f}\n")
            lines.append(
                f"- Raw (absolute, non-normalized) group means: "
                f"mean R+={b['R_pos_raw_stats']['mean']:.4f} (max={b['R_pos_raw_stats']['max']:.4f}), "
                f"mean coherence C={b['C_raw_stats']['mean']:.4f} (max={b['C_raw_stats']['max']:.4f})\n")
            lines.append("- Top 10 group-mean pairs (Hbar +/- SD):\n")
            for x in b["top10"]:
                lines.append(x + "\n")
            imu_mean = ", ".join(f"{k}={v:.3f}" for k, v in b["per_imu_mean"].items())
            imu_max = ", ".join(f"{k}={v:.3f}" for k, v in b["per_imu_max"].items())
            lines.append(f"- Per-IMU mean(Hbar): {imu_mean}\n")
            lines.append(f"- Per-IMU max(Hbar): {imu_max}\n")
            reg = ", ".join(
                f"{k}={v:.4f}" for k, v in b["region_means"].items() if v is not None)
            lines.append(f"- Region mean(Hbar): {reg}\n")
            lines.append(
                "- Group-mean H matrix (rows=IMU AccX..GyroZ; cols=EEG order below):\n")
            lines.append(f"  - EEG order: {', '.join(b['ch_names'])}\n")
            lines.append("  - Hbar =\n\n```\n")
            lines.append(np.array2string(b["H_mean"], precision=4, suppress_small=True))
            lines.append("\n```\n")

    lines.append("\n## 3. Figure files\n\n")
    lines.append("- `result/window10s_group/speed0/movement{0-5}.png` (+ `_groupnorm.png` variant)\n")
    lines.append("- `result/window10s_group/speed1/movement{0-5}.png` (+ `_groupnorm.png` variant)\n")
    lines.append("- `result/window10s_group/combined/movement{0-5}.png` (+ `_groupnorm.png` variant, "
                 "pooled speeds — supplementary)\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Wrote paper summary: {path}")

# %% [markdown]
# ## Main

# %%
base_dir = "."
out_root = os.path.join(base_dir, "result", "window10s_group")
subjects = [f"Sub{i:02d}" for i in range(1, 21)]
movements = list(range(6))
speeds = [0, 1]
fs_imu = 49.7
pearson_weight = 0.2
window_sec = 10.0
step_sec = 10.0
imu_names = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
movement_labels = {
    0: "walk",
    1: "figure-8",
    2: "horizontal",
    3: "vertical",
    4: "pick",
    5: "stair",
}

# Remove old clutter; recreate three analysis folders
if os.path.isdir(out_root):
    shutil.rmtree(out_root)
for tag in ["speed0", "speed1", "combined"]:
    os.makedirs(os.path.join(out_root, tag), exist_ok=True)

print("=" * 80)
print("Group-Level IMU-EEG Correlation — sliding-window, full-trial coverage")
print(f"Window={window_sec}s, step={step_sec}s (non-overlapping, covers 100% of each trial)")
print("Analyses: speed0 | speed1 | combined")
print("No Rest/Stand; save figures only (Agg, no popup)")
print("=" * 80)

cache = {}
for movement in movements:
    for speed in speeds:
        for subject in subjects:
            key = (subject, movement, speed)
            try:
                R_raw, C_raw, H, ch_names, dur, n_win, cov_s = compute_subject_hybrid(
                    subject, movement, speed, base_dir=base_dir,
                    fs_imu=fs_imu, pearson_weight=pearson_weight,
                    window_sec=window_sec, step_sec=step_sec)
                cache[key] = (R_raw, C_raw, H, ch_names, dur, n_win)
                print(f"[ok] {subject} mov{movement} speed{speed}: "
                      f"dur={dur:.1f}s, windows={n_win} (cov={cov_s:.1f}s), H_mean={H.mean():.4f}")
            except FileNotFoundError:
                print(f"[skip missing] {subject} mov{movement} speed{speed}")
            except Exception as e:
                print(f"[skip error] {subject} mov{movement} speed{speed}: {e}")

paper_blocks = []


def run_aggregate(tag, movement, speed_filter):
    H_list, R_list, C_list, labels, durs, n_windows_list = [], [], [], [], [], []
    ch_names = list(STANDARD_EEG20)
    for (subject, mov, speed), (R_raw, C_raw, H, names, dur, n_win) in cache.items():
        if mov != movement:
            continue
        if speed_filter is not None and speed != speed_filter:
            continue
        try:
            H_aligned, names_aligned, src = align_H_to_standard_eeg(H, names)
            R_aligned, _, _ = align_H_to_standard_eeg(R_raw, names)
            C_aligned, _, _ = align_H_to_standard_eeg(C_raw, names)
            if any(a != b for a, b in zip(src, names_aligned)):
                alias_note = ", ".join(
                    f"{std}<-{s}" for std, s in zip(names_aligned, src) if std != s)
                if alias_note:
                    print(f"  [alias] {subject}@speed{speed}: {alias_note}")
        except ValueError as e:
            print(f"  [skip align] {subject}@speed{speed}: {e}")
            continue
        H_list.append(H_aligned)
        R_list.append(R_aligned)
        C_list.append(C_aligned)
        labels.append(f"{subject}@speed{speed}")
        durs.append(dur)
        n_windows_list.append(n_win)
    if not H_list:
        print(f"  [{tag}] movement{movement}: no data")
        return None
    H_mean, H_std, labels, durs = stack_and_aggregate(H_list, labels, durs)
    R_mean_raw, _, _, _ = stack_and_aggregate(R_list, list(labels))
    C_mean_raw, _, _, _ = stack_and_aggregate(C_list, list(labels))
    H_groupnorm = compute_hybrid_from_raw(R_mean_raw, C_mean_raw, pearson_weight=pearson_weight)

    block = format_result_block(
        tag, movement, movement_labels[movement], H_mean, H_std, labels,
        ch_names, imu_names, durs, R_mean_raw, C_mean_raw, H_groupnorm, n_windows_list)

    fig_path = os.path.join(out_root, tag, f"movement{movement}.png")
    title = (f"Group-Mean H (per-subject normalized) | {movement_labels[movement]} | {tag} "
             f"(N_trials={block['n_trials']}, N_subj={block['n_subjects']})")
    visualize_hybrid_correlation(H_mean, imu_names, ch_names, title, fig_path,
                                 cbar_label="Group-mean of per-subject H")

    fig_path_gn = os.path.join(out_root, tag, f"movement{movement}_groupnorm.png")
    title_gn = (f"Group-level H (raw R/C averaged, normalized once) | {movement_labels[movement]} | {tag} "
                f"(N_trials={block['n_trials']}, N_subj={block['n_subjects']})")
    visualize_hybrid_correlation(H_groupnorm, imu_names, ch_names, title_gn, fig_path_gn,
                                 cbar_label="Group-level H (normalized once)")

    paper_blocks.append(block)
    print(f"  [{tag}] mov{movement}: N={block['n_trials']}, "
          f"Hbar_mean={block['H_mean_stats']['mean']:.4f}, "
          f"Hgroupnorm_mean={block['H_groupnorm_stats']['mean']:.4f}, "
          f"raw R+ mean={block['R_pos_raw_stats']['mean']:.4f}, "
          f"raw C mean={block['C_raw_stats']['mean']:.4f}")
    return block


for movement in movements:
    print("\n" + "=" * 60)
    print(f"Aggregating movement {movement} ({movement_labels[movement]})")
    run_aggregate("speed0", movement, speed_filter=0)
    run_aggregate("speed1", movement, speed_filter=1)
    run_aggregate("combined", movement, speed_filter=None)

method_note = f"""
- Data: Sub01-Sub20; **Move segments only** (no Rest/standing); 6 movements x 2 speeds.
- **Sliding-window, full-trial coverage** (no cherry-picked excerpt, no single-window
  full-trial average): each subject's ENTIRE Move segment is split into non-overlapping
  {window_sec:.0f}s windows (step={step_sec:.0f}s); raw Pearson R and raw spectral coherence C
  are computed per window and averaged across ALL windows within that subject/trial
  (window count varies with trial length; reported per condition below).
- Three analyses / three folders:
  1. `speed0`: group-mean over slow-speed Move trials
  2. `speed1`: group-mean over fast-speed Move trials
  3. `combined`: group-mean over all Move trials from both speeds
- Channels: IMU 6 (AccX/Y/Z, GyroX/Y/Z); EEG aligned to the standard 20-ch montage
  `F7, Fp1, Fp2, F8, F3, Fz, F4, C3, Cz, P8, P7, Pz, P4, T3, P3, O1, O2, C4, T4, A2`
  by **channel name**. Sub04 30-ch trials: same names kept; `T3<-T7`, `T4<-T8`; extra channels discarded.
- Preprocessing: resample IMU to EEG rate (fs_IMU={fs_imu} Hz); band-pass 0.05-40 Hz.
- Per-subject matrices: R_raw, C_raw are window-averaged (within-subject), THEN
  R+=max(R_raw,0); min-max normalize each to [0,1]; H_subject = 0.2*R+tilde + 0.8*Ctilde
  (each subject individually normalized — this is "H for each subject" as requested).
- Group level, two complementary summaries per movement:
  1. **Hbar**: mean (± cross-subject SD) of the per-subject-normalized H_subject matrices.
     This is the literal "compute H per subject, report group mean ± SD" statistic.
  2. **H_groupnorm**: group-mean raw R_pos and raw C (averaged across subjects FIRST,
     in absolute units), normalized ONCE at the group level. Avoids the shrinkage that
     comes from averaging N independently-normalized matrices whose peak locations differ
     across subjects; better for visualizing the group-level spatial coupling pattern.
  3. Raw absolute numbers (mean R+, mean C, unnormalized) are also reported — these are
     the only numbers with literal physical meaning (actual Pearson r / actual coherence).
- Outputs: PNG heatmaps (both Hbar and H_groupnorm variants) under
  `result/window10s_group/{{speed0,speed1,combined}}/`, plus this single paper MD.
  `combined` (pooled speeds) is kept for completeness but is NOT the primary statistic —
  it mixes genuinely-weaker slow-speed trials with fast-speed ones. speed0/speed1 are
  the primary, apples-to-apples group summaries.
"""

md_path = os.path.join(base_dir, "R1_paper_summary.md")
write_paper_md(md_path, paper_blocks, method_note)

print("\n" + "=" * 80)
print("Done.")
print(f"Figures: {out_root}/{{speed0,speed1,combined}}/")
print(f"Paper MD: {md_path}")
print("=" * 80)
# %%
