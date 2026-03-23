# %%
import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

# %% [markdown]
# ## Data Loading

# %%
def load_eeg_data(subject, movement, speed, base_dir="."):
    eeg_file = os.path.join(base_dir, "EEG_Data", subject, "Raw_mat_segmented",
                           f"{subject}_movement{movement}_speed{speed}_Move.mat")

    if not os.path.exists(eeg_file):
        raise FileNotFoundError(f"EEG file not found: {eeg_file}")

    data = sio.loadmat(eeg_file, squeeze_me=True, struct_as_record=False)
    eeg = data['EEG']
    eeg_data = np.array(eeg.data)
    sfreq = float(eeg.srate)
    ch_names = [str(ch.labels) for ch in eeg.chanlocs]

    print(f"Loaded EEG: {eeg_file}")
    print(f"  Shape: {eeg_data.shape}, sr={sfreq} Hz, channels={len(ch_names)}")

    return eeg_data, sfreq, ch_names


def load_imu_data(subject, movement, speed, base_dir="."):
    acc_file = os.path.join(base_dir, "IMU_Data", subject, "Raw_mat_segmented",
                           f"{subject}_movement{movement}_speed{speed}_ACC_Move.mat")
    gyro_file = os.path.join(base_dir, "IMU_Data", subject, "Raw_mat_segmented",
                            f"{subject}_movement{movement}_speed{speed}_GYRO_Move.mat")

    if not os.path.exists(acc_file):
        raise FileNotFoundError(f"ACC file not found: {acc_file}")
    if not os.path.exists(gyro_file):
        raise FileNotFoundError(f"GYRO file not found: {gyro_file}")

    acc_data = sio.loadmat(acc_file, squeeze_me=True)
    acc = acc_data['acc_data']
    gyro_data = sio.loadmat(gyro_file, squeeze_me=True)
    gyro = gyro_data['gyro_data']
    acc_t = acc.T
    gyro_t = gyro.T
    imu_data = np.vstack([acc_t, gyro_t])

    print(f"Loaded IMU: {acc_file}, {gyro_file}")
    print(f"  Shape: {imu_data.shape} (ACC: {acc_t.shape}, GYRO: {gyro_t.shape})")

    return imu_data

# %% [markdown]
# ## Signal Processing

# %%
def bandpass_filter(signal_data, fs, lowcut=0.05, highcut=40, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal_data)
    return filtered_signal


def resample_imu_to_eeg(imu_data, fs_imu=50, fs_eeg=500):
    num_channels = imu_data.shape[0]
    num_samples_orig = imu_data.shape[1]
    num_samples_new = int(num_samples_orig * fs_eeg / fs_imu)
    resampled_data = np.zeros((num_channels, num_samples_new))
    for i in range(num_channels):
        resampled_data[i] = signal.resample(imu_data[i], num_samples_new)
    return resampled_data


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
# ## Correlation

# %%
def compute_spectral_coherence_scipy(x, y, fs=500, freq_band=(0.05, 40)):
    f, Cxy = signal.coherence(x, y, fs=fs, nperseg=min(256, len(x)))
    band_idx = (f >= freq_band[0]) & (f <= freq_band[1])
    if np.any(band_idx):
        mean_coherence = np.mean(Cxy[band_idx])
    else:
        mean_coherence = 0.0
    return mean_coherence


def compute_fft_coherence_matrix(filtered_imu, filtered_eeg, fs=500, duration_seconds=None):
    n_imu_channels = filtered_imu.shape[0]
    n_eeg_channels = filtered_eeg.shape[0]

    if duration_seconds is not None:
        n_samples = int(duration_seconds * fs)
        n_samples = min(n_samples, filtered_imu.shape[1], filtered_eeg.shape[1])
        filtered_imu = filtered_imu[:, :n_samples]
        filtered_eeg = filtered_eeg[:, :n_samples]
        print(f"Using first {duration_seconds}s for FFT coherence ({n_samples} samples)")

    coherence_matrix = np.zeros((n_imu_channels, n_eeg_channels))
    for i in range(n_imu_channels):
        for j in range(n_eeg_channels):
            coherence_matrix[i, j] = compute_spectral_coherence_scipy(
                filtered_imu[i], filtered_eeg[j], fs=fs)

    print(f"FFT coherence min: {np.min(coherence_matrix):.4f}, max: {np.max(coherence_matrix):.4f}")
    print(f"FFT coherence mean: {np.mean(coherence_matrix):.4f}, std: {np.std(coherence_matrix):.4f}")

    if np.std(coherence_matrix) < 0.05:
        print("Warning: low FFT coherence variance, applying contrast stretch")
        if np.max(coherence_matrix) > np.min(coherence_matrix):
            coherence_matrix = (coherence_matrix - np.min(coherence_matrix)) / (np.max(coherence_matrix) - np.min(coherence_matrix))

    return coherence_matrix


def compute_pearson_correlation(filtered_imu, filtered_eeg, fs=500, duration_seconds=None):
    n_imu_channels = filtered_imu.shape[0]
    n_eeg_channels = filtered_eeg.shape[0]

    if duration_seconds is not None:
        n_samples = int(duration_seconds * fs)
        n_samples = min(n_samples, filtered_imu.shape[1], filtered_eeg.shape[1])
        filtered_imu = filtered_imu[:, :n_samples]
        filtered_eeg = filtered_eeg[:, :n_samples]
        print(f"Using first {duration_seconds}s for Pearson ({n_samples} samples)")

    correlation_matrix = np.zeros((n_imu_channels, n_eeg_channels))
    for i in range(n_imu_channels):
        for j in range(n_eeg_channels):
            correlation_matrix[i, j] = np.corrcoef(filtered_imu[i], filtered_eeg[j])[0, 1]

    return correlation_matrix

# %% [markdown]
# ## Visualization

# %%
def normalize_matrix(matrix):
    if np.max(matrix) == np.min(matrix):
        return np.zeros_like(matrix)
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))


def visualize_hybrid_correlation(hybrid_matrix, imu_names=None, eeg_names=None):
    if imu_names is None:
        imu_names = [f'IMU {i}' for i in range(hybrid_matrix.shape[0])]
    max_eeg_ch = len(eeg_names) if eeg_names is not None else min(32, hybrid_matrix.shape[1])
    hybrid_matrix_subset = hybrid_matrix[:, :max_eeg_ch]
    hybrid_matrix_transposed = hybrid_matrix_subset.T

    plt.figure(figsize=(9, 8))
    if eeg_names is not None:
        yticklabels = eeg_names[:max_eeg_ch]
    else:
        yticklabels = [f'EEG {i}' for i in range(max_eeg_ch)]

    sns.heatmap(hybrid_matrix_transposed,
                cmap='viridis',
                yticklabels=yticklabels,
                xticklabels=imu_names)
    plt.xlabel('IMU Channels', fontweight='bold')
    plt.ylabel('EEG Channels', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.tight_layout()
    plt.show()


def find_extreme_pairs(matrix, imu_names=None, n_pairs=30, preference='positive'):
    if preference == 'positive':
        matrix_work = np.copy(matrix)
        matrix_work[matrix_work < 0] = 0
    else:
        matrix_work = np.abs(matrix)

    flat_indices = np.argsort(matrix_work.flatten())[::-1]
    flat_matrix = matrix_work.flatten()

    top_pairs = []
    for idx in flat_indices[:n_pairs]:
        imu_idx, eeg_idx = np.unravel_index(idx, matrix.shape)
        value = matrix[imu_idx, eeg_idx]
        top_pairs.append((imu_idx, eeg_idx, value))

    bottom_pairs = []
    for idx in flat_indices[-n_pairs:]:
        imu_idx, eeg_idx = np.unravel_index(idx, matrix.shape)
        value = matrix[imu_idx, eeg_idx]
        bottom_pairs.append((imu_idx, eeg_idx, value))

    return top_pairs, bottom_pairs


def visualize_extreme_pairs(filtered_imu, filtered_eeg, pairs, imu_names, eeg_names, title_prefix,
                            duration_seconds=2, fs=500):
    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(12, 4*n_pairs))
    if n_pairs == 1:
        axes = [axes]

    n_samples_to_show = int(duration_seconds * fs)

    for i, (imu_idx, eeg_idx, value) in enumerate(pairs):
        imu_signal = filtered_imu[imu_idx]
        eeg_signal = filtered_eeg[eeg_idx]
        n_samples_actual = min(n_samples_to_show, len(imu_signal), len(eeg_signal))
        imu_signal = imu_signal[:n_samples_actual]
        eeg_signal = eeg_signal[:n_samples_actual]
        imu_norm = (imu_signal - np.mean(imu_signal)) / np.std(imu_signal)
        eeg_norm = (eeg_signal - np.mean(eeg_signal)) / np.std(eeg_signal)
        time = np.arange(n_samples_actual) / fs

        axes[i].plot(time, imu_norm, 'b-', label='IMU')
        axes[i].plot(time, eeg_norm, 'r-', label='EEG')
        axes[i].set_title(f'{title_prefix} - {imu_names[imu_idx]} vs {eeg_names[eeg_idx]} (corr={value:.4f})')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Normalized Amplitude')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, min(duration_seconds, n_samples_actual / fs))

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Main

# %%
subject = "Sub15"
movement = 5
speed = 1
base_dir = "."
duration_seconds = 10
n_visualize_pairs = 2
fs_imu = 49.7
visualize_bottom_pairs = False

imu_names = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']

print("=" * 80)
print(f"Loading: Subject={subject}, Movement={movement}, Speed={speed}")
print("=" * 80)

eeg_data, sfreq, ch_names = load_eeg_data(subject, movement, speed, base_dir)
imu_data = load_imu_data(subject, movement, speed, base_dir)

print("\n" + "=" * 80)
print("IMU-EEG Correlation Analysis")
print("=" * 80)

filtered_eeg, filtered_imu = analyze_imu_eeg_correlation(
    eeg_data, imu_data, fs_eeg=sfreq, fs_imu=fs_imu)

print(f"\nFiltered shapes: EEG {filtered_eeg.shape}, IMU {filtered_imu.shape}")

if isinstance(duration_seconds, (int, float)):
    duration_list = [duration_seconds]
else:
    duration_list = duration_seconds

for duration in duration_list:
    print("\n" + "=" * 80)
    print(f"Processing: first {duration}s")
    print("=" * 80)

    print(f"\nComputing correlation matrix ({duration}s)...")
    print("FFT coherence...")
    fft_matrix = compute_fft_coherence_matrix(filtered_imu, filtered_eeg, fs=sfreq, duration_seconds=duration)
    norm_fft = normalize_matrix(fft_matrix)

    print("Pearson...")
    pearson_matrix = compute_pearson_correlation(filtered_imu, filtered_eeg, fs=sfreq, duration_seconds=duration)
    norm_pearson = normalize_matrix(pearson_matrix)

    pearson_weight = 0.2
    positive_pearson = np.copy(pearson_matrix)
    positive_pearson[positive_pearson < 0] = 0
    norm_positive_pearson = normalize_matrix(positive_pearson)
    hybrid_fft_matrix = pearson_weight * norm_positive_pearson + (1 - pearson_weight) * norm_fft

    print("Visualizing hybrid matrix...")
    visualize_hybrid_correlation(hybrid_fft_matrix, imu_names, ch_names)

    print("Finding extreme pairs...")
    top_pairs, bottom_pairs = find_extreme_pairs(hybrid_fft_matrix, imu_names, n_pairs=30, preference='positive')
    top_pairs_filtered = [(imu_idx, eeg_idx, value) for imu_idx, eeg_idx, value in top_pairs if eeg_idx < len(ch_names)]
    if visualize_bottom_pairs:
        bottom_pairs_filtered = [(imu_idx, eeg_idx, value) for imu_idx, eeg_idx, value in bottom_pairs if eeg_idx < len(ch_names)]

    print("\n" + "=" * 80)
    print(f"Top pairs (FFT hybrid, {duration}s):")
    print("=" * 80)
    for i, (imu_idx, eeg_idx, value) in enumerate(top_pairs_filtered[:10]):
        print(f"{i+1}. {imu_names[imu_idx]} - {ch_names[eeg_idx]}: {value:.4f}")

    if visualize_bottom_pairs:
        print("\n" + "=" * 80)
        print(f"Least correlated pairs ({duration}s):")
        print("=" * 80)
        for i, (imu_idx, eeg_idx, value) in enumerate(bottom_pairs_filtered[:10]):
            print(f"{i+1}. {imu_names[imu_idx]} - {ch_names[eeg_idx]}: {value:.4f}")

    if top_pairs_filtered:
        print(f"\nVisualizing top {min(n_visualize_pairs, len(top_pairs_filtered))} pairs...")
        visualize_extreme_pairs(filtered_imu, filtered_eeg,
                               top_pairs_filtered[:n_visualize_pairs],
                               imu_names, ch_names, f"Most Correlated ({duration}s)",
                               duration_seconds=duration, fs=sfreq)

    if visualize_bottom_pairs and bottom_pairs_filtered:
        print(f"\nVisualizing bottom {min(n_visualize_pairs, len(bottom_pairs_filtered))} pairs...")
        visualize_extreme_pairs(filtered_imu, filtered_eeg,
                               bottom_pairs_filtered[:n_visualize_pairs],
                               imu_names, ch_names, f"Least Correlated ({duration}s)",
                               duration_seconds=duration, fs=sfreq)

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)

# %%
subject = "Sub06"
movement = 4
speed = 1
base_dir = "."
duration_seconds = 10
n_visualize_pairs = 5
fs_imu = 49.7
visualize_bottom_pairs = False

imu_names = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']

print("=" * 80)
print(f"Loading: Subject={subject}, Movement={movement}, Speed={speed}")
print("=" * 80)

eeg_data, sfreq, ch_names = load_eeg_data(subject, movement, speed, base_dir)
imu_data = load_imu_data(subject, movement, speed, base_dir)

print("\n" + "=" * 80)
print("IMU-EEG Correlation Analysis")
print("=" * 80)

filtered_eeg, filtered_imu = analyze_imu_eeg_correlation(
    eeg_data, imu_data, fs_eeg=sfreq, fs_imu=fs_imu)

print(f"\nFiltered shapes: EEG {filtered_eeg.shape}, IMU {filtered_imu.shape}")

if isinstance(duration_seconds, (int, float)):
    duration_list = [duration_seconds]
else:
    duration_list = duration_seconds

for duration in duration_list:
    print("\n" + "=" * 80)
    print(f"Processing: first {duration}s")
    print("=" * 80)

    print(f"\nComputing correlation matrix ({duration}s)...")
    print("FFT coherence...")
    fft_matrix = compute_fft_coherence_matrix(filtered_imu, filtered_eeg, fs=sfreq, duration_seconds=duration)
    norm_fft = normalize_matrix(fft_matrix)

    print("Pearson...")
    pearson_matrix = compute_pearson_correlation(filtered_imu, filtered_eeg, fs=sfreq, duration_seconds=duration)
    norm_pearson = normalize_matrix(pearson_matrix)

    pearson_weight = 0.2
    positive_pearson = np.copy(pearson_matrix)
    positive_pearson[positive_pearson < 0] = 0
    norm_positive_pearson = normalize_matrix(positive_pearson)
    hybrid_fft_matrix = pearson_weight * norm_positive_pearson + (1 - pearson_weight) * norm_fft

    print("Visualizing hybrid matrix...")
    visualize_hybrid_correlation(hybrid_fft_matrix, imu_names, ch_names)

    print("Finding extreme pairs...")
    top_pairs, bottom_pairs = find_extreme_pairs(hybrid_fft_matrix, imu_names, n_pairs=30, preference='positive')
    top_pairs_filtered = [(imu_idx, eeg_idx, value) for imu_idx, eeg_idx, value in top_pairs if eeg_idx < len(ch_names)]
    if visualize_bottom_pairs:
        bottom_pairs_filtered = [(imu_idx, eeg_idx, value) for imu_idx, eeg_idx, value in bottom_pairs if eeg_idx < len(ch_names)]

    print("\n" + "=" * 80)
    print(f"Top pairs (FFT hybrid, {duration}s):")
    print("=" * 80)
    for i, (imu_idx, eeg_idx, value) in enumerate(top_pairs_filtered[:10]):
        print(f"{i+1}. {imu_names[imu_idx]} - {ch_names[eeg_idx]}: {value:.4f}")

    if visualize_bottom_pairs:
        print("\n" + "=" * 80)
        print(f"Least correlated pairs ({duration}s):")
        print("=" * 80)
        for i, (imu_idx, eeg_idx, value) in enumerate(bottom_pairs_filtered[:10]):
            print(f"{i+1}. {imu_names[imu_idx]} - {ch_names[eeg_idx]}: {value:.4f}")

    if top_pairs_filtered:
        print(f"\nVisualizing top {min(n_visualize_pairs, len(top_pairs_filtered))} pairs...")
        visualize_extreme_pairs(filtered_imu, filtered_eeg,
                               top_pairs_filtered[:n_visualize_pairs],
                               imu_names, ch_names, f"Most Correlated ({duration}s)",
                               duration_seconds=duration, fs=sfreq)

    if visualize_bottom_pairs and bottom_pairs_filtered:
        print(f"\nVisualizing bottom {min(n_visualize_pairs, len(bottom_pairs_filtered))} pairs...")
        visualize_extreme_pairs(filtered_imu, filtered_eeg,
                               bottom_pairs_filtered[:n_visualize_pairs],
                               imu_names, ch_names, f"Least Correlated ({duration}s)",
                               duration_seconds=duration, fs=sfreq)

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)

# %%
subject = "Sub19"
movement = 3
speed = 1
base_dir = "."
duration_seconds = 10
n_visualize_pairs = 5
fs_imu = 49.7
visualize_bottom_pairs = False

imu_names = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']

print("=" * 80)
print(f"Loading: Subject={subject}, Movement={movement}, Speed={speed}")
print("=" * 80)

eeg_data, sfreq, ch_names = load_eeg_data(subject, movement, speed, base_dir)
imu_data = load_imu_data(subject, movement, speed, base_dir)

print("\n" + "=" * 80)
print("IMU-EEG Correlation Analysis")
print("=" * 80)

filtered_eeg, filtered_imu = analyze_imu_eeg_correlation(
    eeg_data, imu_data, fs_eeg=sfreq, fs_imu=fs_imu)

print(f"\nFiltered shapes: EEG {filtered_eeg.shape}, IMU {filtered_imu.shape}")

if isinstance(duration_seconds, (int, float)):
    duration_list = [duration_seconds]
else:
    duration_list = duration_seconds

for duration in duration_list:
    print("\n" + "=" * 80)
    print(f"Processing: first {duration}s")
    print("=" * 80)

    print(f"\nComputing correlation matrix ({duration}s)...")
    print("FFT coherence...")
    fft_matrix = compute_fft_coherence_matrix(filtered_imu, filtered_eeg, fs=sfreq, duration_seconds=duration)
    norm_fft = normalize_matrix(fft_matrix)

    print("Pearson...")
    pearson_matrix = compute_pearson_correlation(filtered_imu, filtered_eeg, fs=sfreq, duration_seconds=duration)
    norm_pearson = normalize_matrix(pearson_matrix)

    pearson_weight = 0.2
    positive_pearson = np.copy(pearson_matrix)
    positive_pearson[positive_pearson < 0] = 0
    norm_positive_pearson = normalize_matrix(positive_pearson)
    hybrid_fft_matrix = pearson_weight * norm_positive_pearson + (1 - pearson_weight) * norm_fft

    print("Visualizing hybrid matrix...")
    visualize_hybrid_correlation(hybrid_fft_matrix, imu_names, ch_names)

    print("Finding extreme pairs...")
    top_pairs, bottom_pairs = find_extreme_pairs(hybrid_fft_matrix, imu_names, n_pairs=30, preference='positive')
    top_pairs_filtered = [(imu_idx, eeg_idx, value) for imu_idx, eeg_idx, value in top_pairs if eeg_idx < len(ch_names)]
    if visualize_bottom_pairs:
        bottom_pairs_filtered = [(imu_idx, eeg_idx, value) for imu_idx, eeg_idx, value in bottom_pairs if eeg_idx < len(ch_names)]

    print("\n" + "=" * 80)
    print(f"Top pairs (FFT hybrid, {duration}s):")
    print("=" * 80)
    for i, (imu_idx, eeg_idx, value) in enumerate(top_pairs_filtered[:10]):
        print(f"{i+1}. {imu_names[imu_idx]} - {ch_names[eeg_idx]}: {value:.4f}")

    if visualize_bottom_pairs:
        print("\n" + "=" * 80)
        print(f"Least correlated pairs ({duration}s):")
        print("=" * 80)
        for i, (imu_idx, eeg_idx, value) in enumerate(bottom_pairs_filtered[:10]):
            print(f"{i+1}. {imu_names[imu_idx]} - {ch_names[eeg_idx]}: {value:.4f}")

    if top_pairs_filtered:
        print(f"\nVisualizing top {min(n_visualize_pairs, len(top_pairs_filtered))} pairs...")
        visualize_extreme_pairs(filtered_imu, filtered_eeg,
                               top_pairs_filtered[:n_visualize_pairs],
                               imu_names, ch_names, f"Most Correlated ({duration}s)",
                               duration_seconds=duration, fs=sfreq)

    if visualize_bottom_pairs and bottom_pairs_filtered:
        print(f"\nVisualizing bottom {min(n_visualize_pairs, len(bottom_pairs_filtered))} pairs...")
        visualize_extreme_pairs(filtered_imu, filtered_eeg,
                               bottom_pairs_filtered[:n_visualize_pairs],
                               imu_names, ch_names, f"Least Correlated ({duration}s)",
                               duration_seconds=duration, fs=sfreq)

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)

# %%
subject = "Sub10"
movement = 2
speed = 1
base_dir = "."
duration_seconds = 10
n_visualize_pairs = 5
fs_imu = 49.7
visualize_bottom_pairs = False

imu_names = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']

print("=" * 80)
print(f"Loading: Subject={subject}, Movement={movement}, Speed={speed}")
print("=" * 80)

eeg_data, sfreq, ch_names = load_eeg_data(subject, movement, speed, base_dir)
imu_data = load_imu_data(subject, movement, speed, base_dir)

print("\n" + "=" * 80)
print("IMU-EEG Correlation Analysis")
print("=" * 80)

filtered_eeg, filtered_imu = analyze_imu_eeg_correlation(
    eeg_data, imu_data, fs_eeg=sfreq, fs_imu=fs_imu)

print(f"\nFiltered shapes: EEG {filtered_eeg.shape}, IMU {filtered_imu.shape}")

if isinstance(duration_seconds, (int, float)):
    duration_list = [duration_seconds]
else:
    duration_list = duration_seconds

for duration in duration_list:
    print("\n" + "=" * 80)
    print(f"Processing: first {duration}s")
    print("=" * 80)

    print(f"\nComputing correlation matrix ({duration}s)...")
    print("FFT coherence...")
    fft_matrix = compute_fft_coherence_matrix(filtered_imu, filtered_eeg, fs=sfreq, duration_seconds=duration)
    norm_fft = normalize_matrix(fft_matrix)

    print("Pearson...")
    pearson_matrix = compute_pearson_correlation(filtered_imu, filtered_eeg, fs=sfreq, duration_seconds=duration)
    norm_pearson = normalize_matrix(pearson_matrix)

    pearson_weight = 0.2
    positive_pearson = np.copy(pearson_matrix)
    positive_pearson[positive_pearson < 0] = 0
    norm_positive_pearson = normalize_matrix(positive_pearson)
    hybrid_fft_matrix = pearson_weight * norm_positive_pearson + (1 - pearson_weight) * norm_fft

    print("Visualizing hybrid matrix...")
    visualize_hybrid_correlation(hybrid_fft_matrix, imu_names, ch_names)

    print("Finding extreme pairs...")
    top_pairs, bottom_pairs = find_extreme_pairs(hybrid_fft_matrix, imu_names, n_pairs=30, preference='positive')
    top_pairs_filtered = [(imu_idx, eeg_idx, value) for imu_idx, eeg_idx, value in top_pairs if eeg_idx < len(ch_names)]
    if visualize_bottom_pairs:
        bottom_pairs_filtered = [(imu_idx, eeg_idx, value) for imu_idx, eeg_idx, value in bottom_pairs if eeg_idx < len(ch_names)]

    print("\n" + "=" * 80)
    print(f"Top pairs (FFT hybrid, {duration}s):")
    print("=" * 80)
    for i, (imu_idx, eeg_idx, value) in enumerate(top_pairs_filtered[:10]):
        print(f"{i+1}. {imu_names[imu_idx]} - {ch_names[eeg_idx]}: {value:.4f}")

    if visualize_bottom_pairs:
        print("\n" + "=" * 80)
        print(f"Least correlated pairs ({duration}s):")
        print("=" * 80)
        for i, (imu_idx, eeg_idx, value) in enumerate(bottom_pairs_filtered[:10]):
            print(f"{i+1}. {imu_names[imu_idx]} - {ch_names[eeg_idx]}: {value:.4f}")

    if top_pairs_filtered:
        print(f"\nVisualizing top {min(n_visualize_pairs, len(top_pairs_filtered))} pairs...")
        visualize_extreme_pairs(filtered_imu, filtered_eeg,
                               top_pairs_filtered[:n_visualize_pairs],
                               imu_names, ch_names, f"Most Correlated ({duration}s)",
                               duration_seconds=duration, fs=sfreq)

    if visualize_bottom_pairs and bottom_pairs_filtered:
        print(f"\nVisualizing bottom {min(n_visualize_pairs, len(bottom_pairs_filtered))} pairs...")
        visualize_extreme_pairs(filtered_imu, filtered_eeg,
                               bottom_pairs_filtered[:n_visualize_pairs],
                               imu_names, ch_names, f"Least Correlated ({duration}s)",
                               duration_seconds=duration, fs=sfreq)

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)

# %%
subject = "Sub10"
movement = 1
speed = 1
base_dir = "."
duration_seconds = 10
n_visualize_pairs = 5
fs_imu = 49.7
visualize_bottom_pairs = False

imu_names = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']

print("=" * 80)
print(f"Loading: Subject={subject}, Movement={movement}, Speed={speed}")
print("=" * 80)

eeg_data, sfreq, ch_names = load_eeg_data(subject, movement, speed, base_dir)
imu_data = load_imu_data(subject, movement, speed, base_dir)

print("\n" + "=" * 80)
print("IMU-EEG Correlation Analysis")
print("=" * 80)

filtered_eeg, filtered_imu = analyze_imu_eeg_correlation(
    eeg_data, imu_data, fs_eeg=sfreq, fs_imu=fs_imu)

print(f"\nFiltered shapes: EEG {filtered_eeg.shape}, IMU {filtered_imu.shape}")

if isinstance(duration_seconds, (int, float)):
    duration_list = [duration_seconds]
else:
    duration_list = duration_seconds

for duration in duration_list:
    print("\n" + "=" * 80)
    print(f"Processing: first {duration}s")
    print("=" * 80)

    print(f"\nComputing correlation matrix ({duration}s)...")
    print("FFT coherence...")
    fft_matrix = compute_fft_coherence_matrix(filtered_imu, filtered_eeg, fs=sfreq, duration_seconds=duration)
    norm_fft = normalize_matrix(fft_matrix)

    print("Pearson...")
    pearson_matrix = compute_pearson_correlation(filtered_imu, filtered_eeg, fs=sfreq, duration_seconds=duration)
    norm_pearson = normalize_matrix(pearson_matrix)

    pearson_weight = 0.2
    positive_pearson = np.copy(pearson_matrix)
    positive_pearson[positive_pearson < 0] = 0
    norm_positive_pearson = normalize_matrix(positive_pearson)
    hybrid_fft_matrix = pearson_weight * norm_positive_pearson + (1 - pearson_weight) * norm_fft

    print("Visualizing hybrid matrix...")
    visualize_hybrid_correlation(hybrid_fft_matrix, imu_names, ch_names)

    print("Finding extreme pairs...")
    top_pairs, bottom_pairs = find_extreme_pairs(hybrid_fft_matrix, imu_names, n_pairs=30, preference='positive')
    top_pairs_filtered = [(imu_idx, eeg_idx, value) for imu_idx, eeg_idx, value in top_pairs if eeg_idx < len(ch_names)]
    if visualize_bottom_pairs:
        bottom_pairs_filtered = [(imu_idx, eeg_idx, value) for imu_idx, eeg_idx, value in bottom_pairs if eeg_idx < len(ch_names)]

    print("\n" + "=" * 80)
    print(f"Top pairs (FFT hybrid, {duration}s):")
    print("=" * 80)
    for i, (imu_idx, eeg_idx, value) in enumerate(top_pairs_filtered[:10]):
        print(f"{i+1}. {imu_names[imu_idx]} - {ch_names[eeg_idx]}: {value:.4f}")

    if visualize_bottom_pairs:
        print("\n" + "=" * 80)
        print(f"Least correlated pairs ({duration}s):")
        print("=" * 80)
        for i, (imu_idx, eeg_idx, value) in enumerate(bottom_pairs_filtered[:10]):
            print(f"{i+1}. {imu_names[imu_idx]} - {ch_names[eeg_idx]}: {value:.4f}")

    if top_pairs_filtered:
        print(f"\nVisualizing top {min(n_visualize_pairs, len(top_pairs_filtered))} pairs...")
        visualize_extreme_pairs(filtered_imu, filtered_eeg,
                               top_pairs_filtered[:n_visualize_pairs],
                               imu_names, ch_names, f"Most Correlated ({duration}s)",
                               duration_seconds=duration, fs=sfreq)

    if visualize_bottom_pairs and bottom_pairs_filtered:
        print(f"\nVisualizing bottom {min(n_visualize_pairs, len(bottom_pairs_filtered))} pairs...")
        visualize_extreme_pairs(filtered_imu, filtered_eeg,
                               bottom_pairs_filtered[:n_visualize_pairs],
                               imu_names, ch_names, f"Least Correlated ({duration}s)",
                               duration_seconds=duration, fs=sfreq)

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)

# %%
subject = "Sub10"
movement = 0
speed = 1
base_dir = "."
duration_seconds = 10
n_visualize_pairs = 5
fs_imu = 50
visualize_bottom_pairs = False

imu_names = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']

print("=" * 80)
print(f"Loading: Subject={subject}, Movement={movement}, Speed={speed}")
print("=" * 80)

eeg_data, sfreq, ch_names = load_eeg_data(subject, movement, speed, base_dir)
imu_data = load_imu_data(subject, movement, speed, base_dir)

print("\n" + "=" * 80)
print("IMU-EEG Correlation Analysis")
print("=" * 80)

filtered_eeg, filtered_imu = analyze_imu_eeg_correlation(
    eeg_data, imu_data, fs_eeg=sfreq, fs_imu=fs_imu)

print(f"\nFiltered shapes: EEG {filtered_eeg.shape}, IMU {filtered_imu.shape}")

if isinstance(duration_seconds, (int, float)):
    duration_list = [duration_seconds]
else:
    duration_list = duration_seconds

for duration in duration_list:
    print("\n" + "=" * 80)
    print(f"Processing: first {duration}s")
    print("=" * 80)

    print(f"\nComputing correlation matrix ({duration}s)...")
    print("FFT coherence...")
    fft_matrix = compute_fft_coherence_matrix(filtered_imu, filtered_eeg, fs=sfreq, duration_seconds=duration)
    norm_fft = normalize_matrix(fft_matrix)

    print("Pearson...")
    pearson_matrix = compute_pearson_correlation(filtered_imu, filtered_eeg, fs=sfreq, duration_seconds=duration)
    norm_pearson = normalize_matrix(pearson_matrix)

    pearson_weight = 0.2
    positive_pearson = np.copy(pearson_matrix)
    positive_pearson[positive_pearson < 0] = 0
    norm_positive_pearson = normalize_matrix(positive_pearson)
    hybrid_fft_matrix = pearson_weight * norm_positive_pearson + (1 - pearson_weight) * norm_fft

    print("Visualizing hybrid matrix...")
    visualize_hybrid_correlation(hybrid_fft_matrix, imu_names, ch_names)

    print("Finding extreme pairs...")
    top_pairs, bottom_pairs = find_extreme_pairs(hybrid_fft_matrix, imu_names, n_pairs=30, preference='positive')
    top_pairs_filtered = [(imu_idx, eeg_idx, value) for imu_idx, eeg_idx, value in top_pairs if eeg_idx < len(ch_names)]
    if visualize_bottom_pairs:
        bottom_pairs_filtered = [(imu_idx, eeg_idx, value) for imu_idx, eeg_idx, value in bottom_pairs if eeg_idx < len(ch_names)]

    print("\n" + "=" * 80)
    print(f"Top pairs (FFT hybrid, {duration}s):")
    print("=" * 80)
    for i, (imu_idx, eeg_idx, value) in enumerate(top_pairs_filtered[:10]):
        print(f"{i+1}. {imu_names[imu_idx]} - {ch_names[eeg_idx]}: {value:.4f}")

    if visualize_bottom_pairs:
        print("\n" + "=" * 80)
        print(f"Least correlated pairs ({duration}s):")
        print("=" * 80)
        for i, (imu_idx, eeg_idx, value) in enumerate(bottom_pairs_filtered[:10]):
            print(f"{i+1}. {imu_names[imu_idx]} - {ch_names[eeg_idx]}: {value:.4f}")

    if top_pairs_filtered:
        print(f"\nVisualizing top {min(n_visualize_pairs, len(top_pairs_filtered))} pairs...")
        visualize_extreme_pairs(filtered_imu, filtered_eeg,
                               top_pairs_filtered[:n_visualize_pairs],
                               imu_names, ch_names, f"Most Correlated ({duration}s)",
                               duration_seconds=duration, fs=sfreq)

    if visualize_bottom_pairs and bottom_pairs_filtered:
        print(f"\nVisualizing bottom {min(n_visualize_pairs, len(bottom_pairs_filtered))} pairs...")
        visualize_extreme_pairs(filtered_imu, filtered_eeg,
                               bottom_pairs_filtered[:n_visualize_pairs],
                               imu_names, ch_names, f"Least Correlated ({duration}s)",
                               duration_seconds=duration, fs=sfreq)

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
# %%
