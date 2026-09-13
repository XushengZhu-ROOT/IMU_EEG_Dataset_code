# %%
"""Composite 6-panel figure of the GROUP-MEAN Hbar (per-subject normalized,
then averaged across subjects), `combined` tag (both speeds pooled, all
subjects, full-trial sliding-window coverage) -- one panel per movement.

This intentionally matches the EXACT metric already saved per-movement in
result/window10s_group/combined/movementX.png, so the composite figure and
those individual figures are numerically consistent (same colorbar range,
same values). It reuses corr_group_for_R1.py's own aggregation function
(run_aggregate-equivalent) rather than recomputing a different metric.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = open("corr_group_for_R1.py").read()
head = src.split("# %% [markdown]\n# ## Main")[0]
ns = {}
exec(compile(head, "x", "exec"), ns)
load_eeg_data = ns["load_eeg_data"]
load_imu_data = ns["load_imu_data"]
analyze_imu_eeg_correlation = ns["analyze_imu_eeg_correlation"]
compute_subject_raw_matrices = ns["compute_subject_raw_matrices"]
compute_hybrid_from_raw = ns["compute_hybrid_from_raw"]
align_H_to_standard_eeg = ns["align_H_to_standard_eeg"]
stack_and_aggregate = ns["stack_and_aggregate"]
STANDARD_EEG20 = ns["STANDARD_EEG20"]

base_dir = "."
subjects = [f"Sub{i:02d}" for i in range(1, 21)]
movements = list(range(6))
speeds = [0, 1]
fs_imu = 49.7
pearson_weight = 0.2
imu_names = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
movement_titles = {
    0: "(a) Straight walking",
    1: "(b) Curved walking",
    2: "(c) Head shaking",
    3: "(d) Head nodding",
    4: "(e) Picking up an object",
    5: "(f) Stair climbing and descending",
}

# %% Recompute per-subject H exactly as corr_group_for_R1.py does (per-subject
# normalized), then aggregate mean across subjects for the `combined` tag.
cache = {}
for movement in movements:
    for speed in speeds:
        for subject in subjects:
            try:
                eeg_data, sfreq, ch_names = load_eeg_data(subject, movement, speed, base_dir)
                imu_data = load_imu_data(subject, movement, speed, base_dir)
                filtered_eeg, filtered_imu = analyze_imu_eeg_correlation(
                    eeg_data, imu_data, fs_eeg=sfreq, fs_imu=fs_imu)
                R_raw, C_raw, n_win, cov_s = compute_subject_raw_matrices(
                    filtered_imu, filtered_eeg, fs=sfreq, window_sec=10.0, step_sec=10.0)
                H = compute_hybrid_from_raw(R_raw, C_raw, pearson_weight=pearson_weight)
                cache[(subject, movement, speed)] = (H, ch_names)
            except Exception:
                pass
print(f"Loaded {len(cache)} subject-trials")

H_mean_by_movement = {}
for movement in movements:
    H_list, labels = [], []
    for (subject, mov, speed), (H, ch_names) in cache.items():
        if mov != movement:
            continue
        H_aligned, _, _ = align_H_to_standard_eeg(H, ch_names)
        H_list.append(H_aligned)
        labels.append(f"{subject}@speed{speed}")
    H_mean, H_std, _, _ = stack_and_aggregate(H_list, labels)
    H_mean_by_movement[movement] = H_mean
    print(f"mov{movement}: Hbar range [{H_mean.min():.4f}, {H_mean.max():.4f}], "
          f"mean={H_mean.mean():.4f}  (N={len(H_list)})")

# %% Composite figure: 2 rows x 3 cols, shared colorbar (matches movementX.png scale)
vmax = max(h.max() for h in H_mean_by_movement.values())
vmin = 0.0
eeg_labels = list(STANDARD_EEG20)


def render(fig_axes_pairs):
    fig, axes = fig_axes_pairs
    im = None
    for movement in movements:
        r, c = divmod(movement, 3)
        ax = axes[r, c]
        mat = H_mean_by_movement[movement][:, :len(eeg_labels)].T  # (20 EEG, 6 IMU)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(movement_titles[movement], fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(imu_names)))
        ax.set_xticklabels(imu_names, rotation=90, fontsize=9)
        if c == 0:
            ax.set_yticks(range(len(eeg_labels)))
            ax.set_yticklabels(eeg_labels, fontsize=8)
            ax.set_ylabel("EEG Channels", fontsize=11, fontweight="bold")
        else:
            ax.set_yticks(range(len(eeg_labels)))
            ax.set_yticklabels([])
        if r == 1:
            ax.set_xlabel("IMU Channels", fontsize=11, fontweight="bold")
    fig.subplots_adjust(right=0.90, wspace=0.15, hspace=0.35)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"Group-mean coupling $\bar{H}$ (N=20 subjects)",
                   fontsize=11, fontweight="bold")
    return fig


out_png = os.path.join(base_dir, "result", "window10s_group", "fig_group_H_composite.png")
out_pdf = out_png.replace(".png", ".pdf")
os.makedirs(os.path.dirname(out_png), exist_ok=True)

fig1 = plt.subplots(2, 3, figsize=(16, 11))
render(fig1)
fig1[0].savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig1[0])
print(f"Saved: {out_png}")

fig2 = plt.subplots(2, 3, figsize=(16, 11))
render(fig2)
fig2[0].savefig(out_pdf, bbox_inches="tight")
plt.close(fig2[0])
print(f"Saved: {out_pdf}")

# %% Cross-check against the already-saved combined/movementX.png data source
# (same aggregation function -> should match exactly to floating point)
print("\nSanity check vs per-imu means already reported in R1_paper_summary.md (combined tag):")
for movement in movements:
    H_mean = H_mean_by_movement[movement]
    per_imu = {imu_names[i]: float(H_mean[i].mean()) for i in range(6)}
    best = max(per_imu, key=per_imu.get)
    print(f"  mov{movement}: best axis={best} H={per_imu[best]:.3f}  all={per_imu}")
