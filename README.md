# IMU_EEG_Dataset_code

Code of the dataset paper "Real-World Multiactivity EEG-IMU Dataset for Motion Artifact Removal and Classification Research": synchronization of the head IMU with the EEG, the BIDS dataset built from the raw recordings, cleaning of the EEG of the movement periods, the analyses and figures of the paper, and the movement classification.

## Data

The dataset is on OpenNeuro. Its `sourcedata/` holds the raw recordings: `sub-XX/eeg/SubXX_<recording>.xdf` (EEG, headset accelerometer and tone markers, recorded with Lab Streaming Layer) and `sub-XX/motion/SubXX_<recording>.txt` (head IMU).

The scripts take the dataset folder with `--bids` (`tone_onsets/run.py` takes its `sourcedata/` with `--data`), read the tables in [tables/](tables) and write into a new or empty folder given with `--out`. `--help` prints the usage of a script; the docstring at its top lists its outputs.

## Requirements

- Python 3.9 with the packages in [requirements.txt](requirements.txt): `pip install -r requirements.txt`.
- MATLAB R2024a (Signal Processing and Statistics toolboxes) for the MATLAB code, with EEGLAB 2025.1.0 and its clean_rawdata 2.11 plugin for [eeg_cleaning/](eeg_cleaning), and load_xdf (EEGLAB plugin xdfimport, or [xdf-Matlab](https://github.com/xdf-modules/xdf-Matlab)) for [sync/matlab/](sync/matlab).
- The Python figures use Helvetica, the macOS system font; on other systems matplotlib uses its default sans-serif font.
- [classification/](classification) lists its own requirements.

## Contents

| Folder | Content |
|---|---|
| [tables/](tables) | the cut tables (raw EEG samples and IMU rows of every segment), the IMU correction tables, the tone onsets and the delay of each audio set-up |
| [sync/](sync) | every segment rebuilt from the raw recordings in Python and MATLAB, and the checks of the IMU timing ([README](sync/README.md)) |
| [tone_onsets/](tone_onsets) | the onset of every tone: LSL marker time minus the delay of its audio set-up, and its EEG sample |
| [bids/](bids) | the BIDS dataset from the raw recordings and the tables ([README](bids/README.md)) |
| [eeg_cleaning/](eeg_cleaning) | cleaning of the EEG of the Move segments (`derivatives/clean` of the dataset and the versions compared in the paper) and the ERP measures |
| [imu_analysis/](imu_analysis) | IMU timing, movement rate, IMU orientation and IMU-EEG correlation |
| [figures/](figures) | the figures of the paper |
| [classification/](classification) | movement classification: CBraMod fine-tuned with leave-one-subject-out cross-validation ([README](classification/README.md)) |
| [stimulus/](stimulus) | the auditory oddball program that played the tones during the recordings |

## Figures and tables of the paper

| Paper | Code |
|---|---|
| Fig. 2, Table 2 | [make_fig_speed.py](figures/make_fig_speed.py), with [imu_movement_rate.py](imu_analysis/imu_movement_rate.py) and [cycle_rate.py](imu_analysis/cycle_rate.py) |
| Fig. 3 | [make_gyro_figure.m](figures/make_gyro_figure.m), with [declip_gyro.m](eeg_cleaning/declip_gyro.m) |
| Fig. 5 | [make_fig_5s_example.py](figures/make_fig_5s_example.py) |
| Fig. 6, Table 5 | [imu_eeg_correlation.py](imu_analysis/imu_eeg_correlation.py), then [make_fig_group.py](figures/make_fig_group.py) |
| Fig. 7 | [make_fig_single_trial.py](figures/make_fig_single_trial.py) |
| Fig. 8, Fig. 9 | [run_all.m](eeg_cleaning/run_all.m): [clean_paper_versions.m](eeg_cleaning/clean_paper_versions.m), [erp_measures.m](eeg_cleaning/erp_measures.m), [plot_erp_figures.m](figures/plot_erp_figures.m) |
| Table 6 | [classification/](classification) |
| Technical Validation, timing of the IMU relative to the EEG | [imu_sync_validation.py](imu_analysis/imu_sync_validation.py), [alignment_report.py](sync/alignment_report.py) |

Fig. 1 and 4 and Tables 1, 3 and 4 contain no computed values.

From the repository folder:

```bash
python figures/make_fig_speed.py --bids /path/to/dataset --out out/fig_speed
python figures/make_fig_5s_example.py --bids /path/to/dataset --out out/fig_5s_example
python imu_analysis/imu_eeg_correlation.py --bids /path/to/dataset --out out/imu_eeg_correlation
python figures/make_fig_group.py --results out/imu_eeg_correlation --out out/fig_group
python figures/make_fig_single_trial.py --bids /path/to/dataset --out out/fig_single_trial
python imu_analysis/imu_sync_validation.py --bids /path/to/dataset --out out/imu_sync_validation
python sync/alignment_report.py --bids /path/to/dataset --out out/alignment
```

In MATLAB, set `BIDS`, `OUT` and `EEGLAB` at the top of [run_all.m](eeg_cleaning/run_all.m) and run `run('eeg_cleaning/run_all.m')` from the repository folder. It cleans the EEG of the 238 Move segments in the six versions of Fig. 8 and 9, computes the ERPs and draws Fig. 3, 8 and 9. On a 10-core Apple-silicon Mac this takes about 5 to 8 hours, almost all of it for the two versions with IMU regression.

## Rebuilding the dataset

[bids/README.md](bids/README.md) gives the order (`tone_onsets/run.py`, `bids/make_bids.py`, `eeg_cleaning/run_all.m` with `eeg_cleaning/make_clean_erp.py`, `bids/make_eeg_text.py`) and what each table is used for. [sync/README.md](sync/README.md) explains the cut tables and how to cut a segment from the raw recordings.

## Contributors

- Xusheng Zhu: synchronization, BIDS dataset, EEG cleaning, analyses, figures and classification.
- Ella (HsinYu) Shih: EEG preprocessing of the classification ([preprocess_eeg_forASR.m](classification/preprocess_eeg_forASR.m), [apply_asr.m](classification/apply_asr.m)).

## License

The code is released under the [MIT License](LICENSE). The dataset is released under CC0.

## Citation

Zhu X, Shih H, Lu C, Jian M, Wu YC, Jung T-P. Real-World Multiactivity EEG-IMU Dataset for Motion Artifact Removal and Classification Research.
