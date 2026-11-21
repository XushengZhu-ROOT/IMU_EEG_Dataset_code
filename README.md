# IMU_EEG_Dataset_code

Code of the dataset paper "Real-World Multiactivity EEG-IMU Dataset for Motion Artifact Removal and Classification Research".

## Data

The dataset is on OpenNeuro; the raw recordings are in its `sourcedata/`. The scripts read the dataset folder given with `--bids` and write into a new folder given with `--out`; `--help` prints the usage.

## Requirements

- Python 3.9 and the packages in [requirements.txt](requirements.txt)
- MATLAB R2024a with EEGLAB 2025.1.0 (clean_rawdata 2.11) and load_xdf

## Contents

| Folder | Content |
|---|---|
| [tables/](tables) | cut tables, IMU correction tables, tone onsets and audio delays |
| [sync/](sync) | rebuilding every segment from the raw recordings (Python, MATLAB) |
| [tone_onsets/](tone_onsets) | corrected onset and EEG sample of every tone |
| [bids/](bids) | building the BIDS dataset |
| [eeg_cleaning/](eeg_cleaning) | cleaning of the Move EEG (`derivatives/clean`) and the ERPs |
| [imu_analysis/](imu_analysis) | IMU timing, movement rate and IMU-EEG correlation |
| [figures/](figures) | figures of the paper |
| [classification/](classification) | movement classification with CBraMod |
| [stimulus/](stimulus) | auditory oddball program used in the recordings |

## License

MIT, see [LICENSE](LICENSE).

## Citation

Zhu X, Shih H, Lu C, Jian M, Wu YC, Jung T-P. Real-World Multiactivity EEG-IMU Dataset for Motion Artifact Removal and Classification Research.
