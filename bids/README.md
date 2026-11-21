# BIDS dataset

| File | Content |
|---|---|
| [make_bids.py](make_bids.py) | writes the BIDS dataset (root files, `sub-XX/eeg/`, `sub-XX/motion/`) from the raw recordings and the tables |
| [imu_corrections.py](imu_corrections.py) | IMU of one segment from the raw IMU rows, one function per correction type |
| [make_eeg_text.py](make_eeg_text.py) | `derivatives/eeg_text/`: the raw EEG as tab-separated text |

## Building the dataset

From the repository folder, in this order:

```bash
python tone_onsets/run.py --data /path/to/dataset/sourcedata --out /path/to/tone_onsets
python bids/make_bids.py --bids /path/to/dataset --out /path/to/new_dataset
# MATLAB: set BIDS, OUT and EEGLAB at the top of eeg_cleaning/run_all.m, then run('eeg_cleaning/run_all.m')
python eeg_cleaning/make_clean_erp.py --bids /path/to/new_dataset --mat <OUT>/imureg_asr/k30 \
    --out /path/to/new_dataset/derivatives/clean
python bids/make_eeg_text.py --bids /path/to/new_dataset --out /path/to/new_dataset/derivatives/eeg_text
```

Copy `sourcedata/` of the dataset into the new folder.

## Tables

| Table | Used for |
|---|---|
| `segments.csv`, `pieces.csv`, `slot_maps/` | EEG samples and IMU rows of every segment |
| `imu_segments.csv`, `imu_pieces.csv`, `imu_time_base_runs.csv` | IMU correction of every segment and the fields of `_motion.json` |
| `tone_onsets.csv`, `onset_correction.csv` | tone events and the delay of each audio set-up |
