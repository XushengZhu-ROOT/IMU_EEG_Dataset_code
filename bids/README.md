# The BIDS dataset

The dataset is built from its raw recordings (`sourcedata/`) and the tables in [tables/](../tables).

| File | Content |
|---|---|
| [make_bids.py](make_bids.py) | writes the raw BIDS dataset: the root files and, for every segment, the EEG (`sub-XX/eeg/`) and the IMU (`sub-XX/motion/`) with their sidecars and events |
| [imu_corrections.py](imu_corrections.py) | the IMU of one segment from the raw IMU rows, one function per correction type (column `correction` of `imu_segments.csv`) |
| [make_eeg_text.py](make_eeg_text.py) | `derivatives/eeg_text/`: the raw EEG of every recording as tab-separated text |

## Building the dataset

From the repository folder, in this order:

```bash
python tone_onsets/run.py --data /path/to/dataset/sourcedata --out /path/to/tone_onsets
python bids/make_bids.py --bids /path/to/dataset --out /path/to/new_dataset
# MATLAB: set BIDS = '/path/to/new_dataset', OUT and EEGLAB at the top of eeg_cleaning/run_all.m, then
#   run('eeg_cleaning/run_all.m')
python eeg_cleaning/make_clean_erp.py --bids /path/to/new_dataset --mat <OUT>/imureg_asr/k30 \
    --out /path/to/new_dataset/derivatives/clean
python bids/make_eeg_text.py --bids /path/to/new_dataset --out /path/to/new_dataset/derivatives/eeg_text
```

1. `tone_onsets/run.py` writes `tone_onsets.csv` (the corrected onset and EEG sample of every tone) and `onset_correction.csv` (the delay c of each audio set-up). The copies in `tables/` are the ones the dataset was made with; a new run gives the same two files. To build the dataset with the files of another run, copy `tables/`, replace the two files and pass the copy with `--tables`.
2. `bids/make_bids.py` reads `sourcedata/` of `--bids` and the tables, and writes the root files, `sub-XX/eeg/` and `sub-XX/motion/` into a new folder. Copy `sourcedata/` of the dataset into that folder to complete the raw dataset.
3. `eeg_cleaning/run_all.m` cleans the EEG of the Move segments in the versions compared in the paper (ASR cutoffs 10, 20 and 30) and draws the ERP and gyroscope figures; `eeg_cleaning/make_clean_erp.py` writes the version kept in the dataset (IMU regression + ASR, cutoff 30) as `derivatives/clean/`.
4. `bids/make_eeg_text.py` writes `derivatives/eeg_text/` from the BrainVision files of step 2. Every text file is read back and compared with its `.eeg` file value by value.

## What the tables are used for

| Table | Used for |
|---|---|
| `segments.csv`, `pieces.csv` | the EEG samples of every segment (cut with `sync/resync.py`), the file names (`bids_name`), the `boundary` events (where pieces meet, and `eeg_gaps`), the channel counts and headset models in `participants.tsv` |
| `imu_segments.csv` | the correction type and its parameters (`imu_corrections.py`); `Source`, `Correction` and `Quality` of the `_motion.json`; `Source` and `SubjectArtefactDescription` of the `_eeg.json`; `anomalies.tsv` |
| `imu_pieces.csv` | the IMU rows of every piece (corrections `none` and `offset`), `Source.IMURows` and `Source.IMURowsReleased`, `Segment.Pieces` |
| `imu_time_base_runs.csv` | the slot-by-slot placement of the corrections `drift`, `lost samples`, `eeg packet loss` and `fast imu clock` |
| `tone_onsets.csv` | the `standard` and `target` events in `_events.tsv` and `.vmrk` (tones without an EEG sample are left out) |
| `onset_correction.csv` | the delay per audio set-up in `task-<task>_events.json` |
| `slot_maps/` | the IMU slot by slot, read by `sync/`; the cut tables and the IMU tables give the same IMU in all 494 segments with IMU |

Computed from the data by `make_bids.py`:

- `Quality.MeasuredMovementRate` of each Move segment with usable IMU: the main measure and the intensity of `imu_analysis/imu_movement_rate.py`; for head shaking, head nodding and picking up the median of three estimates (`imu_analysis/cycle_rate.py`); for stairs the intensity only.
- `Quality.IMUTiltFromGravityDeg` and `Quality.GyroClippedSamples` of each `_motion.json`.
- `RestNote` of 8 short Rest files and of 6 Rest files that lie between the two speeds: the head motion from the headset accelerometer, compared with the median over all 258 Rest files. This needs every Rest file, so it is added after all participants are written.

Written as constants in `make_bids.py`: `README`, `CHANGES`, `.bidsignore`, `dataset_description.json`, `participants.json`, the ages in `participants.tsv`, `missing_recordings.tsv`, the task descriptions and the key-press notes of `RestNote`.

## Run times

MacBook with Apple silicon (10 cores), Python 3.9.6, MATLAB R2024a:

| Step | Time |
|---|---|
| `tone_onsets/run.py --jobs 3` | 32 s |
| `bids/make_bids.py --workers 3` | 50 s; about 1.7 GB of memory per worker for the largest recording (816 MB xdf) |
| `eeg_cleaning/run_all.m` | about 5 to 8 h, almost all of it for the two versions with IMU regression |
| `eeg_cleaning/make_clean_erp.py` | 9 s |
| `bids/make_eeg_text.py --workers 3` | about 1.5 min (496 files, 7.3 GB) |

## Checks

- `make_bids.py`: all 4,967 files outside `sourcedata/` and `derivatives/` are identical to the published dataset.
- `make_eeg_text.py`: all 496 text files, `README` and `dataset_description.json` are identical to `derivatives/eeg_text/` of the published dataset.
