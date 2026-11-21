# IMU-EEG synchronization

The head IMU writes text files without time stamps (49.647537 Hz). Its timing was corrected against the accelerometer in the EEG headset, which runs on the EEG clock. The cut tables in [tables/](../tables) list, for each of the 496 segments, the raw EEG samples (xdf) and the raw IMU rows (txt) that form it.

| File | Content |
|---|---|
| [resync.py](resync.py) | cuts a segment from the raw recordings; agreement of the IMU with the headset accelerometer |
| [verify_resync.py](verify_resync.py) | rebuilds all 496 segments and compares them with the tables |
| [alignment_report.py](alignment_report.py) | agreement of the IMU with the headset accelerometer in every segment, before and after the correction |
| [matlab/](matlab) | the same in MATLAB |

The raw recordings are read from the dataset: `sourcedata/sub-XX/eeg/SubXX_<recording>.xdf` and `sourcedata/sub-XX/motion/SubXX_<recording>.txt`.

## The cut tables

Every sample, row and slot number is 1-based and inclusive. A segment is named `<sub>_<cond>_<seg>`, e.g. `Sub01_movement3_speed0_Move`; `bids_name` is its name in the dataset. EEG sample k of a segment lies (k - 1)/500 s and IMU slot u lies (u - 1)/49.647537 s after the first EEG sample of the segment. A segment consists of one or more pieces.

[segments.csv](../tables/segments.csv), one row per segment:

| Column | Meaning |
|---|---|
| `sub`, `cond`, `seg`, `move`, `speed`, `bids_name` | the segment |
| `eeg_xdf`, `eeg_stream` | xdf file and the name of its EEG stream |
| `eeg_channels`, `eeg_acc_rows` | rows of the stream with the EEG channels and with the headset accelerometer (micro-g) |
| `eeg_start`, `eeg_end`, `eeg_samples` | first and last EEG sample (`file:sample`) and the number of samples |
| `eeg_gaps` | EEG packet loss: `[xdf, first sample after the gap, samples lost]` |
| `imu_txt`, `imu_method`, `imu_slot_map` | IMU file, `rows` or `slot map`, and the slot map |
| `imu_start`, `imu_end`, `imu_slots`, `imu_empty` | first and last IMU slot with data (`file:row`), the number of slots and of empty slots |
| `imu_usable`, `problem` | `yes`, `partly` or `no`, and the problem of the segment |
| `rho`, `check_imu_sum` | reference values for `verify_resync` |

[pieces.csv](../tables/pieces.csv), one row per piece:

| Column | Meaning |
|---|---|
| `stream`, `piece`, `file` | `EEG` or `IMU`, order in the segment, xdf or txt file |
| `first`, `last`, `n` | EEG: first and last sample; IMU: first and last txt row; number of samples or slots |
| `seg_first`, `seg_last` | the same, counted inside the segment |
| `imu_method`, `note` | `rows` or `slot map`; `tail: leave this piece out`, or the empty slots at the end of the IMU file |
| `check_first`, `check_last` | raw values at the first and last sample or row, for `verify_resync` |

[slot_maps/](../tables/slot_maps): one file per segment whose IMU rows were placed slot by slot (29 segments), with `slot`, `txt_file`, `row` and `status` (`raw`, `interpolated` or `missing`).

## Cutting a segment

EEG: take samples `first` to `last` of every EEG piece and put the pieces one after the other. The headset accelerometer is rows `eeg_acc_rows` divided by 1e6 (g).

IMU: a data row of the txt is a line that starts with a number and has at least 6 comma-separated numbers (accX, accY, accZ in g, gyroX, gyroY, gyroZ). With `imu_method = rows`, slot s of a piece is txt row `first + s - 1`. With `imu_method = slot map`, `raw` slots take the values of their row, `interpolated` slots lie on the straight line between the raw slots on either side, and `missing` slots are empty. Pieces marked `tail` are left out.

Python:

```python
import sys
sys.path.insert(0, '/path/to/IMU_EEG_Dataset_code')
from sync import resync as rs

T = rs.load_tables()
C = rs.cut(T, 'Sub01', 'movement3_speed0', 'Move')
eeg, imu = rs.cut_from_raw(C, '/path/to/dataset')     # EEG stream rows (samples x rows), IMU (slots x 6)
A = rs.alignment(rs.cgx_acc(eeg, C), imu[:, :3])      # rho and best lag
```

MATLAB, with `load_xdf` on the path:

```matlab
addpath('/path/to/IMU_EEG_Dataset_code/sync/matlab')
T = resync_tables();
C = resync_cut(T, 'Sub01', 'movement3_speed0', 'Move');
xdf = fullfile('/path/to/dataset', 'sourcedata', 'sub-01', 'eeg', C.eeg.file(1));
txt = fullfile('/path/to/dataset', 'sourcedata', 'sub-01', 'motion', C.imu.file(1));
[eeg, imu] = resync_apply(C, load_xdf_eeg(xdf, C.info.eeg_stream), read_rabboni_txt(txt));
```

A segment over two xdf files (Sub05 and Sub07 `movement4_speed1` Move) needs a `containers.Map` from file name to data, as in [demo_resync.m](matlab/demo_resync.m).

## Rebuilding all segments

```bash
python sync/verify_resync.py --bids /path/to/dataset --out verify_out
python sync/alignment_report.py --bids /path/to/dataset --out alignment_out
```

In MATLAB: `verify_resync('/path/to/dataset', 'Out', 'verify_out')`.
