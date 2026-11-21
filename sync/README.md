# IMU-EEG synchronization

The head IMU writes text files without time stamps (effective rate 49.647537 Hz). Its timing was corrected against the accelerometer built into the EEG headset, which runs on the EEG clock. The result is stored in the cut tables in [tables/](../tables): for each of the 496 segments they list the raw EEG samples (xdf) and the raw IMU rows (txt) that form the segment, so a segment is rebuilt by reading positions from these tables.

| File | Content |
|---|---|
| [resync.py](resync.py) | reads the cut tables and the raw recordings, cuts a segment, and measures how well the IMU follows the headset accelerometer |
| [verify_resync.py](verify_resync.py) | rebuilds all 496 segments from the raw recordings and checks them against the tables |
| [alignment_report.py](alignment_report.py) | agreement of the IMU with the headset accelerometer in every segment, for the IMU as originally released and after the correction |
| [matlab/](matlab) | the cut and the checks in MATLAB: `resync_tables`, `resync_cut`, `resync_apply`, `load_xdf_eeg`, `read_rabboni_txt`, `resync_alignment`, `demo_resync`, `verify_resync` |

The raw recordings are read from the dataset: `sourcedata/sub-XX/eeg/SubXX_<recording>.xdf` and `sourcedata/sub-XX/motion/SubXX_<recording>.txt`. The tables give these file names.

## The cut tables

Every sample, row and slot number is 1-based and inclusive. A segment is named `<sub>_<cond>_<seg>`, for example `Sub01_movement3_speed0_Move`; `bids_name` gives its name in the dataset. EEG sample k of a segment lies (k - 1)/500 s and IMU slot u lies (u - 1)/49.647537 s after the first EEG sample of the segment. A segment consists of one or more pieces; both streams leave out the pauses between pieces in the same way.

[segments.csv](../tables/segments.csv), one row per segment:

| Column | Meaning |
|---|---|
| `sub`, `cond`, `seg` | the segment; `move` and `speed` give the movement and the speed in words, `bids_name` the name in the dataset |
| `eeg_xdf`, `eeg_stream` | xdf file (two files are separated by `\|`) and the name of the EEG stream |
| `eeg_channels`, `eeg_acc_rows` | rows of the stream with the EEG channels and with the headset accelerometer (micro-g) |
| `eeg_start`, `eeg_end`, `eeg_samples` | first and last EEG sample (`file:sample`) and the number of samples |
| `eeg_gaps` | EEG packet loss inside the segment: `[xdf, first sample after the gap, samples lost]` |
| `imu_txt`, `imu_method`, `imu_slot_map` | IMU file(s); `rows`, `slot map`, or `none` for the 2 segments whose IMU file has no accelerometer; the slot map |
| `imu_start`, `imu_end`, `imu_slots`, `imu_empty` | first and last IMU slot with data (`file:row`), the number of slots and of empty slots |
| `imu_usable` | `yes`, `partly` or `no` |
| `problem` | samples lost by the EEG or the IMU, an IMU file that ends before the segment, or why the IMU is not usable |
| `rho`, `check_imu_sum` | reference values for `verify_resync` |

[pieces.csv](../tables/pieces.csv), one row per piece:

| Column | Meaning |
|---|---|
| `stream`, `piece`, `file` | `EEG` or `IMU`; order in the segment; xdf or txt file |
| `first`, `last` | EEG: first and last sample of the piece. IMU: first and last txt row with data |
| `n` | samples (EEG) or slots (IMU) of the piece |
| `seg_first`, `seg_last` | the same, counted inside the segment |
| `imu_method` | `rows` or `slot map` |
| `note` | `tail: leave this piece out`, or the number of empty slots at the end of a piece whose IMU file ends before the segment |
| `check_first`, `check_last` | raw values at the first and last sample or row of the piece, for `verify_resync` |

[slot_maps/](../tables/slot_maps) holds one file per segment whose IMU rows were placed slot by slot (29 segments: clock drift, lost IMU samples, EEG packet loss), with one line per IMU slot: `slot`, `txt_file`, `row` and `status` (`raw`, `interpolated` or `missing`). Lines after `imu_slots` belong to a tail.

The IMU corrections are in `imu_segments.csv`, `imu_pieces.csv` and `imu_time_base_runs.csv`; [bids/README.md](../bids/README.md) lists what they are used for.

## Cutting a segment

EEG: take samples `first` to `last` of every EEG piece of the segment from the named stream and put the pieces one after the other. The headset accelerometer is rows `eeg_acc_rows` divided by 1e6 (g). Continuous EEG can be filtered or re-referenced before the cut as long as every sample keeps its position; resample only after the cut.

IMU: a data row of the txt is a line that starts with a number and has at least 6 comma-separated numbers (accX, accY, accZ in g, gyroX, gyroY, gyroZ); row 1 is the first such line. With `imu_method = rows`, slot s of a piece is txt row `first + s - 1`, and a row past the end of the txt gives an empty slot (NaN). With `imu_method = slot map`, `raw` slots take the values of their row, `interpolated` slots (samples the IMU lost) lie on the straight line between the raw slots on either side, and `missing` slots are empty. Pieces marked `tail` are left out in both streams.

Python:

```python
import sys
sys.path.insert(0, '/path/to/IMU_EEG_Dataset_code')
from sync import resync as rs

T = rs.load_tables()                                     # tables/ of this repository
C = rs.cut(T, 'Sub01', 'movement3_speed0', 'Move')       # positions of the segment
eeg, imu = rs.cut_from_raw(C, '/path/to/dataset')        # EEG stream rows (samples x rows), IMU (slots x 6)
A = rs.alignment(rs.cgx_acc(eeg, C), imu[:, :3])         # rho and best lag of the whole segment, 5-s windows
```

`rs.apply(C, eeg_src, imu_src)` cuts data that are already loaded (dictionaries from file name to data), for example continuous EEG after filtering.

MATLAB, with `load_xdf` (EEGLAB plugin xdfimport, or xdf-Matlab) on the path:

```matlab
addpath('/path/to/IMU_EEG_Dataset_code/sync/matlab')
T = resync_tables();
C = resync_cut(T, 'Sub01', 'movement3_speed0', 'Move');
xdf = fullfile('/path/to/dataset', 'sourcedata', 'sub-01', 'eeg', C.eeg.file(1));
txt = fullfile('/path/to/dataset', 'sourcedata', 'sub-01', 'motion', C.imu.file(1));
[eeg, imu] = resync_apply(C, load_xdf_eeg(xdf, C.info.eeg_stream), read_rabboni_txt(txt));
```

[demo_resync.m](matlab/demo_resync.m) does this for one segment and prints its alignment. A segment over two xdf files (Sub05 and Sub07 `movement4_speed1` Move) needs a `containers.Map` from file name to data, as in `demo_resync.m`.

## Checking the cut: verify_resync

```bash
python sync/verify_resync.py --bids /path/to/dataset --out verify_out
```

```matlab
R = verify_resync('/path/to/dataset', 'Out', 'verify_out');
```

Every segment is cut from the raw recordings with the tables and compared with the tables:

1. Positions. EEG: number of samples, first and last xdf sample. IMU: number of slots and of empty slots, first and last txt row with data.
2. Values. Raw values at the first and last sample or row of every piece (`check_first`, `check_last`), and the sum of accX + accY + accZ over the IMU slots with data (`check_imu_sum`). A cut that is off by one sample fails here.
3. Alignment. Segments of 40 s or more: the whole-segment rho of `resync.alignment` lies within 0.02 of `rho` and the best lag is 0 (+/-1 slot). Shorter segments, mostly quiet rest, are only reported.

`verify_out/verify_segments.csv` has one row per segment. Output of the Python version (35 s with 3 workers on a 10-core Apple-silicon Mac; the MATLAB version takes 6 min):

```
EEG: 496/496 segments pass checks 1 and 2
IMU: 494/494 segments pass checks 1 and 2 (2 segments have no IMU accelerometer)
check 3: {'ok': 275, 'shorter than 40 s': 216, 'not usable (imu_usable = no)': 3, 'no IMU': 2}
```

## Agreement before and after the correction: alignment_report.py

```bash
python sync/alignment_report.py --bids /path/to/dataset --out alignment_out
```

For every segment it compares the IMU accelerometer with the headset accelerometer, for the IMU as originally released (the released rows of `imu_pieces.csv`, read from the raw txt) and for the corrected IMU of the dataset (`_motion.tsv`). The headset accelerometer is low-passed at 20 Hz and interpolated at the IMU sample times; both signals are band-passed at 0.3-10 Hz. rho is the rotation-invariant Procrustes correlation of the two 3-axis signals: the IMU is turned into the headset axes by the best proper rotation, and rho = 1 means the same movement. 2 s at both ends and 1 s around empty IMU slots are left out; the best lag is searched within +/-0.5 s.

Output: `alignment_release_vs_final.csv` (one row per segment) and `alignment_summary.txt`, the medians over the segments with usable IMU:

```
Move: 237 segments with an IMU accelerometer, 236 of them with usable IMU (imu_usable yes or partly)
  as released rho median 0.960 (IQR 0.869-0.985), rho >= 0.8: 190/236; best lag 0: 75/236, |best lag| <= 1 sample (20 ms): 177/236
  corrected   rho median 0.984 (IQR 0.973-0.992), rho >= 0.8: 229/236; best lag 0: 236/236, |best lag| <= 1 sample (20 ms): 236/236
Rest: 257 segments with an IMU accelerometer, 255 of them with usable IMU (imu_usable yes or partly)
  as released rho median 0.711 (IQR 0.514-0.853), rho >= 0.5: 195/255; best lag 0: 105/255, |best lag| <= 1 sample (20 ms): 195/255
  corrected   rho median 0.772 (IQR 0.686-0.896), rho >= 0.5: 242/255; best lag 0: 253/255, |best lag| <= 1 sample (20 ms): 255/255
```
