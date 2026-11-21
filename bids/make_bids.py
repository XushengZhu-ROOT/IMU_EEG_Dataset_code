"""Write the BIDS dataset from the raw recordings and the tables of this repository.

    python bids/make_bids.py --bids /path/to/dataset --out /path/to/new_folder [--workers 3]

Reads sourcedata/sub-XX/eeg/*.xdf and sourcedata/sub-XX/motion/*.txt of --bids and, from tables/: the cut tables
(segments.csv, pieces.csv) for the EEG, the IMU tables (imu_segments.csv, imu_pieces.csv, imu_time_base_runs.csv) for the
IMU, tone_onsets.csv and onset_correction.csv for the tones. Writes the root files and sub-XX/eeg, sub-XX/motion of every
segment into --out. sourcedata/ comes with the dataset; derivatives/ is written by eeg_cleaning/make_clean_erp.py and
bids/make_eeg_text.py.
"""
import argparse
import json
import os
import pathlib
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import pyxdf
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import imu_corrections as ic  # noqa: E402  (this folder)
from imu_analysis import cycle_rate as cr  # noqa: E402
from imu_analysis import imu_movement_rate as mr  # noqa: E402
from sync import resync as rs  # noqa: E402

TABLES = pathlib.Path(__file__).resolve().parents[1] / 'tables'
FS_EEG = 500
FS_IMU = rs.FS_IMU
COLUMNS = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']
CLIP = 990.0                                       # gyroscope values at or above this count as clipped
SOS_HEAD = butter(2, [0.3, 10], 'bandpass', fs=FS_EEG, output='sos')
KEEP_HEADER = ('ACC FSR', 'GYRO FSR', 'Interrupt Threshold', 'Data Rate', 'Data Type', 'Feature', 'Gyro Cal', 'Gyro Data',
               'Acc Data')
NAME = 'Real-World Multiactivity EEG-IMU Dataset for Motion Artifact Removal and Classification Research'
AUTHORS = ['Xusheng Zhu', 'HsinYu Shih', 'ChiaEn Lu', 'Muqing Jian', 'Ying Choon Wu', 'Tzyy-Ping Jung']
AGE = {'sub-01': 21, 'sub-02': 24, 'sub-03': 23, 'sub-04': 24, 'sub-05': 20, 'sub-06': 29, 'sub-07': 24, 'sub-08': 23,
       'sub-09': 23, 'sub-10': 23, 'sub-11': 26, 'sub-12': 23, 'sub-13': 21, 'sub-14': 24, 'sub-15': 27, 'sub-16': 24,
       'sub-17': 21, 'sub-18': 23, 'sub-19': 27, 'sub-20': 22}
MISSING = {'sub-05': 'headshake (both speeds)', 'sub-11': 'headnod slow: no usable IMU'}

TASK_TEXT = {
    'rest': 'Baseline: the participant stayed still (sitting; standing for Sub01 and Sub02).',
    'walkstraight': 'Straight walking: back and forth along a straight path across an enclosed room, turning around at each '
                    'end.',
    'walkcurved': 'Curved walking: from one end of the same room to the other along a self-chosen S-shaped path, and back.',
    'headshake': 'Head shaking: repeated left-right head rotation while standing.',
    'headnod': 'Head nodding: repeated up-down head rotation while standing.',
    'pickup': 'Picking up: repeatedly picking up an object from the floor, in the way natural to the participant (bending '
              'or squatting).',
    'stairs': 'Stair climbing and descending on an outdoor staircase.'}
INSTRUCTIONS = ('Each movement block: 30 s standing still (acq-slowrest or acq-fastrest), then 3 min of continuous movement '
                '(acq-slowmove or acq-fastmove), then 30 s standing still. Speed was self-paced: participants were asked to '
                'make the slow and the fast condition as different as they could. An auditory oddball task ran throughout '
                '(clench the right fist for the high tone, the left fist for the low tone).')
GYRO_NOTE = ('The file values are as written by the device with its nominal setting of +/-1000 deg/s. Compared with the '
             'tilt from gravity (slow nods) and with the 180 deg of a U-turn, they are about 3.7 and 4.1 times the true '
             'angular rate, respectively, and they stop at about +/-1000 (clipping).')
CHANNELS = pd.DataFrame(dict(
    name=COLUMNS + ['latency'], component=['x', 'y', 'z', 'x', 'y', 'z', 'n/a'],
    type=['ACCEL'] * 3 + ['GYRO'] * 3 + ['LATENCY'], tracked_point=['head'] * 6 + ['n/a'],
    units=['g'] * 3 + ['deg/s'] * 3 + ['s'], placement=['EEG cap on top of the head'] * 6 + ['n/a'],
    reference_frame=['local'] * 6 + ['n/a'],
    description=['IMU accelerometer'] * 3 + ['IMU gyroscope'] * 3 + ['time from the first EEG sample of the segment']))
CH_JSON = {
    'reference_frame': {'Description': 'Reference frame of the IMU axes.', 'Levels': {'local': {
        'SpatialAxes': 'RAS', 'RotationRule': 'right-hand',
        'Description': 'Axes of the sensor (rabboni manual, section 4.2): X along the writing direction of the word '
                       '"rabboni", Y towards the top of the letters, Z out of the front face. The sensor was fixed on the '
                       'EEG cap on top of the head, with the word upright when seen from behind the head, so X = right, '
                       'Y = anterior, Z = superior. The frame moves with the head. How the cap sat differed a little '
                       'between participants: the tilt of this segment from gravity is Quality.IMUTiltFromGravityDeg in '
                       'the _motion.json.'}}},
    'placement': {'Description': 'Where the sensor sat on the body.'}}
PARTICIPANTS_JSON = {
    'age': {'Description': 'age of the participant at the time of the recordings', 'Units': 'year'},
    'baseline_posture': {'Description': 'posture during the baseline recording (sit or stand)'},
    'eeg_channels': {'Description': 'number of EEG channels of the headset used; two numbers if it differed between '
                                    'recordings'},
    'eeg_30_channel_recordings': {'Description': 'number of segments recorded with the 30-channel headset'},
    'eeg_device': {'Description': 'EEG headset model'},
    'missing_recordings': {'Description': 'see missing_recordings.tsv'}}
MISSING_RECORDINGS = (
    [dict(participant_id='sub-05', condition=f'movement2_speed{sp}', part=p, task='headshake', what='not recorded',
          reason='the participant could not perform the head-shaking task') for sp in (0, 1) for p in ('Move', 'Rest')]
    + [dict(participant_id='sub-11', condition='movement3_speed0', part=p, task='headnod', what='no usable IMU',
            reason='the IMU file was recorded at 10 Hz instead of 50 Hz; the EEG exists') for p in ('Move', 'Rest')])
CHANGES = '1.0.0 2026-11-18\n  - Initial release.\n'
BIDSIGNORE = 'anomalies.tsv\nmissing_recordings.tsv\nderivatives/eeg_text/\n'
README = """# Real-World Multiactivity EEG-IMU Dataset

EEG (20 channels; 30 for sub-04 except head shaking and for the sitting baseline of sub-05) and a head-mounted \
6-axis IMU recorded simultaneously from 20 participants during a seated or standing baseline and six
self-paced movement tasks, with a concurrent auditory oddball task.

## Files
- One recording per segment. `task` is the movement; `acq` is the speed and the part of the block: `slowmove`, \
`slowrest`, `fastmove`, `fastrest` (for `task-rest`: `sit` or `stand`).
  `move` = the movement period of about 3 minutes, `rest` = the standing period of about 30 s before it.
- `sub-XX/eeg/`: EEG, BrainVision (`.vhdr .vmrk .eeg`, 32-bit float), exactly the values stored in the original xdf \
recording. `_events.tsv` (and the `.vmrk`) holds the auditory oddball tones (`standard`, `target`) at the corrected \
sound onsets, and marks the places where pieces meet or the EEG lost samples (`boundary`); the columns and the onset \
correction are described in `task-<task>_events.json`.
- `sub-XX/motion/`: IMU, `_motion.tsv` without header: accX accY accZ (g), gyroX gyroY gyroZ, latency (s). `n/a` = \
no data.
- `sourcedata/sub-XX/eeg/` and `sourcedata/sub-XX/motion/`: the original recordings, xdf files (EEG, headset \
accelerometer and marker streams) and IMU text files, named as recorded. The recording date was taken out: in the \
header of the IMU text files, and in the marker strings of the xdf files, where `readable_time` keeps the time of \
day and the Unix times (`system_time`, `offset`, audio markers) are shifted by a whole number of days per \
participant, so that times of day and time differences are unchanged. `Source` in each `_eeg.json` and \
`_motion.json` gives the files and the EEG samples and IMU rows used.

## Time alignment of EEG and IMU
The IMU file has no time stamps. It was aligned to the EEG with the accelerometer built into the EEG headset, which \
runs on the EEG clock.
EEG sample k (0-based) of a segment lies at k / 500 s, IMU sample u (0-based) at u / 49.647537 s (column latency), \
both counted from the first EEG sample of the segment.
How each segment was corrected is in `Correction`, and the rows used in `Source`, of its `_motion.json`.

## IMU
- Device: rabboni (InvenSense ICM-20689). Sampling rate: nominal 50 Hz, effective 49.647537 Hz. Nyquist frequency \
24.8 Hz.
- Coordinate system: X = right, Y = anterior, Z = superior, right-handed, fixed to the head (`_channels.json`). The \
IMU was fixed on the EEG cap on top of the head; how the cap sat differed a little between participants: the tilt of \
each segment from gravity is in `Quality.IMUTiltFromGravityDeg`.
- Accelerometer: g, range +/-2 g.
- Gyroscope: """ + GYRO_NOTE + """ The number of clipped samples of each segment is in `Quality.GyroClippedSamples`.

## EEG
CGX Quick-20 (Quick-32 for sub-04 except head shaking, and for the sitting baseline of sub-05), dry electrodes at \
the fixed positions of the headset, named after the 10-20 system (10-10 names for the additional Quick-32 channels), \
500 Hz, referenced to the A1 earclip; A2 is recorded as a channel. Channels ACC1-3 are the built-in accelerometer of \
the headset (micro-g; its axes agree with: ACC1 = left, ACC2 = posterior, ACC3 = superior in the Quick-20 and ACC1 = \
superior, ACC2 = anterior, ACC3 = left in the Quick-32, estimated from the data; the manufacturer gives neither the \
position nor the axes).

## Slow and fast
Self-paced: participants were asked to make the two speeds as different as they could. The measured rate of each \
segment (steps, cycles or pick-ups per minute, or the one-way time across the room) is in \
`Quality.MeasuredMovementRate` of the `_motion.json`, together with the movement intensity `IntensityAccelG` (root \
mean square of the three-axis acceleration after a 0.3-10 Hz band-pass, in g). For stairs only the intensity is \
given.

## Known problems
- Some recordings were not made, or their IMU file cannot be used: listed in `missing_recordings.tsv`.
- In some segments the IMU does not follow the EEG: listed in `anomalies.tsv` and described in the `_motion.json` of \
the segment.
- Some segments lost EEG samples, end with IMU slots without data, or have interpolated IMU samples: described in \
`Quality.DataNote*` of the `_motion.json` of the segment.

## Derivatives

- `derivatives/clean/`: cleaned EEG of the 238 movement segments that keeps event-related potentials: regression on \
the head motion (headset accelerometer and head IMU) and Artifact Subspace Reconstruction (cutoff 30), 500 Hz, \
reference A1 as recorded, same channels, samples and events as the raw files. See `derivatives/clean/README`.
- `derivatives/eeg_text/`: the raw EEG of every recording as tab-separated text with a header row: `latency_s` (s \
from the first sample, the time base of `_events.tsv` and of the IMU `latency`), then the channels of the \
BrainVision file in the same order and units (`_channels.tsv`); values identical to the `.eeg` files. See \
`derivatives/eeg_text/README`.
"""

# Rest files that are short or lie between the two speeds: RestNote in their _eeg.json and _motion.json
SHORT_REST = {
    'sub-01_task-walkstraight_acq-slowrest': 'The S key was held for 24.6 s only.',
    'sub-02_task-walkstraight_acq-slowrest': 'The S key was held for 12.2 s only; the recording started 2.5 s before.',
    'sub-02_task-walkcurved_acq-slowrest': 'No S key was pressed; the recording started 23 s before key 1. Rest = start '
                                           'of the recording to key 1.',
    'sub-02_task-headshake_acq-slowrest': 'No S key was pressed; the recording started 19 s before key 1. Rest = start '
                                          'of the recording to key 1.',
    'sub-02_task-headnod_acq-slowrest': 'No S key was pressed; the recording started 13 s before key 1. Rest = start '
                                        'of the recording to key 1.',
    'sub-02_task-pickup_acq-slowrest': 'No S key was pressed; the recording started 15 s before key 1. Rest = start of '
                                       'the recording to key 1.',
    'sub-02_task-stairs_acq-slowrest': 'No S key was pressed; the recording started 15 s before key 1. Rest = start of '
                                       'the recording to key 1.',
    'sub-06_task-headshake_acq-fastrest': 'The S key was held for 24.7 s only.'}
TRANSITION = ['sub-01_task-walkcurved_acq-fastrest', 'sub-02_task-walkstraight_acq-fastrest',
              'sub-02_task-headshake_acq-fastrest', 'sub-02_task-headnod_acq-fastrest', 'sub-02_task-pickup_acq-fastrest',
              'sub-02_task-stairs_acq-fastrest']
QUIET_SPAN = 'sub-01_task-walkstraight_acq-slowrest'      # stood still from the start of the recording to key 1
HEAD_MOTION_METHOD = 'CGX accelerometer (ACC1-3) of this file, 0.3-10 Hz, RMS of the 3-axis norm in 1-s windows'


def write_json(path, obj, ensure_ascii=False):
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=ensure_ascii, default=lambda o: o.item())


def write_text(path, text):
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)


def write_tsv(frame, path):
    frame.to_csv(path, sep='\t', index=False, lineterminator='\n')


def load_tables(tables):
    """The tables as data frames; imu_pieces without the tail pieces, tone_onsets without tones that have no EEG sample."""
    S = pd.read_csv(os.path.join(tables, 'imu_segments.csv'), keep_default_na=False)
    P = pd.read_csv(os.path.join(tables, 'imu_pieces.csv'), keep_default_na=False)
    return dict(cut=rs.load_tables(tables), segments=S, pieces=P[P.tail_piece != 'yes'],
                runs=pd.read_csv(os.path.join(tables, 'imu_time_base_runs.csv'), keep_default_na=False),
                tones=pd.read_csv(os.path.join(tables, 'tone_onsets.csv')).dropna(subset=['corrected_eeg_sample']),
                delays=pd.read_csv(os.path.join(tables, 'onset_correction.csv')))


def num(x):
    """Table value -> number (None if empty); integers stay integers unless the column is a float column."""
    if x == '' or x is None:
        return None
    v = float(x)
    return int(v) if v.is_integer() and abs(v) < 1e9 and not isinstance(x, float) else v


def headset_model(stream):
    """'CGX Quick-20r Q20r-0162' (name of the EEG stream in the xdf) -> 'CGX Quick-20'."""
    return re.sub(r'^(CGX Quick-\d+)r .*$', r'\1', stream)


def device_settings(path):
    """The device settings (KEEP_HEADER) written at the top of a rabboni txt file."""
    out = {}
    with open(path, errors='replace') as fh:
        for line in fh:
            if line.startswith('=') and 'DATA START' in line:
                break
            if ':' in line and not line.startswith('='):
                k, v = line.split(':', 1)
                out[k.strip().lstrip('#').strip()] = v.strip()
    return {k: v for k, v in out.items() if k in KEEP_HEADER}


def boundaries(C):
    """(0-based sample, duration s, text) where a new EEG piece starts or the EEG lost samples."""
    marks, cum = [], 0
    gaps = json.loads(C['info'].eeg_gaps) if C['info'].eeg_gaps else []
    for i, (f, p0, p1) in enumerate(C['eeg']):
        if i:
            marks.append((cum, 0.0, 'pause between two pieces, left out'))
        for x, s, lost in gaps:
            if x == f and p0 < s <= p1:
                marks.append((cum + s - p0, lost / FS_EEG, f'EEG packet loss: {lost} samples lost'))
        cum += p1 - p0 + 1
    return sorted(marks)


def place_tones(tones, pieces):
    """(1-based sample of the segment, label, trial) of every tone whose EEG sample lies in one of the EEG pieces."""
    out, offset = [], 0
    for f, first, last in pieces:
        e = tones[(tones.file == f) & (tones.corrected_eeg_sample >= first) & (tones.corrected_eeg_sample <= last)]
        out += [(int(offset + s - first + 1), lab, int(t)) for s, lab, t in zip(e.corrected_eeg_sample, e.label, e.trial)]
        offset += last - first + 1
    return out


def write_eeg(d, stem, arr, names, units, marks, tones):
    """BrainVision files and _events.tsv of a segment."""
    arr.astype('<f4').tofile(os.path.join(d, stem + '_eeg.eeg'))
    head = ['Brain Vision Data Exchange Header File Version 1.0', '; EEG segment cut from the xdf', '',
            '[Common Infos]', 'Codepage=UTF-8', f'DataFile={stem}_eeg.eeg', f'MarkerFile={stem}_eeg.vmrk',
            'DataFormat=BINARY', 'DataOrientation=MULTIPLEXED', f'NumberOfChannels={len(names)}', 'SamplingInterval=2000',
            '', '[Binary Infos]', 'BinaryFormat=IEEE_FLOAT_32', '', '[Channel Infos]']
    head += [f'Ch{i + 1}={n},,1,{u}' for i, (n, u) in enumerate(zip(names, units))]
    with open(os.path.join(d, stem + '_eeg.vhdr'), 'w', encoding='utf-8', newline='\r\n') as fh:
        fh.write('\n'.join(head) + '\n')
    mk = [(p, 0, f'New Segment,,{p},1,0') for p in [1] + [m[0] + 1 for m in marks]]
    mk += [(p, 1, f'Stimulus,{lab},{p},1,0') for p, lab, _ in tones]
    mk.sort(key=lambda m: m[:2])
    lines = ['Brain Vision Data Exchange Marker File, Version 1.0', '', '[Common Infos]', 'Codepage=UTF-8',
             f'DataFile={stem}_eeg.eeg', '', '[Marker Infos]'] + [f'Mk{j}={m[2]}' for j, m in enumerate(mk, 1)]
    with open(os.path.join(d, stem + '_eeg.vmrk'), 'w', encoding='utf-8', newline='\r\n') as fh:
        fh.write('\n'.join(lines) + '\n')
    rows = [(round(p / FS_EEG, 3), round(dur, 3), 'boundary', text) for p, dur, text in marks]
    rows += [(round((p - 1) / FS_EEG, 3), 0.0, lab, str(t)) for p, lab, t in tones]
    E = pd.DataFrame(rows, columns=['onset', 'duration', 'trial_type', 'value']).sort_values('onset', kind='stable')
    write_tsv(E, os.path.join(d, stem + '_events.tsv'))


def artefact(g):
    """SubjectArtefactDescription: the anomaly, else the data note of the segment itself, else n/a."""
    note = g.data_note if g.data_note_extent != 'outside the segment' else ''
    return g.anomaly or note or 'n/a'


def eeg_sidecar(g, sid, task, C, n_eeg, arr, marks):
    return {'TaskName': task, 'TaskDescription': TASK_TEXT[task], 'Instructions': INSTRUCTIONS,
            'SamplingFrequency': FS_EEG,
            'EEGReference': 'A1: left earclip (CGX manual: the A1 earclip holds Reference and Ground); A2 is recorded as '
                            'a channel',
            'EEGGround': 'A1 earclip', 'EEGPlacementScheme': '10-20', 'PowerLineFrequency': 60, 'SoftwareFilters': 'n/a',
            'HardwareFilters': {'CGX amplifier': {'Bandwidth': '0-131 Hz, true DC coupling'}},
            'Manufacturer': 'Cognionics (CGX)', 'ManufacturersModelName': headset_model(C['info'].eeg_stream),
            'EEGChannelCount': n_eeg, 'MiscChannelCount': 4, 'RecordingDuration': round(len(arr) / FS_EEG, 3),
            'RecordingType': 'discontinuous' if marks else 'continuous', 'SubjectArtefactDescription': artefact(g),
            'Source': {'EEGFile': [f'sourcedata/{sid}/eeg/{x}' for x in g.eeg_xdf.split('|')],
                       'EEGFirstSample': int(g.eeg_first_sample), 'EEGLastSample': int(g.eeg_last_sample),
                       'EEGSamples': len(arr), 'FirstSampleLSLTime': num(g.eeg_lsl_first),
                       'LastSampleLSLTime': num(g.eeg_lsl_last)},
            'Processing': 'samples of the xdf EEG stream as stored, cut at the segment; pieces are concatenated',
            'LostSamplesInside': int(g.eeg_lost_samples)}


def motion_sidecar(g, sid, task, pieces, settings, rate, orient):
    corr = {'Type': g.correction, 'Method': g.method}
    if g.correction == 'offset':
        corr.update(Direction=g.offset_direction, ShiftSamples=int(g.offset_samples), ShiftMs=float(g.offset_ms))
    elif g.correction != 'none':
        corr.update(ShiftStartSamples=int(g.shift_start_samples), ShiftEndSamples=int(g.shift_end_samples),
                    ShiftMinSamples=int(g.shift_min_samples), ShiftMaxSamples=int(g.shift_max_samples),
                    LostSamplesInterpolated=int(g.lost_samples_interpolated), SlotsWithoutData=int(g.slots_without_data),
                    RowsLeftOut=int(g.rows_left_out))
    if g.end_fix:
        corr.update(EndFix=g.end_fix, EndSlotsAdded=int(g.end_slots_added))
    corr.update(TailSlotsLeftOut=int(g.tail_slots_dropped))
    lag, low = num(g.best_lag_final), num(g.low_windows_after)
    return {
        'TaskName': task, 'TaskDescription': TASK_TEXT[task], 'Instructions': INSTRUCTIONS,
        'TrackingSystemName': 'rabboni head IMU', 'Manufacturer': 'Sipp (矽譜科技股份有限公司)',
        'ManufacturersModelName': 'rabboni six-axis sensor (InvenSense ICM-20689)',
        'SamplingFrequency': 50, 'SamplingFrequencyEffective': FS_IMU, 'SubjectArtefactDescription': artefact(g),
        'RecordingType': 'continuous' if len(pieces) == 1 else 'discontinuous',
        'RecordingDuration': round(int(g.imu_slots) / FS_IMU, 3),
        'MotionChannelCount': 7, 'ACCELChannelCount': 3, 'GYROChannelCount': 3, 'LATENCYChannelCount': 1,
        'TrackedPointsCount': 1, 'MissingValues': 'n/a', 'SoftwareFilters': 'n/a',
        'ACCELRange': '+/-2 g', 'GYRORange': '+/-1000 deg/s (nominal device setting)',
        'GYROScaleNote': GYRO_NOTE + ' See README.', 'DeviceConfiguration': settings,
        'Segment': {'Subject': g.sub, 'Condition': g.cond, 'Part': g.seg, 'Activity': g.activity, 'Speed': g.speed,
                    'SpeedDefinition': 'slow / fast: self-paced by the participant; the measured rate is in '
                                       'Quality.MeasuredMovementRate',
                    'Pieces': len(pieces), 'Note': 'pieces are concatenated; the pauses between them are left out'},
        'Source': {'IMUFile': [f'sourcedata/{sid}/motion/{t}' for t in g.imu_txt.split('|')],
                   'IMURows': [[p.txt, int(p.final_first_row), int(p.final_last_row)] for p in pieces.itertuples()],
                   'IMURowsReleased': [[p.txt, int(p.released_first_row), int(p.released_last_row)]
                                       for p in pieces.itertuples()],
                   'EEGFile': [f'sourcedata/{sid}/eeg/{x}' for x in g.eeg_xdf.split('|')],
                   'EEGFirstSample': int(g.eeg_first_sample), 'EEGLastSample': int(g.eeg_last_sample),
                   'EEGFirstSampleLSLTime': num(g.eeg_lsl_first), 'EEGLastSampleLSLTime': num(g.eeg_lsl_last),
                   'EEGSamples': int(g.eeg_samples), 'RowNumbers': '1-based data rows of the IMU file (header lines not counted)'},
        'Timing': {'Reference': 'the first EEG sample of the segment (EEG LSL clock)',
                   'LatencyColumn': 'latency of every IMU sample = (u - 1) / 49.647537 s, u = 1-based sample of the segment'},
        'Correction': corr,
        'Quality': {'RhoWithCGXRelease': num(g.rho_release), 'RhoWithCGXFinal': num(g.rho_final),
                    'BestLagFinalSamples': None if lag is None else int(lag),
                    'LowWindows5sAfter': None if low is None else int(low),
                    'IMUUsable': g.imu_usable, 'Anomaly': g.anomaly, 'AnomalyType': g.anomaly_type,
                    'AnomalyExtent': g.anomaly_extent, 'AnomalyPeriodsSeconds': g.anomaly_periods_s,
                    'DataNote': g.data_note, 'DataNoteType': g.data_note_type, 'DataNoteExtent': g.data_note_extent,
                    'DataNotePeriodsSeconds': g.data_note_periods_s, 'MeasuredMovementRate': rate, **orient}}


def orientation(Y):
    """Tilt of the sensor Z axis from gravity (mean acceleration) and the number of clipped gyroscope samples."""
    ok = ~np.isnan(Y[:, 0])
    G = Y[ok, :3].mean(0)
    tilt = float(np.degrees(np.arccos(G[2] / np.linalg.norm(G))))
    return {'IMUTiltFromGravityDeg': float(np.round(tilt, 4)),
            'GyroClippedSamples': int((np.abs(Y[ok, 3:]) >= CLIP).any(1).sum())}


def movement_rate(g, Y, rest):
    """Quality.MeasuredMovementRate of a Move segment: the main measure of imu_movement_rate.py and the intensity;
    for head shaking, nodding and picking up the median of three estimates (cycle_rate.py); for stairs the intensity."""
    m = mr.one_segment(g, Y)
    mv = int(g.cond[8])
    if mv == 5:
        return {'IntensityAccelG': m['intensity_acc_g']}
    v = m['main_measure']
    v = None if v != v else int(v) if float(v).is_integer() else float(v)      # a whole number is written as an integer
    if mv in cr.MOVEMENTS:                         # the median of three estimates, where it differs from the event count
        rate = cr.segment_rates(g, Y, rest)['rate']
        v = rate if rate != v else v
    return {'Value': v, 'Unit': m['main_unit'], 'IntensityAccelG': m['intensity_acc_g']}


def head_motion(acc_ug):
    """Head motion of the headset accelerometer (n, 3; micro-g): 0.3-10 Hz, RMS of the 3-axis norm in 1-s windows ->
    median and 90th percentile (mg)."""
    a = sosfiltfilt(SOS_HEAD, np.asarray(acc_ug, float) / 1e6, axis=0)
    m = [1e3 * np.sqrt(np.mean(np.sum(a[k:k + FS_EEG] ** 2, 1))) for k in range(0, len(a) - FS_EEG + 1, FS_EEG)]
    return float(np.median(m)), float(np.percentile(m, 90))


def run_subject(sub, bids, out, tables):
    """Write the EEG and IMU files of one subject; return the head motion of its Rest segments."""
    T = load_tables(tables)
    S = T['segments'][T['segments']['sub'] == sub]
    P, R = T['pieces'], T['runs']
    G = T['cut']['segments']
    stems = G[G['sub'] == sub].set_index(['cond', 'seg']).bids_name
    sid = 'sub-' + sub[3:]
    imu_raw = {t: rs.read_txt(rs.source_file(bids, sub, t)) for x in S.imu_txt for t in x.split('|') if t}
    settings = {t: device_settings(rs.source_file(bids, sub, t)) for t in imu_raw}
    Y, pcs = {}, {}
    for g in S.itertuples():
        key = (g.cond, g.seg)
        pcs[key] = P[(P['sub'] == sub) & (P.cond == g.cond) & (P.seg == g.seg)]
        if int(g.imu_slots):
            runs = R[(R['sub'] == sub) & (R.cond == g.cond) & (R.seg == g.seg)]
            Y[key] = ic.process_segment(g, pcs[key], runs, imu_raw)
    cuts = {(g.cond, g.seg): rs.cut(T['cut'], sub, g.cond, g.seg) for g in S.itertuples()}
    left = Counter(f for C in cuts.values() for f in {f for f, _, _ in C['eeg']})
    tones = T['tones'][T['tones'].file.isin(list(left))]
    xdf, rest = {}, []
    for g in S.itertuples():
        key = (g.cond, g.seg)
        C = cuts[key]
        stem = stems.loc[key]
        task = stem.split('_task-')[1].split('_')[0]
        files = {f for f, _, _ in C['eeg']}
        for f in files - set(xdf):
            xdf[f] = rs.load_eeg_stream(rs.source_file(bids, sub, f), C['info'].eeg_stream)
        eeg = rs.apply(C, {f: xdf[f][0] for f in files}, None)[0]
        labels = xdf[C['eeg'][0][0]][1]
        a, b = map(int, C['info'].eeg_channels.split('-'))
        c0, c1 = map(int, C['info'].eeg_acc_rows.split('-'))
        arr = np.asarray(eeg[:, list(range(a - 1, b)) + list(range(c0 - 1, c1)) + [labels.index('Packet Counter')]],
                         np.float32)
        for f in files:                            # an xdf is kept in memory while later segments still need it
            left[f] -= 1
            if not left[f]:
                del xdf[f]
        eeg_names = labels[a - 1:b]
        names = eeg_names + ['ACC1', 'ACC2', 'ACC3', 'PacketCounter']
        marks = boundaries(C)
        d = os.path.join(out, sid, 'eeg')
        os.makedirs(d, exist_ok=True)
        write_eeg(d, stem, arr, names, ['µV'] * len(eeg_names) + ['ug'] * 3 + ['n/a'], marks,
                  place_tones(tones, C['eeg']))
        write_tsv(pd.DataFrame(dict(
            name=names, type=['EEG'] * len(eeg_names) + ['MISC'] * 4, units=['uV'] * len(eeg_names) + ['ug'] * 3 + ['n/a'],
            description=['EEG channel, referenced to A1'] * len(eeg_names)
            + ['CGX built-in accelerometer on the EEG clock (micro-g); position not given by the manufacturer'] * 3
            + ['CGX packet counter, wraps at 128'], status=['good'] * len(names))), os.path.join(d, stem + '_channels.tsv'))
        write_json(os.path.join(d, stem + '_eeg.json'), eeg_sidecar(g, sid, task, C, len(eeg_names), arr, marks))
        if g.seg == 'Rest':
            rest.append((stem, *head_motion(arr[:, len(eeg_names):len(eeg_names) + 3]), len(arr) / FS_EEG))
        if key not in Y:
            continue
        d = os.path.join(out, sid, 'motion')
        os.makedirs(d, exist_ok=True)
        stem_m = os.path.basename(g.bids_file)[:-len('_motion.tsv')]
        D = pd.DataFrame(Y[key], columns=COLUMNS)
        D['latency'] = np.round(np.arange(len(D)) / FS_IMU, 6)
        D.to_csv(os.path.join(d, stem_m + '_motion.tsv'), sep='\t', header=False, index=False, na_rep='n/a',
                 lineterminator='\n')
        rate = movement_rate(g, Y[key], Y.get((g.cond, 'Rest'))) if g.seg == 'Move' and g.imu_usable != 'no' else {}
        write_json(os.path.join(d, stem_m + '_motion.json'),
                   motion_sidecar(g, sid, task, pcs[key], {t: settings[t] for t in g.imu_txt.split('|')}, rate,
                                  orientation(Y[key])))
        write_tsv(CHANNELS, os.path.join(d, stem_m + '_channels.tsv'))
        write_json(os.path.join(d, stem_m + '_channels.json'), CH_JSON, ensure_ascii=True)
    print(sub, len(S), 'segments', flush=True)
    return rest


def quiet_span(bids, tables, stem):
    """Samples of the raw recording from its start to key 1 and their head motion."""
    S = rs.load_tables(tables)['segments']
    g = S[S.bids_name == stem].iloc[0]
    rel = f'sourcedata/sub-{g["sub"][3:]}/eeg/{g.eeg_xdf}'
    streams = pyxdf.load_xdf(os.path.join(bids, rel), dejitter_timestamps=False, synchronize_clocks=False)[0]
    eeg = [s for s in streams if s['info']['type'][0] == 'EEG' and s['info']['name'][0] == g.eeg_stream][0]
    keys = [s for s in streams if s['info']['name'][0] == 'Keyboard'][0]
    t1 = [t for v, t in zip(keys['time_series'], keys['time_stamps']) if v[0] == '1 pressed'][0]
    last = int(np.searchsorted(np.asarray(eeg['time_stamps']), t1))        # samples before key 1
    r0, r1 = map(int, g.eeg_acc_rows.split('-'))
    med, p90 = head_motion(np.asarray(eeg['time_series'], float)[:last, r0 - 1:r1])
    return {'Description': 'The participant stood still from the start of the recording until key 1, longer than this '
                           'file (which covers the S key only). To use the whole quiet span, cut these samples of the raw '
                           'recording.',
            'EEGFile': rel, 'EEGFirstSample': 1, 'EEGLastSample': last, 'Seconds': round((last - 1) / FS_EEG, 1),
            'HeadMotionMedianMg': round(med, 1), 'HeadMotionPercentile90Mg': round(p90, 1)}


def add_rest_notes(out, bids, tables, rest):
    """RestNote in the sidecars of the short Rest files and of the transitions between speeds. rest = (stem, median mg,
    90th percentile mg, duration s) of every Rest file; the median over all of them is the reference."""
    stats = {r[0]: r[1:] for r in rest}
    ref = float(np.median([r[1] for r in rest]))
    for stem in list(SHORT_REST) + TRANSITION:
        med, p90, dur = stats[stem]
        motion = {'MedianMg': round(med, 1), 'Percentile90Mg': round(p90, 1), 'MedianOfAllRestFilesMg': round(ref, 1),
                  'Method': HEAD_MOTION_METHOD}
        artefact_text = None
        if stem in SHORT_REST:
            note = {'Type': 'short rest', 'DurationSeconds': round(dur, 1),
                    'Description': 'Standing still, but shorter than the planned 30 s. ' + SHORT_REST[stem],
                    'HeadMotion': motion}
            if stem == QUIET_SPAN:
                note['QuietSpan'] = quiet_span(bids, tables, stem)
        else:
            moved = med > 2 * ref
            note = {'Type': 'transition between speeds', 'DurationSeconds': round(dur, 1),
                    'Description': 'No rest was recorded between the two speeds and no S key was pressed. This file is the '
                                   'time between releasing the key of the slow speed (1) and pressing the key of the fast '
                                   'speed (2). ' + ('The head moved during it (head motion more than twice the median of '
                                                    'the Rest files).' if moved else
                                                    'Head motion is slightly above the median of the Rest files.'),
                    'HeadMotion': motion}
            artefact_text = ('Not a planned rest: transition between the slow and the fast speed'
                             + (', with movement' if moved else '') + '; see RestNote.')
        sid = stem[:6]
        task, acq = stem.split('_task-')[1].split('_acq-')
        for p in (os.path.join(out, sid, 'eeg', stem + '_eeg.json'),
                  os.path.join(out, sid, 'motion', f'{sid}_task-{task}_tracksys-imu_acq-{acq}_motion.json')):
            if not os.path.exists(p):
                continue
            with open(p, encoding='utf-8') as fh:
                j = json.load(fh)
            j['RestNote'] = note
            if artefact_text:
                j['SubjectArtefactDescription'] = artefact_text
            write_json(p, j)
    return ref


def events_json(delays):
    return {
        'onset': {'Description': 'Time of the event from the first sample of this EEG file.', 'Units': 's'},
        'duration': {'Description': 'Duration of the event (0: an instant).', 'Units': 's'},
        'trial_type': {
            'Description': 'Event type.',
            'Levels': {
                'standard': 'Onset of a standard (frequent) tone of the auditory oddball task.',
                'target': 'Onset of a target (rare) tone of the auditory oddball task.',
                'boundary': 'Place where two cut pieces of the recording meet, or where the EEG lost samples; the EEG is '
                            'not continuous across it.'}},
        'value': {'Description': 'For standard and target: the trial number written in the audio marker of the raw '
                                 'recording. For boundary: what happens at that place.'},
        'StimulusPresentation': {
            'OnsetCorrection': (
                'The LSL audio markers arrive after the sounds by a constant delay that depends on the laptop that played '
                'them. Onsets here are the corrected sound onsets: marker time minus that delay, placed on the EEG sample '
                'clock reconstructed from the CGX packet counter and the LSL arrival times. Delay per audio set-up, '
                'estimated from the auditory N1 (about 100 ms after the sound): '
                + ', '.join(f'{r.setup} {r.c_ms} ms' for r in delays.itertuples()) + '.'),
            'Note': 'Tones played between the cut segments are not in any file.'}}


def write_root_files(out, tables):
    T = load_tables(tables)
    write_text(os.path.join(out, 'README'), README)
    write_text(os.path.join(out, 'CHANGES'), CHANGES)
    write_text(os.path.join(out, '.bidsignore'), BIDSIGNORE)
    write_json(os.path.join(out, 'dataset_description.json'),
               {'Name': NAME, 'BIDSVersion': '1.10.0', 'DatasetType': 'raw', 'License': 'CC0', 'Authors': AUTHORS})
    G = T['cut']['segments']
    for task in sorted({n.split('_task-')[1].split('_')[0] for n in G.bids_name}):
        write_json(os.path.join(out, f'task-{task}_events.json'), events_json(T['delays']))
    S = T['segments']
    A = S[S.anomaly_type != ''][['sub', 'cond', 'seg', 'bids_file', 'imu_usable', 'anomaly_type', 'anomaly_extent',
                                 'anomaly_periods_s', 'anomaly']].copy()
    A['bids_file'] = [b if int(n) else 'n/a' for b, n in zip(A.bids_file, S.loc[A.index, 'imu_slots'])]
    write_tsv(A.replace('', 'n/a'), os.path.join(out, 'anomalies.tsv'))
    write_tsv(pd.DataFrame(MISSING_RECORDINGS), os.path.join(out, 'missing_recordings.tsv'))
    Q = []
    for sub, x in G.groupby('sub'):
        sid = 'sub-' + sub[3:]
        ch = x.eeg_channels.str.split('-').str[1].astype(int)
        Q.append(dict(participant_id=sid, age=AGE[sid],
                      baseline_posture=x[x.cond.str.lower().isin(['sit', 'stand'])].cond.str.lower().iloc[0],
                      eeg_channels='/'.join(str(v) for v in sorted(ch.unique())),
                      eeg_30_channel_recordings=int((ch == 30).sum()),
                      eeg_device='; '.join(sorted({headset_model(s) for s in x.eeg_stream})),
                      missing_recordings=MISSING.get(sid, 'n/a')))
    write_tsv(pd.DataFrame(Q), os.path.join(out, 'participants.tsv'))
    write_json(os.path.join(out, 'participants.json'), PARTICIPANTS_JSON)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids', required=True, help='dataset root with sourcedata/')
    ap.add_argument('--out', required=True, help='output folder (new or empty)')
    ap.add_argument('--tables', default=str(TABLES), help='folder of the tables (default: tables/ of this repository)')
    ap.add_argument('--workers', type=int, default=3, help='subjects processed in parallel')
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f'{out} is not empty')
    os.makedirs(out, exist_ok=True)
    subjects = sorted(pd.read_csv(os.path.join(a.tables, 'imu_segments.csv'), usecols=['sub'])['sub'].unique())
    with ProcessPoolExecutor(a.workers) as ex:
        rest = [r for x in ex.map(partial(run_subject, bids=a.bids, out=out, tables=a.tables), subjects) for r in x]
    ref = add_rest_notes(out, a.bids, a.tables, rest)
    write_root_files(out, a.tables)
    n = sum(len(f) for _, _, f in os.walk(out))
    print(f'{n} files in {out}; head motion of the {len(rest)} Rest files: median {ref:.1f} mg')


if __name__ == '__main__':
    main()
