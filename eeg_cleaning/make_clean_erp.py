"""Write the cleaned EEG of the Move segments (clean_paper_versions.m, version imureg_asr, k = 30) as the BIDS
derivative derivatives/clean.

    python eeg_cleaning/make_clean_erp.py --bids <dataset root> --mat <OUT of run_all.m>/imureg_asr/k30 \
        --out <new folder> [--report <folder>]

Per Move segment, in <out>/sub-XX/eeg/ with the raw file name + '_desc-clean': BrainVision _eeg.eeg/.vhdr/.vmrk
(float32, with the channels, samples, time base and events of the raw file), _events.tsv (copied from the raw file),
_channels.tsv (channels left out of ASR marked bad) and _eeg.json (filters, ASR calibration data, motion signals of
the regression). At <out>: dataset_description.json, README and the task-<task>_events.json files of the raw
dataset. Each written .eeg is read back and compared with the .mat data. --report writes clean_erp_files.csv.
"""
import argparse
import glob
import json
import os
import re
import shutil

import numpy as np
import pandas as pd
from scipy.io import loadmat

FS = 500
GYRO_CLIP = 990      # |gyroscope value| counted as clipped, as in declip_gyro.m
TASKS = ['walkstraight', 'walkcurved', 'headshake', 'headnod', 'pickup', 'stairs']
SPEEDS = ['slow', 'fast']
PIPELINE = (
    'MATLAB R2024a, EEGLAB 2025.1.0 (clean_rawdata), clean_paper_versions.m (version imureg_asr, k = 30), on the raw '
    'EEG of the Move segment and of its ASR calibration data: 0.5 Hz high-pass (Butterworth order 4, zero phase); '
    'band-stop filters at 60, 120, 180 and 240 Hz (Butterworth, fourth order: MATLAB butter with n = 2; +-1 Hz; zero '
    'phase); least-squares regression of every EEG channel on the motion signals at lags of -100 to +100 ms in 10-ms '
    'steps, the motion signals being the headset accelerometer (ACC1-3) and the head IMU accelerometer and gyroscope '
    'of the motion file (interpolated to the EEG samples; clipped gyroscope peaks restored with a cubic spline; IMU '
    'marked unusable in anomalies.tsv not used; IMU n/a samples left out of the fit), all high-passed at 0.5 Hz; '
    'Artifact Subspace Reconstruction (asr_calibrate with a fixed memory argument, then asr_process; cutoff 30, '
    'window 0.5 s, step 0.25 s, maxdims 0.66) calibrated on the Rest recorded just before the movement, or on the '
    'sitting/standing baseline when that Rest is shorter than 20 s or is a transition between speeds (ASRCalibration '
    'in each _eeg.json). Channels flat for 5 s in the calibration data are passed through unchanged by ASR and marked '
    'bad in _channels.tsv. All channels and samples are kept; the reference is the A1 earclip as recorded.')


def read_json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def write_json(path, obj):
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def write_brainvision(d, stem, x, names):
    x.T.astype('<f4').tofile(os.path.join(d, stem + '_eeg.eeg'))
    hdr = ['Brain Vision Data Exchange Header File Version 1.0', '', '[Common Infos]', 'Codepage=UTF-8',
           f'DataFile={stem}_eeg.eeg', f'MarkerFile={stem}_eeg.vmrk', 'DataFormat=BINARY', 'DataOrientation=MULTIPLEXED',
           f'NumberOfChannels={len(names)}', f'SamplingInterval={int(1e6 / FS)}', '', '[Binary Infos]',
           'BinaryFormat=IEEE_FLOAT_32', '', '[Channel Infos]']
    hdr += [f'Ch{i + 1}={n},,1,µV' for i, n in enumerate(names)]
    with open(os.path.join(d, stem + '_eeg.vhdr'), 'w', encoding='utf-8', newline='\r\n') as fh:
        fh.write('\n'.join(hdr) + '\n')


def write_vmrk(d, stem, ev):
    mk = ['Brain Vision Data Exchange Marker File, Version 1.0', '', '[Common Infos]', 'Codepage=UTF-8',
          f'DataFile={stem}_eeg.eeg', '', '[Marker Infos]', 'Mk1=New Segment,,1,1,0']
    for _, e in ev.iterrows():
        p = int(np.floor(float(e.onset) * FS + 0.5)) + 1
        mk.append(f'Mk{len(mk) - 6}=' + (f'New Segment,,{p},1,0' if e.trial_type == 'boundary'
                                          else f'Stimulus,{e.trial_type},{p},1,0'))
    with open(os.path.join(d, stem + '_eeg.vmrk'), 'w', encoding='utf-8', newline='\r\n') as fh:
        fh.write('\n'.join(mk) + '\n')


def unusable_imu(anom, sub, task, acq):
    """imu_usable = no in anomalies.tsv for this segment (matched on sub, cond and seg, as imu_unusable.m)."""
    for r in anom.itertuples():
        if r.imu_usable != 'no' or int(r.sub[3:]) != int(sub):
            continue
        m = re.match(r'movement(\d)_speed(\d)', r.cond)
        tk, aq = ('rest', r.cond.lower()) if not m else (TASKS[int(m[1])], SPEEDS[int(m[2])] + r.seg.lower())
        if tk == task and aq == acq:
            return True
    return False


def restored_by_declip(x):
    """True at the samples of one gyroscope axis that declip_gyro.m replaces: runs of clipped values of one sign
    with three unclipped, non-n/a samples right before and right after the run."""
    clipped = np.abs(x) >= GYRO_CLIP
    usable = ~clipped & ~np.isnan(x)
    out = np.zeros(len(x), bool)
    i = 0
    while i < len(x):
        if not clipped[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(x) and clipped[j + 1] and np.sign(x[j + 1]) == np.sign(x[i]):
            j += 1
        if i >= 3 and usable[i - 3:i].all() and j + 3 < len(x) and usable[j + 1:j + 4].all():
            out[i:j + 1] = True
        i = j + 1
    return out


def motion_info(bids, anom, stem):
    """the motion signals of the regression for one EEG file (stem without _eeg)"""
    sub, task, acq = re.match(r'sub-(\d+)_task-([a-z]+)_acq-([a-z]+)$', stem).groups()
    mf = os.path.join(bids, f'sub-{sub}', 'motion', f'sub-{sub}_task-{task}_tracksys-imu_acq-{acq}_motion.tsv')
    if not os.path.exists(mf):
        return {'Signals': 'headset accelerometer ACC1-3 only (no motion file)'}
    if unusable_imu(anom, sub, task, acq):
        return {'Signals': 'headset accelerometer ACC1-3 only (IMU marked unusable in anomalies.tsv)'}
    v = pd.read_csv(mf, sep='\t', header=None, na_values='n/a').to_numpy(float)
    restored = np.column_stack([restored_by_declip(v[:, a]) for a in (3, 4, 5)])
    return {'Signals': 'headset accelerometer ACC1-3 and head IMU accelerometer and gyroscope',
            'GyroClippedSamplesRestored': int(restored.any(axis=1).sum()),
            'IMUSamplesNotAvailable': int(np.isnan(v[:, :6]).any(axis=1).sum())}


def main():
    ap = argparse.ArgumentParser(description='Write the cleaned Move segments as the BIDS derivative '
                                             'derivatives/clean.')
    ap.add_argument('--bids', required=True, help='BIDS dataset root')
    ap.add_argument('--mat', required=True, help='clean_paper_versions.m output of version imureg_asr, k = 30')
    ap.add_argument('--out', required=True, help='output folder, new or empty')
    ap.add_argument('--report', help='folder for clean_erp_files.csv, one row per segment')
    a = ap.parse_args()
    if os.path.exists(a.out) and (not os.path.isdir(a.out) or os.listdir(a.out)):
        raise SystemExit(f'{a.out} exists and is not an empty folder')
    files = sorted(glob.glob(os.path.join(a.mat, 'sub-*', '*move.mat')))
    if not files:
        raise SystemExit(f'no sub-*/*move.mat files in {a.mat}')
    raw_desc = read_json(os.path.join(a.bids, 'dataset_description.json'))
    anom = pd.read_csv(os.path.join(a.bids, 'anomalies.tsv'), sep='\t', dtype=str)
    rows, tasks = [], set()
    for f in files:
        stem = os.path.basename(f)[:-4]
        subdir = stem[:6]
        rawbase = os.path.join(a.bids, subdir, 'eeg', stem)
        m = loadmat(f)
        x = m['data']
        names = [str(np.squeeze(n)) for n in np.ravel(m['labels'])]
        bad = np.ravel(m['bad']).astype(bool)
        rch = pd.read_csv(rawbase + '_channels.tsv', sep='\t', dtype=str)
        if names != rch.name[rch.type == 'EEG'].tolist():
            raise SystemExit(f'{stem}: channels differ from the raw file')
        nraw = os.path.getsize(rawbase + '_eeg.eeg') // (4 * len(rch))
        if x.shape != (len(names), nraw) or x.dtype != np.float32:
            raise SystemExit(f'{stem}: data {x.shape} {x.dtype}, raw has {nraw} samples')
        d = os.path.join(a.out, subdir, 'eeg')
        os.makedirs(d, exist_ok=True)
        cstem = stem + '_desc-clean'
        write_brainvision(d, cstem, x, names)
        back = np.fromfile(os.path.join(d, cstem + '_eeg.eeg'), '<f4').reshape(-1, len(names)).T
        if not np.array_equal(back, x):
            raise SystemExit(f'{stem}: written data differ from the .mat')
        shutil.copyfile(rawbase + '_events.tsv', os.path.join(d, cstem + '_events.tsv'))
        ev = pd.read_csv(rawbase + '_events.tsv', sep='\t', dtype=str, keep_default_na=False)
        write_vmrk(d, cstem, ev)
        ch = pd.DataFrame(dict(name=names, type='EEG', units='uV',
                               description='EEG channel, reference A1 as recorded, cleaned (see _eeg.json)',
                               status=np.where(bad, 'bad', 'good'),
                               status_description=np.where(bad, 'flat for 5 s in the ASR calibration data; '
                                                                'passed through ASR unchanged', 'n/a')))
        ch.to_csv(os.path.join(d, cstem + '_channels.tsv'), sep='\t', index=False, lineterminator='\n')
        rj = read_json(rawbase + '_eeg.json')
        sj = {k: rj[k] for k in ('TaskName', 'TaskDescription', 'Instructions') if k in rj}
        calf = os.path.basename(str(np.squeeze(m['cal_file'])))
        mot = motion_info(a.bids, anom, stem)
        cal_mot = motion_info(a.bids, anom, calf[:-9])['Signals']   # calibration data: own motion signals
        sj.update(SamplingFrequency=FS, EEGReference=rj.get('EEGReference', 'A1'), PowerLineFrequency=60,
                  SoftwareFilters={'ERP-preserving cleaning': {'Description': PIPELINE}},
                  EEGChannelCount=len(names), RecordingDuration=round(nraw / FS, 3), RecordingType='continuous',
                  Sources=[f'bids:raw:{subdir}/eeg/{stem}_eeg.eeg'],
                  ASRCalibration={'File': f'bids:raw:{subdir}/eeg/{calf[:-5]}.eeg',
                                  'Reason': str(np.squeeze(m['cal_reason'])),
                                  'Seconds': round(float(np.squeeze(m['cal_seconds'])), 3),
                                  'Cutoff': int(np.squeeze(m['k']))},
                  MotionRegression=dict(mot, Lags='-100 to +100 ms in 10-ms steps', CalibrationSignals=cal_mot),
                  Description=('The raw file in Sources after the cleaning in SoftwareFilters: same channels, same '
                               'samples, same time base (sample j, 0-based, at j / 500 s from the first sample of '
                               'the raw file). Events are the raw file\'s.'))
        write_json(os.path.join(d, cstem + '_eeg.json'), sj)
        task = re.search(r'task-([a-z]+)', stem).group(1)
        tasks.add(task)
        rows.append(dict(file=cstem, channels=len(names), samples=nraw, bad_channels=';'.join(np.array(names)[bad]),
                         calibration=calf[:-5], calibration_reason=str(np.squeeze(m['cal_reason'])),
                         motion=mot['Signals'], calibration_motion=cal_mot,
                         gyro_clipped_restored=mot.get('GyroClippedSamplesRestored', 0),
                         imu_na=mot.get('IMUSamplesNotAvailable', 0),
                         tones=int(ev.trial_type.isin(['standard', 'target']).sum()),
                         boundaries=int((ev.trial_type == 'boundary').sum())))
    for t in sorted(tasks):
        shutil.copyfile(os.path.join(a.bids, f'task-{t}_events.json'), os.path.join(a.out, f'task-{t}_events.json'))
    write_json(os.path.join(a.out, 'dataset_description.json'), {
        'Name': 'ERP-preserving cleaned EEG of the movement segments',
        'BIDSVersion': raw_desc['BIDSVersion'], 'DatasetType': 'derivative',
        'GeneratedBy': [{'Name': 'clean_paper_versions.m', 'Version': 'MATLAB R2024a, EEGLAB 2025.1.0',
                         'Description': PIPELINE},
                        {'Name': 'make_clean_erp.py', 'Description': 'Wrote the cleaned Move segments as BIDS: EEG '
                         'channels only, raw time base, events from the raw files.'}],
        'DatasetLinks': {'raw': '../../'}})
    with open(os.path.join(a.out, 'README'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(f'Cleaned EEG of the {len(files)} movement segments (Move only) that keeps event-related potentials '
                 '(the P3 included).\n\n'
                 + PIPELINE + '\n\nEach file has the same channels, samples and time base as its raw file (Sources '
                 'in _eeg.json): sample j at j / 500 s. Events are the raw file\'s events (corrected sound onsets and '
                 'boundaries). The accelerometer, packet counter and IMU are in the raw files only.\n')
    if a.report:
        os.makedirs(a.report, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.report, 'clean_erp_files.csv'), index=False)
    print(f'{len(rows)} segments written to {a.out}')


if __name__ == '__main__':
    main()
