%DEMO_RESYNC  Cut one segment from the raw recordings of the dataset with the cut tables and check its alignment.
%
%   Set BIDS (root folder of the dataset) and EEGLAB (for load_xdf of its xdfimport plugin), then run.
%   Continuous EEG can be processed before the cut, as long as every sample keeps its column.

BIDS = '/path/to/dataset';
EEGLAB = '/path/to/eeglab';
if startsWith(BIDS, '/path/to') || startsWith(EEGLAB, '/path/to')
    error('demo_resync:paths', 'Set BIDS and EEGLAB at the top of demo_resync.m.');
end
addpath(fileparts(mfilename('fullpath')));
if isempty(which('load_xdf'))
    d = dir(fullfile(EEGLAB, 'plugins', 'xdfimport*', 'xdf', 'load_xdf.m'));
    if isempty(d)
        error('demo_resync:load_xdf', 'load_xdf.m not found in %s/plugins/xdfimport*/xdf.', EEGLAB);
    end
    addpath(d(1).folder);
end

sub = 'Sub01'; cond = 'movement3_speed0'; seg = 'Move';

% positions of the segment in the raw files
T = resync_tables();
C = resync_cut(T, sub, cond, seg);
fprintf('EEG %s samples %d .. %s %d (%d piece(s))\n', C.eeg_start.file, C.eeg_start.sample, C.eeg_end.file, ...
    C.eeg_end.sample, height(C.eeg));
fprintf('IMU %s rows %d .. %s %d (%d slots, %s; IMU usable: %s)\n', C.imu_start.file, C.imu_start.row, ...
    C.imu_end.file, C.imu_end.row, height(C.imu), C.info.imu_method, C.info.imu_usable);

% continuous data: the EEG stream of the xdf [rows x samples], the rows of the raw IMU txt
eeg_src = containers.Map();
for f = unique(C.eeg.file)'
    eeg_src(char(f)) = load_xdf_eeg(raw_file(BIDS, sub, f), C.info.eeg_stream);
end
imu_src = containers.Map();
for f = unique(C.imu.file(C.imu.file ~= ""))'
    imu_src(char(f)) = read_rabboni_txt(raw_file(BIDS, sub, f));
end

% cut
[eeg, imu] = resync_apply(C, eeg_src, imu_src);
ch = str2double(split(C.info.eeg_channels, '-'));
ac = str2double(split(C.info.eeg_acc_rows, '-'));
eeg_channels = eeg(ch(1):ch(2), :);                  % EEG channels (uV)
acc = double(eeg(ac(1):ac(2), :)) / 1e6;             % CGX accelerometer (g)
t_eeg = (0:size(eeg, 2) - 1) / 500;                  % s from the segment's first EEG sample
t_imu = (0:size(imu, 1) - 1)' / 49.647537;           % s from the same sample
fprintf('cut: EEG %d x %d (%.2f s), IMU %d x 6 (%.2f s)\n', size(eeg), t_eeg(end), size(imu, 1), t_imu(end));

% alignment of the IMU with the CGX accelerometer
A = resync_alignment(acc', imu(:, 1:3));
fprintf('rho %.3f (table %.3f), best lag %+d; 5-s windows below 0.5: %d\n', A.rho, C.info.rho, A.best_lag, ...
    sum(A.windows.rho < 0.5));


function p = raw_file(bids, sub, name)
% raw recording in the dataset: sourcedata/sub-XX/eeg/<name>.xdf or sourcedata/sub-XX/motion/<name>.txt
kind = "motion";
if endsWith(name, ".xdf"), kind = "eeg"; end
p = fullfile(bids, 'sourcedata', "sub-" + extractAfter(string(sub), 3), kind, name);
end
