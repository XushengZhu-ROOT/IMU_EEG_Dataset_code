function clean_paper_versions(bids, outroot, eeglab_dir, version, k, subjects)
% Clean the EEG of the Move segments of the BIDS dataset, one version per call.
%   clean_paper_versions(bids, outroot, eeglab_dir, version, k, subjects)
%   version   'hp_notch'    0.5 Hz high-pass and band-stop filters at 60, 120, 180 and 240 Hz
%             'imureg'      the filters, then the regression on the head motion
%             'asr'         the filters, then ASR with cutoff k
%             'imureg_asr'  the filters, the regression, then ASR with cutoff k (derivatives/clean: k = 30)
%   k         ASR cutoff (k0 in the output folder of 'hp_notch' and 'imureg')
%   subjects  optional cell array of file-name beginnings, e.g. {'sub-01'} or {'sub-04_task-pickup_acq-fastmove'}
% Writes outroot/<version>/k<k>/sub-XX/<Move file>.mat with data (channels x samples, single, uV, 500 Hz, A1
% reference as recorded), labels, bad (channels left out of ASR), srate, k, cal_file, cal_reason, cal_seconds.

% EEGLAB of eeglab_dir: every EEGLAB installation is taken off the path (the repository folders stay on it), then
% eeglab_dir is added
repo = fileparts(fileparts(mfilename('fullpath')));
for w = reshape(which('eeglab', '-all'), 1, [])
    root = fileparts(w{1});
    p = strsplit(path, pathsep);
    p = p((strcmp(p, root) | startsWith(p, [root filesep])) & ~startsWith(p, [repo filesep]));
    if ~isempty(p), rmpath(p{:}); end
end
addpath(eeglab_dir); eeglab('nogui');
if ~startsWith(which('asr_calibrate'), [fileparts(which('eeglab')) filesep])
    error('clean_rawdata of the EEGLAB in %s is not on the path', eeglab_dir);
end
fs = 500; MIN_CAL = 20;
ASR_MAXMEM = 1e6;   % asr_calibrate maxmem (units of 2^21 bytes): large, so that it averages covariances in blocks of
                    % 10 samples on every computer
useimu = any(strcmp(version, {'imureg', 'imureg_asr'})); useasr = any(strcmp(version, {'asr', 'imureg_asr'}));
if ~useasr, k = 0; end
lags = -50:5:50;                            % regression lags in samples: -100..+100 ms in 10-ms steps
mv = dir(fullfile(bids, 'sub-*', 'eeg', '*move_eeg.vhdr'));
if nargin >= 6 && ~isempty(subjects), mv = mv(cellfun(@(f) any(startsWith(f, subjects)), {mv.name})); end
for i = 1:numel(mv)
    movef = fullfile(mv(i).folder, mv(i).name);
    stem = mv(i).name(1:end - numel('_eeg.vhdr')); sub = stem(1:6);
    % ASR calibration data: the Rest of this condition, or the sitting/standing baseline when that Rest is shorter
    % than 20 s or is a transition between speeds
    restf = strrep(movef, 'move_eeg.vhdr', 'rest_eeg.vhdr');
    b = dir(fullfile(mv(i).folder, '*task-rest_acq-*_eeg.vhdr')); basef = fullfile(b(1).folder, b(1).name);
    j = jsondecode(fileread(strrep(restf, '_eeg.vhdr', '_eeg.json')));
    rest_ok = j.RecordingDuration >= MIN_CAL && ~(isfield(j, 'RestNote') && ...
              strcmp(j.RestNote.Type, 'transition between speeds'));
    if rest_ok, calf = restf; why = 'Rest of this condition';
    else, calf = basef; why = 'baseline (Rest < 20 s or transition)'; end
    [lab, Xm, Am] = read_bv(movef);
    [lc, Xc, Ac] = read_bv(calf);
    if ~isequal(lc, lab), error('%s: calibration file has other channels', stem); end
    % high-pass and band-stop filters, Move and calibration data
    x = prep(Xm, fs, true); c = prep(Xc, fs, true);
    % regression on the head motion, the Move data and the calibration data each with its own motion signals
    if useimu
        x = acc_regress(x, motion(Am, movef, size(Xm, 2), fs), lags);
        c = acc_regress(c, motion(Ac, calf, size(Xc, 2), fs), lags);
    end
    % ASR, calibrated on the calibration data; channels flat for 5 s there pass through ASR unchanged
    bad = flat_channels(c, lab, fs);
    y = x;
    if useasr
        state = asr_calibrate(c(~bad, :), fs, k, [], [], [], [], [], [], [], ASR_MAXMEM);
        y(~bad, :) = run_asr(x(~bad, :), fs, state);
    end
    d = fullfile(outroot, version, sprintf('k%d', k), sub);
    if ~exist(d, 'dir'), mkdir(d); end
    [~, cn, ce] = fileparts(calf);
    out = struct('data', single(y), 'labels', {lab}, 'bad', bad, 'srate', fs, 'k', k, 'cal_file', [cn ce], ...
                 'cal_reason', why, 'cal_seconds', size(c, 2) / fs);
    save(fullfile(d, [stem '.mat']), '-struct', 'out', '-v7');
    fprintf('%s done\n', stem);
end
end

function [lab, X, A] = read_bv(vhdr)
% EEG channels (type EEG in _channels.tsv) and ACC1-3 of a BrainVision file, channels x samples, double
h = fileread(vhdr);
tok = regexp(h, '(?m)^Ch\d+=([^,]+),', 'tokens');
all_lab = cellfun(@(t) t{1}, tok, 'uni', 0);
fid = fopen([vhdr(1:end - 5) '.eeg'], 'r', 'ieee-le');
D = fread(fid, [numel(all_lab), Inf], 'float32=>double'); fclose(fid);
ch = readtable(strrep(vhdr, '_eeg.vhdr', '_channels.tsv'), 'FileType', 'text', 'Delimiter', '\t', ...
               'TextType', 'char');
lab = ch.name(strcmp(ch.type, 'EEG'));
[~, idx] = ismember(lab, all_lab); X = D(idx, :);
[~, ia] = ismember({'ACC1', 'ACC2', 'ACC3'}, all_lab); A = D(ia, :);
end

function x = prep(x, fs, notch)
% 0.5 Hz high-pass (Butterworth order 4, zero phase); with notch also band-stop filters of +-1 Hz at 60, 120, 180
% and 240 Hz (butter with n = 2, a fourth-order band-stop; zero phase)
[z, p, k] = butter(4, 0.5 / (fs / 2), 'high'); [sos, g] = zp2sos(z, p, k);
x = filtfilt(sos, g, x')';
if notch
    for f0 = [60 120 180 240]
        [z, p, k] = butter(2, [f0 - 1, f0 + 1] / (fs / 2), 'stop'); [sos, g] = zp2sos(z, p, k);
        x = filtfilt(sos, g, x')';
    end
end
end

function A = motion(acc, eegvhdr, N, fs)
% motion signals: headset accelerometer (ACC1-3) and head IMU (read_imu), 0.5 Hz high-pass. IMU n/a is filled
% linearly for the filter and set back to NaN after it
A = [acc; read_imu(eegvhdr, N, fs)];
gap = isnan(A);
if any(gap(:)), A = fillmissing(A, 'linear', 2, 'EndValues', 'nearest'); end
A = prep(A, fs, false);
A(gap) = NaN;
end

function M = read_imu(eegvhdr, N, fs)
% head IMU accelerometer and gyroscope of the same segment (BIDS motion file) at the EEG samples, with the clipped
% gyroscope peaks restored (declip_gyro.m); IMU n/a stays NaN. Empty when the segment has no motion file or
% anomalies.tsv marks its IMU unusable; the regression then uses the headset accelerometer alone
[d, f] = fileparts(eegvhdr);
tok = regexp(f, '^sub-(\d+)_task-([a-z]+)_acq-([a-z]+)_eeg$', 'tokens', 'once');      % sub, task, acq
mf = fullfile(fileparts(d), 'motion', sprintf('sub-%s_task-%s_tracksys-imu_acq-%s_motion.tsv', tok{:}));
M = zeros(0, N);
if ~exist(mf, 'file') || imu_unusable(fileparts(fileparts(d)), tok{:}), return, end
V = readmatrix(mf, 'FileType', 'text', 'Delimiter', '\t', 'TreatAsMissing', 'n/a');
lat = V(:, 7); V = V(:, 1:6);
for g = 4:6, V(:, g) = declip_gyro(V(:, g)); end
t = (0:N - 1) / fs;
M = interp1(lat, V, t, 'linear', 'extrap')';
M(:, t > lat(end)) = repmat(V(end, :)', 1, sum(t > lat(end)));
end

function x = acc_regress(x, acc, lags)
% least-squares regression of every EEG channel on the motion signals (rows of acc) at the given lags (samples),
% fitted part subtracted. Where regressors are NaN (IMU n/a), the fit uses the other samples and the subtraction
% uses the regressors with data
N = size(x, 2); na = size(acc, 1);
Z = zeros(N, na * numel(lags));
col = 0;
for a = 1:na
    for L = lags
        col = col + 1;
        if L >= 0, Z(1 + L:end, col) = acc(a, 1:end - L)'; else, Z(1:end + L, col) = acc(a, 1 - L:end)'; end
    end
end
if ~any(isnan(Z(:)))
    Z = Z ./ max(std(Z), eps);
    x = x - (Z * (Z \ x'))';
else
    Z = Z ./ max(std(Z, 0, 1, 'omitnan'), eps);
    ok = ~any(isnan(Z), 2);
    B = Z(ok, :) \ x(:, ok)';
    Z(isnan(Z)) = 0;
    x = x - (Z * B)';
end
end

function bad = flat_channels(x, lab, fs)
% channels flat for 5 s in x (clean_rawdata clean_flatlines)
EEG = eeg_emptyset; EEG.data = x; EEG.srate = fs; EEG.nbchan = size(x, 1); EEG.pnts = size(x, 2);
EEG.trials = 1; EEG.xmin = 0; EEG.xmax = (EEG.pnts - 1) / fs;
for i = 1:numel(lab), EEG.chanlocs(i).labels = lab{i}; end
EEG = eeg_checkset(EEG);
E = clean_flatlines(EEG, 5);
bad = ~ismember({EEG.chanlocs.labels}', {E.chanlocs.labels});
end

function y = run_asr(x, fs, state)
% ASR (clean_rawdata asr_process): window 0.5 s, look-ahead 0.25 s, step 0.25 s, maxdims 0.66, 64 MB. The end is
% padded with 0.25 s point-mirrored about the last sample (2*x(end) - x(end-1:-1:end-N)) so that the last samples
% are processed; the first 0.25 s of the output (the look-ahead delay) is removed
N = round(0.5 / 2 * fs);
sig = [x bsxfun(@minus, 2 * x(:, end), x(:, (end - 1):-1:end - N))];
[y, st] = asr_process(sig, fs, state, 0.5, 0.25, floor(fs * 0.25), 0.66, 64, false);
y(:, 1:size(st.carry, 2)) = [];
end
