function R = verify_resync(bids, varargin)
%VERIFY_RESYNC  Rebuild every segment from the raw recordings and check it against the cut tables.
%
%   R = verify_resync(bids)                          bids = root folder of the dataset
%   R = verify_resync(bids, 'Out', 'verify_out', 'Tables', '')
%
%   Needs load_xdf (EEGLAB plugin xdfimport). Checks: 1 positions (start, end, length), 2 values at the piece ends,
%   3 alignment (rho, best lag). Writes <Out>/verify_segments.csv, one row per segment; Out must be empty or new.

ip = inputParser;
ip.addParameter('Tables', '');
ip.addParameter('Out', 'verify_out');
ip.parse(varargin{:});
o = ip.Results;
if isfolder(o.Out) && numel(dir(o.Out)) > 2
    error('verify_resync:out', 'The output folder %s is not empty.', o.Out);
end
if ~isfolder(o.Out), mkdir(o.Out); end
T = resync_tables(o.Tables);
S = T.segments;
res = cell(height(S), 1);
t0 = tic;
for rec = unique(S.eeg_xdf, 'stable')'
    cache = containers.Map();                           % the raw files of one recording
    for i = find(S.eeg_xdf == rec)'
        res{i} = check(T, S(i, :), bids, cache);
    end
    fprintf('%s done (%.0f s)\n', rec, toc(t0));
end
R = struct2table([res{:}]);
writetable(R, fullfile(o.Out, 'verify_segments.csv'));
has_imu = R.imu_positions ~= "no IMU";
fprintf('EEG: %d/%d segments pass checks 1 and 2\n', sum(R.eeg_positions == "ok" & R.eeg_values == "ok"), height(R));
fprintf('IMU: %d/%d segments pass checks 1 and 2 (%d segments have no IMU accelerometer)\n', ...
    sum(R.imu_positions == "ok" & R.imu_values == "ok"), sum(has_imu), sum(~has_imu));
fprintf('check 3: %d ok, %d FAIL, %d shorter than 40 s, %d not usable, %d no IMU\n', sum(R.alignment == "ok"), ...
    sum(startsWith(R.alignment, "FAIL")), sum(startsWith(R.alignment, "shorter")), ...
    sum(startsWith(R.alignment, "not usable")), sum(R.alignment == "no IMU"));
end


function r = check(T, g, bids, cache)
C = resync_cut(T, g.sub, g.cond, g.seg);
load_files(C, bids, cache);
[eeg, imu] = resync_apply(C, cache, cache);
P = T.pieces(T.pieces.sub == g.sub & T.pieces.cond == g.cond & T.pieces.seg == g.seg ...
    & ~startsWith(T.pieces.note, "tail"), :);
r = struct('sub', g.sub, 'cond', g.cond, 'seg', g.seg, 'bids_name', g.bids_name, 'imu_usable', g.imu_usable);

% EEG: 1 positions, 2 values at the first and last sample of every piece
ok = size(eeg, 2) == g.eeg_samples ...
    && C.eeg_start.file + ":" + C.eeg_start.sample == g.eeg_start ...
    && C.eeg_end.file + ":" + C.eeg_end.sample == g.eeg_end;
r.eeg_positions = result(ok, strings(0, 1));
E = P(P.stream == "EEG", :);
bad = strings(0, 1);
for j = 1:height(E)
    got = double([eeg(1, E.seg_first(j)), eeg(1, E.seg_last(j))]);
    want = str2double([E.check_first(j), E.check_last(j)]);
    if any(abs(got - want) > 1e-3 + 1e-7 * abs(want)), bad(end + 1) = "piece " + E.piece(j); end %#ok<AGROW>
end
r.eeg_values = result(isempty(bad), bad);

if isempty(imu)                                         % no IMU accelerometer
    r.imu_positions = "no IMU"; r.imu_values = "no IMU"; r.alignment = "no IMU";
    r.rho = NaN; r.rho_table = g.rho; r.best_lag = NaN;
    return
end
% IMU: 1 positions, 2 values at the first and last row of every piece and the sum over the segment
ok = size(imu, 1) == g.imu_slots && sum(isnan(imu(:, 1))) == g.imu_empty ...
    && C.imu_start.file + ":" + C.imu_start.row == g.imu_start ...
    && C.imu_end.file + ":" + C.imu_end.row == g.imu_end;
r.imu_positions = result(ok, strings(0, 1));
I = P(P.stream == "IMU", :);
bad = strings(0, 1);
for j = 1:height(I)
    i0 = find(C.imu.file == I.file(j) & C.imu.row == I.first(j), 1);
    i1 = find(C.imu.file == I.file(j) & C.imu.row == I.last(j), 1);
    want = [str2double(split(I.check_first(j)))', str2double(split(I.check_last(j)))'];
    if isempty(i0) || isempty(i1) || any(abs([imu(i0, 1:3), imu(i1, 1:3)] - want) > 1e-9)
        bad(end + 1) = "piece " + I.piece(j); %#ok<AGROW>
    end
end
if abs(sum(imu(:, 1:3), 'all', 'omitnan') - g.check_imu_sum) > 1e-6 * max(1, abs(g.check_imu_sum))
    bad(end + 1) = "check_imu_sum";
end
r.imu_values = result(isempty(bad), bad);

% 3 alignment of the whole segment
rows = str2double(split(g.eeg_acc_rows, "-"));
A = resync_alignment(double(eeg(rows(1):rows(2), :))' / 1e6, imu(:, 1:3));
if g.imu_usable == "no"
    r.alignment = "not usable (imu_usable = no)";
elseif g.eeg_samples < 40 * 500
    r.alignment = "shorter than 40 s: reported only";
else
    r.alignment = result(abs(A.rho - g.rho) <= 0.02 && abs(A.best_lag) <= 1, strings(0, 1));
end
r.rho = A.rho; r.rho_table = g.rho; r.best_lag = A.best_lag;
end


function load_files(C, bids, cache)
for f = unique(C.eeg.file)'
    if ~isKey(cache, char(f))
        cache(char(f)) = load_xdf_eeg(raw_file(bids, C.info.sub, f), C.info.eeg_stream);
    end
end
for f = unique(C.imu.file(C.imu.file ~= ""))'
    if ~isKey(cache, char(f))
        cache(char(f)) = read_rabboni_txt(raw_file(bids, C.info.sub, f));
    end
end
end


function p = raw_file(bids, sub, name)
% raw recording in the dataset: sourcedata/sub-XX/eeg/<name>.xdf or sourcedata/sub-XX/motion/<name>.txt
kind = "motion";
if endsWith(name, ".xdf"), kind = "eeg"; end
p = fullfile(bids, 'sourcedata', "sub-" + extractAfter(string(sub), 3), kind, name);
end


function s = result(ok, bad)
if ok
    s = "ok";
elseif isempty(bad)
    s = "FAIL";
else
    s = "FAIL: " + strjoin(bad, ", ");
end
end
