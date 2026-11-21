function C = resync_cut(T, sub, cond, seg)
%RESYNC_CUT  Where one segment is in the raw files: EEG pieces and IMU slots.
%
%   C = resync_cut(T, sub, cond, seg)     e.g. resync_cut(T, 'Sub01', 'movement3_speed0', 'Move')
%
%   C.eeg   one row per EEG piece: file (xdf), first, last, n, seg_first, seg_last
%   C.imu   one row per IMU slot: file (txt), row (0 = none), status (raw, interpolated or missing)
%   C.info  the segment's row of segments.csv
%   C.eeg_start, C.eeg_end, C.imu_start, C.imu_end  first and last EEG sample and IMU row
%
%   All positions are 1-based.

sub = string(sub); cond = string(cond); seg = string(seg);
S = T.segments;
G = S(S.sub == sub & S.cond == cond & S.seg == seg, :);
if height(G) ~= 1
    error('resync_cut:notFound', 'Segment %s %s %s is not in segments.csv.', sub, cond, seg);
end
P = T.pieces;
P = sortrows(P(P.sub == sub & P.cond == cond & P.seg == seg & ~startsWith(P.note, "tail"), :), 'piece');
C.eeg = P(P.stream == "EEG", {'file', 'first', 'last', 'n', 'seg_first', 'seg_last'});

file = strings(0, 1); row = zeros(0, 1); status = strings(0, 1);
if G.imu_method == "slot map"
    opts = detectImportOptions(fullfile(T.dir, G.imu_slot_map), 'FileType', 'text', 'Delimiter', ',');
    opts = setvartype(opts, 'string');
    M = readtable(fullfile(T.dir, G.imu_slot_map), opts);
    M = M(1:G.imu_slots, :);
    file = M.txt_file;
    row = str2double(M.row);
    row(isnan(row)) = 0;
    status = M.status;
elseif G.imu_method == "rows"
    I = P(P.stream == "IMU", :);
    for k = 1:height(I)
        r = I.first(k) + (0:I.n(k) - 1)';            % slot s of the piece = txt row first + s - 1
        has = r <= I.last(k);                         % past the end of the IMU file: empty
        st = repmat("missing", I.n(k), 1);
        st(has) = "raw";
        file = [file; repmat(I.file(k), I.n(k), 1)]; %#ok<AGROW>
        row = [row; r .* has];                        %#ok<AGROW>
        status = [status; st];                        %#ok<AGROW>
    end
end
C.imu = table(file, row, status);
C.info = G;
% start and end of the segment
C.eeg_start = struct('file', C.eeg.file(1), 'sample', C.eeg.first(1));
C.eeg_end = struct('file', C.eeg.file(end), 'sample', C.eeg.last(end));
r = find(C.imu.status == "raw");
if isempty(r)                                          % no IMU accelerometer
    C.imu_start = struct('file', "", 'row', NaN); C.imu_end = C.imu_start;
else
    C.imu_start = struct('file', C.imu.file(r(1)), 'row', C.imu.row(r(1)));
    C.imu_end = struct('file', C.imu.file(r(end)), 'row', C.imu.row(r(end)));
end
end
