function [eeg, imu] = resync_apply(C, eeg_src, imu_src)
%RESYNC_APPLY  Cut a segment from continuous EEG and raw IMU data.
%
%   [eeg, imu] = resync_apply(C, eeg_src, imu_src)       C from resync_cut
%
%   eeg_src  EEG stream [rows x samples] from load_xdf_eeg (may be processed if every sample keeps its column)
%   imu_src  IMU rows [N x 6] from read_rabboni_txt
%   A segment over two files needs a containers.Map (file name -> matrix); [] skips the EEG or IMU.
%
%   eeg [rows x n], imu [m x 6] (NaN = missing slot)

eeg = [];
if ~isempty(eeg_src)
    parts = cell(1, height(C.eeg));
    for k = 1:height(C.eeg)
        X = pick(eeg_src, C.eeg.file(k));
        parts{k} = X(:, C.eeg.first(k):C.eeg.last(k));
    end
    eeg = [parts{:}];
end

imu = [];
if ~isempty(imu_src) && height(C.imu) > 0
    m = height(C.imu);
    imu = nan(m, 6);
    raw = C.imu.status == "raw";
    for f = unique(C.imu.file(raw))'
        X = pick(imu_src, f);
        k = raw & C.imu.file == f;
        imu(k, :) = X(C.imu.row(k), 1:6);
    end
    st = C.imu.status == "interpolated";              % lost IMU samples
    d = diff([0; st; 0]);
    a = find(d == 1);
    b = find(d == -1) - 1;
    for j = 1:numel(a)
        lo = a(j) - 1;
        hi = b(j) + 1;
        if lo < 1 || hi > m || ~raw(lo) || ~raw(hi)
            error('resync_apply:interp', 'Interpolated slots %d-%d have no raw slot on both sides.', a(j), b(j));
        end
        x = (a(j):b(j))';
        slope = (imu(hi, :) - imu(lo, :)) ./ (hi - lo);
        imu(a(j):b(j), :) = slope .* (x - lo) + imu(lo, :);
    end
end
end


function X = pick(src, name)
if isa(src, 'containers.Map')
    X = src(char(name));
else
    X = src;
end
end
