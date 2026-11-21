function [eeg, imu] = resync_apply(C, eeg_src, imu_src)
%RESYNC_APPLY  Cut one segment out of continuous EEG and raw IMU data at the positions from resync_cut.
%
%   [eeg, imu] = resync_apply(C, eeg_src, imu_src)
%
%   eeg_src  continuous EEG of the segment's xdf, [rows x samples]: column s = sample s of the xdf's EEG stream
%            (load_xdf_eeg returns this). Any rows, and any processing that keeps every sample in its column
%            (filtering, re-referencing, ICA): not resampled, no samples removed. A segment over two xdf files
%            (Sub05 and Sub07 movement4_speed1 Move) needs a containers.Map: xdf name -> matrix. [] skips the EEG.
%   imu_src  raw IMU rows [rows x 6] from read_rabboni_txt, or a containers.Map: txt name -> matrix. [] skips it.
%
%   eeg  [rows x n]: the pieces one after the other.
%   imu  [m x 6] (accX accY accZ gyroX gyroY gyroZ), one row per IMU slot: raw slots = their txt row; interpolated
%        slots = straight line between the raw slots on either side, per column; missing slots = NaN.

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
