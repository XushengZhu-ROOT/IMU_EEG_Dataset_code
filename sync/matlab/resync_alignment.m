function A = resync_alignment(acc, imu, win_s, hop_s, low)
%RESYNC_ALIGNMENT  How well the IMU accelerometer follows the CGX accelerometer (EEG clock) in one segment.
%
%   A = resync_alignment(acc, imu)
%   A = resync_alignment(acc, imu, win_s, hop_s, low)       defaults 5 s, 1 s, 0.5
%
%   acc  [n x 3] CGX accelerometer of the cut EEG (g, 500 Hz): rows eeg_acc_rows of the stream / 1e6, transposed
%   imu  [m x 3] IMU accelerometer of the cut (g, 49.647537 Hz; NaN = empty slot)
%
%   Method (the same as resync.alignment in Python):
%     1. CGX: 4th-order Butterworth low-pass 20 Hz, zero phase; linear interpolation at the IMU slot times.
%     2. Both: empty IMU slots bridged by straight lines; 4th-order Butterworth band-pass 0.3-10 Hz, zero phase.
%        Before each filter the signal is extended by odd reflection (3 s), so the ends of a segment stay usable.
%     3. rho: rotation-invariant Procrustes correlation (the IMU turned into the CGX axes by the best proper
%        rotation); 1 = the same movement.
%   A.rho, A.best_lag, A.rho_best  whole segment: rho at lag 0; best lag within +/-25 slots (+/-0.5 s), d > 0 = the
%                                  IMU moved d slots later. 2 s left out at both ends (shorter segments: 1/8 of their
%                                  length, at least 26 slots) and 1 s around empty slots.
%   A.windows  windows of win_s s (hop hop_s s) over the whole segment: t_start_s, t_end_s, rho (lag 0), amp_g (CGX
%              rms, g). Windows with rho < low also get best_lag (+/-1.5 s), rho_best and a class:
%                quiet     CGX moves less than 0.02 g rms: too little movement to judge
%                offset    the IMU fits clearly better at another lag (rho there >= rho + 0.2), at least in part
%                no match  movement, but the IMU does not follow the CGX at any lag
%   A.X, A.Y, A.miss  the band-passed CGX and IMU on the IMU slots, and the empty IMU slots

if nargin < 3 || isempty(win_s), win_s = 5; end
if nargin < 4 || isempty(hop_s), hop_s = 1; end
if nargin < 5 || isempty(low), low = 0.5; end
FS_EEG = 500; FS_IMU = 49.647537;
LAG = round(0.5 * FS_IMU); WIDE = round(1.5 * FS_IMU); PAD = round(3 * FS_IMU); MOVING = 0.02;

[z, p, k] = butter(4, 20 / (FS_EEG / 2), 'low');            [sos_lp, g_lp] = zp2sos(z, p, k);
[z, p, k] = butter(4, [0.3 10] / (FS_IMU / 2), 'bandpass'); [sos_bp, g_bp] = zp2sos(z, p, k);
c = padfilt(double(acc), sos_lp, g_lp, PAD);
t_c = (0:size(c, 1) - 1)' / FS_EEG;
t_i = (0:size(imu, 1) - 1)' / FS_IMU;
t_i = t_i(t_i <= t_c(end));
X = padfilt(interp1(t_c, c, t_i), sos_bp, g_bp, PAD);
m = size(X, 1);
Y = double(imu(1:m, 1:3));
miss = isnan(Y(:, 1));
if any(miss)
    Y = fillmissing(Y, 'linear', 'EndValues', 'nearest');
end
Y = padfilt(Y, sos_bp, g_bp, PAD);

% whole segment
edge = min(floor(2 * FS_IMU), max(floor(m / 8), LAG + 1));
bad = conv(double(miss), ones(2 * floor(FS_IMU) + 1, 1), 'same') > 0;      % 1 s around empty slots
i = (1:m)';
idx = find(~bad & i > edge & i <= m - edge);
d_all = -LAG:LAG;
curve = nan(size(d_all));
for j = 1:numel(d_all)
    if min(idx - d_all(j)) >= 1 && max(idx - d_all(j)) <= m
        curve(j) = krho(X(idx, :), Y(idx - d_all(j), :));
    end
end
[~, jb] = max(curve);
A.rho = curve(d_all == 0);
A.best_lag = d_all(jb);
A.rho_best = curve(jb);

% windows
W = round(win_s * FS_IMU); H = round(hop_s * FS_IMU);
starts = 1 + (0:H:max(m - W, 0));
if starts(end) - 1 + W < m
    starts(end + 1) = m - W + 1;                                          % the last window ends with the segment
end
rows = cell(0, 7);
for s = starts
    w = (s:min(s + W - 1, m))';
    w = w(~miss(w));
    if numel(w) < floor(W / 2)
        continue
    end
    r0 = krho(X(w, :), Y(w, :));
    amp = sqrt(mean(sum(X(w, :) .^ 2, 2)));
    bl = NaN; rb = NaN; cls = "";
    if r0 < low
        best = -Inf;
        for d = -WIDE:WIDE
            if min(w - d) >= 1 && max(w - d) <= m
                r = krho(X(w, :), Y(w - d, :));
                if r > best, best = r; bl = d; end
            end
        end
        rb = best;
        if amp < MOVING
            cls = "quiet";
        elseif bl ~= 0 && rb >= r0 + 0.2
            cls = "offset";
        else
            cls = "no match";
        end
    end
    rows(end + 1, :) = {(s - 1) / FS_IMU, min(s - 1 + W, m) / FS_IMU, r0, amp, bl, rb, cls}; %#ok<AGROW>
end
A.windows = cell2table(rows, 'VariableNames', {'t_start_s', 't_end_s', 'rho', 'amp_g', 'best_lag', 'rho_best', 'cls'});
A.X = X; A.Y = Y; A.miss = miss;
end


function y = padfilt(x, sos, g, pad)
% zero-phase filter after odd reflection of `pad` samples at both ends
n = min(pad, size(x, 1) - 1);
xp = [2 * x(1, :) - x(n + 1:-1:2, :); x; 2 * x(end, :) - x(end - 1:-1:end - n, :)];
yp = filtfilt(sos, g, xp);
y = yp(n + 1:n + size(x, 1), :);
end


function r = krho(X, Y)
% rotation-invariant Procrustes correlation of two [n x 3] signals
X = X - mean(X, 1); Y = Y - mean(Y, 1);
[U, S, V] = svd(X' * Y);
d = sign(det(V * U'));
s = diag(S);
r = (s(1) + s(2) + d * s(3)) / sqrt(sum(X(:) .^ 2) * sum(Y(:) .^ 2));
end
