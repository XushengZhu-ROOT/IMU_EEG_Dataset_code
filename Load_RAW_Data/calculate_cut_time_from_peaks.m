function cut_time = calculate_cut_time_from_peaks(cgx_peak_time, imu_peak_time, cgx_sampling_rate, imu_sampling_rate)
% Calculate the required cut_time based on manually marked peak times
%
% Inputs:
%   cgx_peak_time - Time of the peak in CGX data (seconds)
%   imu_peak_time - Time of the corresponding peak in IMU data (seconds)
%   cgx_sampling_rate - Sampling rate of CGX data (default 500Hz)
%
% Output:
%   cut_time - Number of samples to remove from the beginning of CGX data

% Calculate the time difference (how much CGX needs to shift forward)
time_diff = cgx_peak_time - imu_peak_time;

if time_diff > 0
    % Convert to number of samples
    cut_time = round(time_diff * cgx_sampling_rate);
elseif time_diff < 0
    cut_time = round(time_diff * imu_sampling_rate);

fprintf('CGX peak time: %.3f seconds\n', cgx_peak_time);
fprintf('IMU peak time: %.3f seconds\n', imu_peak_time);
fprintf('Time difference: %.3f seconds\n', time_diff);
fprintf('Suggested cut_time: %d samples\n', cut_time);
end