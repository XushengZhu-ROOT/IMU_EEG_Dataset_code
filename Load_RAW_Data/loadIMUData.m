function [acc_data, gyro_data, imu_fs] = loadIMUData(filename)
% Load IMU data (acc + gyro) from txt file. Returns [n x 3] arrays.

imu_fs = 49.647537;

try
    data = readtable(filename, 'HeaderLines', 13, 'Delimiter', ',');

    if width(data) < 6
        error('IMU format error: need at least 6 columns (3 acc + 3 gyro)');
    end

    imu_data = table2array(data);
    acc_data = imu_data(:, 1:3);
    gyro_data = imu_data(:, 4:6);

    fprintf('Loaded IMU: %s\n', filename);
    fprintf('  Points: %d, duration: %.2fs (sr=%d Hz)\n', ...
        size(imu_data, 1), size(imu_data, 1)/imu_fs, imu_fs);

catch e
    error('Failed to load IMU data: %s', e.message);
end

end
