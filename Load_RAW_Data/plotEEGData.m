function fig = plotEEGData(sub_idx, task, EEG, skip_channels, acc_data, gyro_data, imu_fs, time_range, scaling_factor, line_width, save_dir)
% PLOTEEGDATA Enhanced EEG data visualization function with IMU data synchronization support
%
% Inputs:
%   sub_idx - Subject identifier (required)
%   task - Task name (required)
%   EEG - EEGLAB EEG structure (required)
%   skip_channels - Channels to skip (cell array of strings or numeric array)
%   acc_data - Accelerometer data, size [n x 3] (optional)
%   gyro_data - Gyroscope data, size [n x 3] (optional)
%   imu_fs - IMU data sampling rate, default 50Hz (optional)
%   time_range - Display time range [start_time, end_time] in seconds (optional)
%   scaling_factor - Signal amplitude scaling factor (default: 0.3)
%   line_width - EEG curve line width (default: 0.3)
%   save_dir - Directory to save figure (optional)
%
% Output:
%   fig - Figure handle
%
% Examples:
%   % Display only EEG data
%   fig = plotEEGData('Sub01', 'RestingState', EEG, {});
%   
%   % Display EEG and IMU data
%   [acc_data, gyro_data, imu_fs] = loadIMUData('el4.txt');
%   fig = plotEEGData('Sub01', 'Task1', EEG, {}, acc_data, gyro_data, imu_fs);
%   
%   % Display specific time range with custom display parameters
%   fig = plotEEGData('Sub01', 'Task1', EEG, {'Fp1', 'Fp2'}, acc_data, gyro_data, imu_fs, [10, 60], 0.5, 0.4);

% Check required input parameters
if nargin < 3 || isempty(EEG)
    error('Please provide sub_idx, task, and EEG structure as inputs');
end

% Set default values for optional parameters
if nargin < 4 || isempty(skip_channels)
    skip_channels = {};
end

if nargin < 5
    acc_data = [];
end

if nargin < 6
    gyro_data = [];
end

% Determine whether IMU data is available
has_imu = ~isempty(acc_data) && ~isempty(gyro_data);

% Check whether CGX accelerometer data is available
has_cgx_acc = isfield(EEG, 'accelerometer') && isfield(EEG.accelerometer, 'data') && ~isempty(EEG.accelerometer.data);

if nargin < 7 || isempty(imu_fs)
    imu_fs = 49.647537; % Default IMU sampling rate
end

if nargin < 8 || isempty(time_range)
    time_range = [EEG.xmin, EEG.xmax]; % Use EEG time range
end

if nargin < 9 || isempty(scaling_factor)
    scaling_factor = 0.3;
end

if nargin < 10 || isempty(line_width)
    line_width = 0.3;
end

if nargin < 11 || isempty(save_dir)
    save_dir = [];
end

% Process skip channel indices
skip_channels_idx = [];
if ~isempty(skip_channels)
    % Get electrode names
    if isfield(EEG, 'chanlocs') && ~isempty(EEG.chanlocs)
        try
            electrode_names = {EEG.chanlocs.labels};
        catch
            electrode_names = cellstr(num2str((1:size(EEG.data,1))'));
        end
    else
        electrode_names = cellstr(num2str((1:size(EEG.data,1))'));
    end
    
    % Convert skip_channels to cell array
    if ischar(skip_channels)
        skip_channels = {skip_channels};
    elseif isstring(skip_channels)
        skip_channels = cellstr(skip_channels);
    elseif isnumeric(skip_channels)
        skip_channels_idx = skip_channels;
        skip_channels = electrode_names(skip_channels_idx);
    end
    
    % Find indices of channels to skip
    if isempty(skip_channels_idx)
        skip_channels_idx = find(ismember(electrode_names, skip_channels));
    end
end

% ===== KEY MODIFICATION: Extract EEG data directly, without re-synchronization =====
eeg_fs = EEG.srate;

% Extract EEG data based on time range
eeg_start_idx = max(1, round((time_range(1) - EEG.xmin) * eeg_fs) + 1);
eeg_end_idx = min(EEG.pnts, round((time_range(2) - EEG.xmin) * eeg_fs) + 1);

% Ensure valid index range
if eeg_start_idx > eeg_end_idx
    error(sprintf('Invalid time range: start index (%d) is later than end index (%d)', eeg_start_idx, eeg_end_idx));
end

% Extract data for the specified time range
eeg_data = EEG.data(:, eeg_start_idx:eeg_end_idx);
t_eeg = EEG.xmin + (eeg_start_idx-1:eeg_end_idx-1) / eeg_fs;

% Process IMU data (if available)
if has_imu
    % Calculate IMU time axis (assumed aligned with EEG)
    imu_start_idx = max(1, round((time_range(1) - EEG.xmin) * imu_fs) + 1);
    imu_end_idx = min(size(acc_data, 1), round((time_range(2) - EEG.xmin) * imu_fs) + 1);

    t_imu_full = EEG.xmin + (imu_start_idx-1:imu_end_idx-1) / imu_fs;
    
    if imu_start_idx <= size(acc_data, 1) && imu_end_idx >= 1
        acc_resampled = acc_data(imu_start_idx:imu_end_idx, :);
        gyro_resampled = gyro_data(imu_start_idx:imu_end_idx, :);
        
        % Resample to EEG time axis
        % if size(acc_resampled, 1) ~= length(t_eeg)
        %     t_imu_temp = EEG.xmin + (imu_start_idx-1:imu_end_idx-1) / imu_fs;
            
            % acc_temp = zeros(length(t_eeg), 3);
            % gyro_temp = zeros(length(t_eeg), 3);
            % 
            % for ch = 1:3
            %     % Initialize to 0
            %     acc_temp(:, ch) = zeros(size(t_eeg));
            %     % Interpolate only for time points within IMU data range
            %     valid_mask = (t_eeg >= min(t_imu_temp)) & (t_eeg <= max(t_imu_temp));
            %     if sum(valid_mask) > 0
            %         acc_temp(valid_mask, ch) = interp1(t_imu_temp, acc_resampled(:, ch), t_eeg(valid_mask), 'linear');
            %     end
            % 
            %     % Initialize to 0
            %     gyro_temp(:, ch) = zeros(size(t_eeg));
            %     % Interpolate only for time points within IMU data range
            %     valid_mask = (t_eeg >= min(t_imu_temp)) & (t_eeg <= max(t_imu_temp));
            %     if sum(valid_mask) > 0
            %         gyro_temp(valid_mask, ch) = interp1(t_imu_temp, gyro_resampled(:, ch), t_eeg(valid_mask), 'linear');
            %     end
            %     % acc_temp(:, ch) = interp1(t_imu_temp, acc_resampled(:, ch), t_eeg, 'linear', 'extrap');
            %     % gyro_temp(:, ch) = interp1(t_imu_temp, gyro_resampled(:, ch), t_eeg, 'linear', 'extrap');
            % end
            
            % acc_resampled = acc_temp;
            % gyro_resampled = gyro_temp;
        % end
    else
        warning('IMU data is invalid within the specified time range');
        has_imu = false;
    end
end

% Process CGX accelerometer data (if available)
if has_cgx_acc
    cgx_acc_full = EEG.accelerometer.data;
    cgx_time_full = EEG.accelerometer.times;
    
    % Check CGX timestamp format
    if max(cgx_time_full) > 1000
        cgx_time_full = cgx_time_full / 1000;
    end
    
    % Check whether CGX data needs scaling
    max_cgx_val = max(abs(cgx_acc_full(:)));
    if max_cgx_val > 100
        cgx_scale_factor = 1000000;
        cgx_acc_full = cgx_acc_full / cgx_scale_factor;
        fprintf('CGX accelerometer data scaled by factor %d\n', cgx_scale_factor);
    end
    
    % Extract CGX data for the specified time range
    cgx_time_mask = (cgx_time_full >= time_range(1)) & (cgx_time_full <= time_range(2));
    
    if any(cgx_time_mask)
        cgx_acc_data = cgx_acc_full(:, cgx_time_mask);
        cgx_time_vec = cgx_time_full(cgx_time_mask);
        % fprintf('CGX data time range: %.2f to %.2f seconds (%d points)\n', ...
        %         min(cgx_time_vec), max(cgx_time_vec), length(cgx_time_vec));
    else
        has_cgx_acc = false;
        warning('No CGX accelerometer data found within the specified time range');
    end
else
    warning('No CGX accelerometer data available');
end

% Get number of channels and data points
total_data_channels = size(eeg_data, 1);

% Get electrode names
if ~exist('electrode_names', 'var')
    if isfield(EEG, 'chanlocs') && ~isempty(EEG.chanlocs)
        try
            electrode_names = {EEG.chanlocs.labels};
        catch
            electrode_names = cellstr(num2str((1:total_data_channels)'));
        end
    else
        electrode_names = cellstr(num2str((1:total_data_channels)'));
    end
end

% ===== The following is the original plotting code, unchanged =====

% Create figure window
if has_imu && has_cgx_acc
    fig_height = 1000;
elseif has_imu || has_cgx_acc
    fig_height = 900;
else
    fig_height = 700;
end

fig = figure('Position', [60 100 1400 fig_height], 'Color', 'white', 'Resize', 'on');

% Create title
title_str = sprintf('%s %s EEG Data Visualization', sub_idx, task);
set(fig, 'Name', title_str, 'NumberTitle', 'off');

% Automatically calculate channel spacing
spacing = 15;
colors = lines(total_data_channels);
eeg_scaled = eeg_data * scaling_factor;

% Create subplot layout
if has_imu && has_cgx_acc
    ax1 = subplot('Position', [0.08, 0.60, 0.88, 0.33]);  % EEG data
    ax2 = subplot('Position', [0.08, 0.40, 0.88, 0.15]);  % CGX accelerometer data
    ax3 = subplot('Position', [0.08, 0.20, 0.88, 0.15]);  % IMU accelerometer data
    ax4 = subplot('Position', [0.08, 0.02, 0.88, 0.15]);  % IMU gyroscope data
elseif has_imu
    ax1 = subplot('Position', [0.08, 0.45, 0.88, 0.48]);  % EEG data
    ax2 = subplot('Position', [0.08, 0.25, 0.88, 0.15]);  % IMU accelerometer data
    ax3 = subplot('Position', [0.08, 0.05, 0.88, 0.15]);  % IMU gyroscope data
elseif has_cgx_acc
    ax1 = subplot('Position', [0.08, 0.35, 0.88, 0.58]);  % EEG data
    ax2 = subplot('Position', [0.08, 0.05, 0.88, 0.25]);  % CGX accelerometer data
else
    ax1 = subplot('Position', [0.08, 0.1, 0.88, 0.82]);
end

% Plot EEG data
axes(ax1);
hold on;
active_channels = 0;
for ch = 1:total_data_channels
    if ismember(ch, skip_channels_idx)
        continue;
    end
    plot(t_eeg, eeg_scaled(ch,:) + active_channels * spacing, 'Color', colors(ch,:), 'LineWidth', line_width);
    active_channels = active_channels + 1;
end
hold off;

% Set axis labels and title
ylabel('EEG Channels', 'FontSize', 12, 'FontWeight', 'bold');
if ~has_imu && ~has_cgx_acc
    xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
end
title(sprintf('%s %s EEG Data', sub_idx, task), 'FontSize', 14, 'FontWeight', 'bold');

% Set display range
active_channel_count = total_data_channels - length(skip_channels_idx);
channel_range = scaling_factor * 0.8;
max_val = (active_channel_count-1) * spacing + channel_range/2;
min_val = -channel_range/2;
upper_padding_ratio = 0.15;
lower_padding_ratio = 0.05;
data_range = max_val - min_val;

if data_range > 0
    ylim([min_val - data_range*lower_padding_ratio, ...
          max_val + data_range*upper_padding_ratio]);
end

xlim([time_range(1), time_range(2)]);

% Calculate appropriate x-axis tick interval
duration = time_range(2) - time_range(1);

if duration <= 5
    tick_interval = 0.5;
elseif duration <= 20
    tick_interval = 1;
elseif duration <= 60
    tick_interval = 5;
elseif duration <= 180
    tick_interval = 10;
elseif duration <= 600
    tick_interval = 30;
else
    tick_interval = 60;
end

% Set x-axis ticks
x_start = ceil(time_range(1) / tick_interval) * tick_interval;
x_end = floor(time_range(2) / tick_interval) * tick_interval;
x_ticks = x_start:tick_interval:x_end;

ax = gca;
ax.XTick = x_ticks;
ax.XMinorTick = 'on';
ax.XMinorGrid = 'on';

% Format time labels
if duration > 60
    tick_labels = cell(size(x_ticks));
    for i = 1:length(x_ticks)
        minutes = floor(x_ticks(i)/60);
        seconds = mod(x_ticks(i), 60);
        tick_labels{i} = sprintf('%d:%02d', minutes, seconds);
    end
    ax.XTickLabel = tick_labels;
end

% Set y-axis ticks and labels
active_electrode_names = electrode_names(~ismember(1:total_data_channels, skip_channels_idx));
if ~isempty(active_electrode_names)
    yticks((0:active_channel_count-1) * spacing);
    ax.YTickLabel = active_electrode_names;
end
ax.YAxis.FontSize = 9;
ax.YAxis.FontWeight = 'bold';

% Add grid
grid on;
ax.GridColor = [0.8 0.8 0.8];
ax.GridAlpha = 0.5;

% Plot CGX accelerometer data
if has_cgx_acc
    if has_imu && has_cgx_acc
        axes(ax2);
    elseif has_cgx_acc
        axes(ax2);
    end
    
    plot(cgx_time_vec, cgx_acc_data(1,:), 'r', 'LineWidth', 0.8); hold on;
    plot(cgx_time_vec, cgx_acc_data(2,:), 'g', 'LineWidth', 0.8);
    plot(cgx_time_vec, cgx_acc_data(3,:), 'b', 'LineWidth', 0.8);
    hold off;
    
    ylabel('CGX Acceleration (g)', 'FontSize', 10, 'FontWeight', 'bold');
    title('CGX Accelerometer Data', 'FontSize', 12, 'FontWeight', 'bold');
    legend('X', 'Y', 'Z', 'Location', 'northeast', 'FontSize', 9);
    xlim([time_range(1), time_range(2)]);
    
    % Auto-scale Y axis
    y_min = min(cgx_acc_data(:));
    y_max = max(cgx_acc_data(:));
    y_range = y_max - y_min;
    
    if y_range < 0.1
        y_center = (y_min + y_max) / 2;
        ylim([y_center - 0.5, y_center + 0.5]);
    else
        y_padding = y_range * 0.1;
        ylim([y_min - y_padding, y_max + y_padding]);
    end
    
    ax_cgx = gca;
    ax_cgx.XTick = x_ticks;
    grid on;
    ax_cgx.GridColor = [0.8 0.8 0.8];
    ax_cgx.GridAlpha = 0.3;
    
    if ~has_imu
        xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
        if duration > 60
            set(ax_cgx, 'XTickLabel', tick_labels);
        end
    end
end

% Plot IMU data
if has_imu
    % Determine which subplot to use
    if has_cgx_acc
        imu_acc_ax = ax3;
        imu_gyro_ax = ax4;
    else
        imu_acc_ax = ax2;
        imu_gyro_ax = ax3;
    end
    
    % IMU accelerometer data
    axes(imu_acc_ax);
    plot(t_imu_full, acc_resampled, 'LineWidth', 0.8);
    ylabel('IMU Acceleration (g)', 'FontSize', 10, 'FontWeight', 'bold');
    title('IMU Accelerometer Data', 'FontSize', 12, 'FontWeight', 'bold');
    legend('X', 'Y', 'Z', 'Location', 'northeast', 'FontSize', 9);
    xlim([time_range(1), time_range(2)]);
    ax = gca;
    ax.XTick = x_ticks;
    grid on;
    ax.GridColor = [0.8 0.8 0.8];
    ax.GridAlpha = 0.3;
    
    % IMU gyroscope data
    axes(imu_gyro_ax);
    plot(t_imu_full, gyro_resampled, 'LineWidth', 0.8);
    xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Angular Velocity (°/s)', 'FontSize', 10, 'FontWeight', 'bold');
    title('IMU Gyroscope Data', 'FontSize', 12, 'FontWeight', 'bold');
    legend('X', 'Y', 'Z', 'Location', 'northeast', 'FontSize', 9);
    xlim([time_range(1), time_range(2)]);
    ax = gca;
    ax.XTick = x_ticks;
    grid on;
    ax.GridColor = [0.8 0.8 0.8];
    ax.GridAlpha = 0.3;
    
    if duration > 60
        set(ax, 'XTickLabel', tick_labels);
    end
end

% Add annotation info
active_channel_count = total_data_channels - length(skip_channels_idx);
annotation('textbox', [0.01, 0.97, 0.25, 0.02], ...
    'String', sprintf('Active Channels: %d/%d', active_channel_count, total_data_channels), ...
    'EdgeColor', 'none', 'FontSize', 9, 'FontWeight', 'bold');

if ~isempty(skip_channels)
    skip_str = strjoin(skip_channels, ', ');
    if length(skip_str) > 50
        skip_str = [skip_str(1:47) '...'];
    end
    annotation('textbox', [0.01, 0.95, 0.4, 0.02], ...
        'String', sprintf('Skipped: %s', skip_str), ...
        'EdgeColor', 'none', 'FontSize', 9, 'FontWeight', 'bold');
end

data_types = {};
if has_cgx_acc
    data_types{end+1} = 'CGX Accelerometer';
end
if has_imu
    data_types{end+1} = 'IMU';
end
if ~isempty(data_types)
    data_str = strjoin(data_types, ' + ');
    annotation('textbox', [0.65, 0.97, 0.3, 0.02], ...
        'String', sprintf('Data: EEG + %s', data_str), ...
        'EdgeColor', 'none', 'FontSize', 9, 'FontWeight', 'bold');
end

info_str = sprintf('Sampling Rate: %d Hz | Scale: ±%.1f μV | Duration: %.1fs', ...
    round(EEG.srate), 1/scaling_factor, duration);
annotation('textbox', [0.25, 0.01, 0.5, 0.02], 'String', info_str, ...
    'EdgeColor', 'none', 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Save figure
if ~isempty(save_dir)
    try
        [folder, ~, ~] = fileparts(save_dir);
        if ~exist(folder, 'dir') && ~isempty(folder)
            mkdir(folder);
        end
        exportgraphics(fig, save_dir, 'Resolution', 300);
        fprintf('Figure saved to: %s\n', save_dir);
    catch ME
        warning('Failed to save figure: %s', ME.message);
    end
end

end