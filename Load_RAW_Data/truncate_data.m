function [EEG_cleaned, acc_data_cleaned, gyro_data_cleaned] = truncate_data(EEG, acc_data, gyro_data, sub_idx, task, skip_channels)
% CUT_BETWEEN_EVENT_MARKERS Cut EEG, accelerometer and gyroscope data between specific event markers
%   [EEG_cleaned, acc_data_cleaned, gyro_data_cleaned] = CUT_BETWEEN_EVENT_MARKERS(EEG, acc_data, gyro_data, sub_idx, task, skip_channels)
%   Displays all event markers, lets the user select start and end event markers,
%   cuts the data between these events, and returns processed EEG, accelerometer and gyroscope data
%
%   Inputs:
%       EEG - EEGLAB EEG structure
%       acc_data - Accelerometer data structure
%       gyro_data - Gyroscope data structure
%       sub_idx - Subject identifier (string)
%       task - Task identifier (string)
%       skip_channels - Channels to skip in visualization (cell array of strings)
%
%   Outputs:
%       EEG_cleaned - EEG structure after cutting
%       acc_data_cleaned - Accelerometer data structure after cutting
%       gyro_data_cleaned - Gyroscope data structure after cutting

% Set default values for optional parameters
if nargin < 4 || isempty(sub_idx)
    sub_idx = 'Subject';
end

if nargin < 5 || isempty(task)
    task = 'Task';
end

if nargin < 6
    skip_channels = {};
end

% Initial data copy
EEG_cleaned = EEG;
acc_data_cleaned = acc_data;
gyro_data_cleaned = gyro_data;

% Event statistics before cutting
fprintf('\n===== Event Statistics Before Cutting =====\n');
if isfield(EEG, 'event') && ~isempty(EEG.event)
    event_types = unique({EEG.event.type}, 'stable');
    for i = 1:length(event_types)
        count = sum(strcmp({EEG.event.type}, event_types{i}));
        fprintf('  %s: %d\n', event_types{i}, count);
    end
    fprintf('Total events: %d\n\n', length(EEG.event));
else
    fprintf('No events in EEG data\n');
    return;
end

% Ask user which event markers to cut between
fprintf('Enter the starting event marker type: ');
start_marker = input('', 's');

fprintf('Enter the ending event marker type: ');
end_marker = input('', 's');

% Validate the entered event markers
if ~any(strcmp(event_types, start_marker))
    fprintf('Error: Starting event marker "%s" does not exist\n', start_marker);
    return;
end

if ~any(strcmp(event_types, end_marker))
    fprintf('Error: Ending event marker "%s" does not exist\n', end_marker);
    return;
end

% Find all start and end event marker indices
start_indices = find(strcmp({EEG.event.type}, start_marker));
end_indices = find(strcmp({EEG.event.type}, end_marker));

if isempty(start_indices)
    fprintf('No starting event markers "%s" found\n', start_marker);
    return;
end

if isempty(end_indices)
    fprintf('No ending event markers "%s" found\n', end_marker);
    return;
end

% List all start and end event markers with times and latencies
fprintf('\n===== Starting Event Markers "%s" =====\n', start_marker);
fprintf('  Index   Type                 Time             Latency\n');
fprintf('  ------------------------------------------------------\n');
for i = 1:length(start_indices)
    idx = start_indices(i);
    evt_type = EEG.event(idx).type;
    evt_time = (EEG.event(idx).latency - 1) / EEG.srate;
    evt_mins = floor(evt_time / 60);
    evt_secs = mod(evt_time, 60);
    fprintf('  %3d    %-20s  %d:%02d (%.2fs)   %d\n', idx, evt_type, evt_mins, evt_secs, evt_time, round(EEG.event(idx).latency));
end

fprintf('\n===== Ending Event Markers "%s" =====\n', end_marker);
fprintf('  Index   Type                 Time             Latency\n');
fprintf('  ------------------------------------------------------\n');
for i = 1:length(end_indices)
    idx = end_indices(i);
    evt_type = EEG.event(idx).type;
    evt_time = (EEG.event(idx).latency - 1) / EEG.srate;
    evt_mins = floor(evt_time / 60);
    evt_secs = mod(evt_time, 60);
    fprintf('  %3d    %-20s  %d:%02d (%.2fs)   %d\n', idx, evt_type, evt_mins, evt_secs, evt_time, round(EEG.event(idx).latency));
end

% Automatically match valid segments
valid_segments = [];
fprintf('\n===== Possible Segments to Cut =====\n');
fprintf('  Seg#   Start Idx  End Idx     Start Time       End Time       Duration\n');
fprintf('  ---------------------------------------------------------------------\n');

for i = 1:length(start_indices)
    s_idx = start_indices(i);
    s_time = (EEG.event(s_idx).latency - 1) / EEG.srate;
    
    % Find the first end event after this start event
    next_end_idx = find(end_indices > s_idx, 1, 'first');
    
    if ~isempty(next_end_idx)
        e_idx = end_indices(next_end_idx);
        e_time = (EEG.event(e_idx).latency - 1) / EEG.srate;
        
        % Calculate duration
        duration = e_time - s_time;
        
        % Convert to readable time format
        s_mins = floor(s_time / 60);
        s_secs = mod(s_time, 60);
        e_mins = floor(e_time / 60);
        e_secs = mod(e_time, 60);
        
        % Display information
        fprintf('  %3d    %5d      %5d      %d:%02d (%.2fs)  %d:%02d (%.2fs)  %.2fs\n', ...
            i, s_idx, e_idx, s_mins, s_secs, s_time, e_mins, e_secs, e_time, duration);
        
        % Save segment information
        valid_segments(end+1).segment_id = i;
        valid_segments(end).start_idx = s_idx;
        valid_segments(end).end_idx = e_idx;
        valid_segments(end).start_time = s_time;
        valid_segments(end).end_time = e_time;
        valid_segments(end).duration = duration;
        valid_segments(end).start_latency = EEG.event(s_idx).latency;
        valid_segments(end).end_latency = EEG.event(e_idx).latency;
    end
end

if isempty(valid_segments)
    fprintf('No valid segments found to cut\n');
    return;
end

% Ask user which segments to cut
fprintf('\nEnter segment numbers to cut (comma-separated, e.g.: 1,3,5): ');
selected_segments_str = input('', 's');
selected_segments = str2num(selected_segments_str); %#ok<ST2NM>

if isempty(selected_segments)
    fprintf('No segments selected, operation canceled\n');
    return;
end

% Validate the entered segment numbers
if any(selected_segments < 1 | selected_segments > length(valid_segments))
    fprintf('Error: Provided segment numbers out of range (1-%d)\n', length(valid_segments));
    return;
end

% Display segments that will be cut
fprintf('\nThe following segments will be cut:\n');
fprintf('  Seg#   Start Idx  End Idx     Start Time       End Time       Duration\n');
fprintf('  ---------------------------------------------------------------------\n');
for i = 1:length(selected_segments)
    segment = valid_segments(selected_segments(i));
    
    s_time = segment.start_time;
    e_time = segment.end_time;
    s_mins = floor(s_time / 60);
    s_secs = mod(s_time, 60);
    e_mins = floor(e_time / 60);
    e_secs = mod(e_time, 60);
    
    fprintf('  %3d    %5d      %5d      %d:%02d (%.2fs)  %d:%02d (%.2fs)  %.2fs\n', ...
        segment.segment_id, segment.start_idx, segment.end_idx, ...
        s_mins, s_secs, s_time, e_mins, e_secs, e_time, segment.duration);
end

% Confirm before cutting
fprintf('\nConfirm cutting these segments? (y/n): ');
confirm = input('', 's');

if ~strcmpi(confirm, 'y')
    fprintf('Operation canceled\n');
    return;
end

% Visualize data before cutting using plotEEGData
fprintf('\nVisualizing data before cutting...\n');
% Determine IMU sampling rate
if isfield(acc_data, 'srate')
    imu_fs = acc_data.srate;
else
    imu_fs = 49.647537;
end

% Visualize full dataset before cutting
before_fig = plotEEGData(sub_idx, [task ' - Before Cutting'], EEG, skip_channels, ...
                         acc_data, gyro_data, imu_fs, [], [], [], []);

% Print original data length
orig_duration = EEG.pnts / EEG.srate;
fprintf('\nOriginal data length: %.2f seconds (%d samples)\n', orig_duration, EEG.pnts);

cut_intervals = [];
for i = 1:length(selected_segments)
    segment = valid_segments(selected_segments(i));
    cut_intervals(end+1, :) = [segment.start_latency, segment.end_latency];
end

cut_intervals = sortrows(cut_intervals, 1);

merged_intervals = [];
if ~isempty(cut_intervals)
    merged_intervals(1, :) = cut_intervals(1, :);
    for i = 2:size(cut_intervals, 1)
        if cut_intervals(i, 1) <= merged_intervals(end, 2)
            merged_intervals(end, 2) = max(merged_intervals(end, 2), cut_intervals(i, 2));
        else
            merged_intervals(end+1, :) = cut_intervals(i, :);
        end
    end
end

fprintf('\nCutting %d intervals:\n', size(merged_intervals, 1));
for i = 1:size(merged_intervals, 1)
    start_time = (merged_intervals(i, 1) - 1) / EEG.srate;
    end_time = (merged_intervals(i, 2) - 1) / EEG.srate;
    fprintf('  Interval %d: %.2f - %.2f seconds (samples %d - %d)\n', ...
        i, start_time, end_time, round(merged_intervals(i, 1)), round(merged_intervals(i, 2)));
end

EEG_temp = EEG;
for i = size(merged_intervals, 1):-1:1
    start_sample = round(merged_intervals(i, 1));
    end_sample = round(merged_intervals(i, 2));
    end_sample = min(end_sample, EEG_temp.pnts);
    
    fprintf('Cutting EEG samples %d to %d...\n', start_sample, end_sample);
    if isfield(EEG_temp, 'accelerometer') && isfield(EEG_temp.accelerometer, 'data')
        fprintf('检测到加速度计数据，将同步裁切...\n');
        
        orig_acc_size = size(EEG_temp.accelerometer.data, 2);
        fprintf('原始加速度计数据长度: %d 样本点\n', orig_acc_size);
        
        EEG_temp.accelerometer.data = [EEG_temp.accelerometer.data(:, 1:start_sample-1), EEG_temp.accelerometer.data(:, end_sample+1:end)];
        EEG_temp.accelerometer.x = EEG_temp.accelerometer.data(1, :);
        EEG_temp.accelerometer.y = EEG_temp.accelerometer.data(2, :);
        EEG_temp.accelerometer.z = EEG_temp.accelerometer.data(3, :);

        new_acc_samples = size(EEG_temp.accelerometer.data, 2);
        if new_acc_samples > 0
            EEG_temp.accelerometer.times = (0:new_acc_samples-1) / EEG_temp.srate;
            
            if isfield(EEG_temp, 'xmin')
                EEG_temp.accelerometer.times = EEG_temp.accelerometer.times + EEG_temp.xmin;
            end
        else
            EEG_temp.accelerometer.times = [];
        end

    end
    EEG_temp = pop_select(EEG_temp, 'nopoint', [start_sample end_sample]);
    
    if exist('acc_data', 'var') && ~isempty(acc_data)
        start_time = (start_sample - 1) / EEG.srate;
        end_time = (end_sample - 1) / EEG.srate;
        
        acc_start_sample = round(start_time * imu_fs) + 1;
        acc_end_sample = round(end_time * imu_fs) + 1;
        
        acc_start_sample = max(1, min(acc_start_sample, size(acc_data, 1)));
        acc_end_sample = max(1, min(acc_end_sample, size(acc_data, 1)));
        
        if acc_start_sample <= acc_end_sample
            fprintf('Cutting accelerometer samples %d to %d...\n', acc_start_sample, acc_end_sample);
            acc_data_temp = [acc_data(1:acc_start_sample-1,:); acc_data(acc_end_sample+1:end,:)];
            acc_data = acc_data_temp;
        end
    end
    
    if exist('gyro_data', 'var') && ~isempty(gyro_data)
        start_time = (start_sample - 1) / EEG.srate;
        end_time = (end_sample - 1) / EEG.srate;
        
        gyro_start_sample = round(start_time * imu_fs) + 1;
        gyro_end_sample = round(end_time * imu_fs) + 1;
        
        gyro_start_sample = max(1, min(gyro_start_sample, size(gyro_data, 1)));
        gyro_end_sample = max(1, min(gyro_end_sample, size(gyro_data, 1)));
        
        if gyro_start_sample <= gyro_end_sample
            fprintf('Cutting gyroscope samples %d to %d...\n', gyro_start_sample, gyro_end_sample);
            gyro_data_temp = [gyro_data(1:gyro_start_sample-1,:); gyro_data(gyro_end_sample+1:end,:)];
            gyro_data = gyro_data_temp;
        end
    end
end

% Update processed structures
EEG_cleaned = EEG_temp;
acc_data_cleaned = acc_data;
gyro_data_cleaned = gyro_data;

% Display event statistics after cutting
fprintf('\n===== Event Statistics After Cutting =====\n');
if isfield(EEG_cleaned, 'event') && ~isempty(EEG_cleaned.event)
    event_types = unique({EEG_cleaned.event.type}, 'stable');
    for i = 1:length(event_types)
        count = sum(strcmp({EEG_cleaned.event.type}, event_types{i}));
        fprintf('  %s: %d\n', event_types{i}, count);
    end
    fprintf('Total events: %d\n', length(EEG_cleaned.event));
    fprintf('Events removed: %d\n', length(EEG.event) - length(EEG_cleaned.event));
else
    fprintf('No events in EEG data after cutting\n');
end

% Visualize data after cutting using plotEEGData
fprintf('\nVisualizing data after cutting...\n');
after_fig = plotEEGData(sub_idx, [task ' - After Cutting'], EEG_cleaned, skip_channels, ...
                         acc_data, gyro_data, imu_fs, [], [], [], []);

% Add length information
new_duration = EEG_cleaned.pnts / EEG_cleaned.srate;
total_cut_duration = sum(diff(merged_intervals, 1, 2)) / EEG.srate;

fprintf('\nData length before cutting: %.2f seconds (%d samples)\n', orig_duration, EEG.pnts);
fprintf('Data length after cutting: %.2f seconds (%d samples)\n', new_duration, EEG_cleaned.pnts);
fprintf('Total cut duration: %.2f seconds (%d samples)\n', total_cut_duration, sum(diff(merged_intervals, 1, 2)));
fprintf('Reduced by: %.2f seconds (%.2f%%)\n', orig_duration - new_duration, 100 * (orig_duration - new_duration) / orig_duration);

if abs((orig_duration - new_duration) - total_cut_duration) > 0.001
    fprintf('WARNING: Cut duration mismatch! Expected: %.2f, Actual: %.2f\n', ...
        total_cut_duration, orig_duration - new_duration);
end

fprintf('\nCutting operation completed\n');
end