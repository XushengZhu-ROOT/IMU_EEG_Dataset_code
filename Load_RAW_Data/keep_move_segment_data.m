function [EEG_kept, acc_data_kept, gyro_data_kept] = keep_move_segment_data(EEG, acc_data, gyro_data, sub_idx, task, skip_channels)
% KEEP_SEGMENT_DATA Keep EEG, accelerometer and gyroscope data for a specific segment between event markers
%   [EEG_kept, acc_data_kept, gyro_data_kept] = KEEP_SEGMENT_DATA(EEG, acc_data, gyro_data, sub_idx, task, skip_channels)
%   Displays all event markers, lets the user select start and end event markers,
%   keeps only the data between these events, and returns processed EEG, accelerometer and gyroscope data
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
%       EEG_kept - EEG structure with only the kept segment
%       acc_data_kept - Accelerometer data structure with only the kept segment
%       gyro_data_kept - Gyroscope data structure with only the kept segment

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
    EEG_kept = EEG;
    acc_data_kept = acc_data;
    gyro_data_kept = gyro_data;
    
    % Event statistics before processing
    fprintf('\n===== Event Statistics Before Processing =====\n');
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
    
    key_pairs = {
        {'1 pressed', '1 released'};
        {'2 pressed', '2 released'};
        {'3 pressed', '3 released'}
    };
    
    start_idx = [];
    end_idx = [];
    start_marker = "";
    end_marker = "";
    detected_pair = '';
    
    for i = 1:length(key_pairs)
        start_marker = key_pairs{i}{1};
        end_marker = key_pairs{i}{2};
        
        temp_start_idx = find(strcmp({EEG.event.type}, start_marker));
        temp_end_idx = find(strcmp({EEG.event.type}, end_marker));
        
        if ~isempty(temp_start_idx) && ~isempty(temp_end_idx)
            if temp_start_idx(1) < temp_end_idx(1)
                start_idx = temp_start_idx(1);
                end_idx = temp_end_idx(1);
                detected_pair = sprintf('%s -> %s', start_marker, end_marker);
                fprintf('✓ found event pair: %s\n', detected_pair);
                break;
            end
        end
    end
    
    if isempty(start_idx) || isempty(end_idx)
        error('Cannot find valid pairs ("1 pressed"/"1 released" or "2 pressed"/"2 released")');
    end
    

    fprintf('\n===== "%s" =====\n', start_marker);
    fprintf('  Index   Type                 Time                       Latency\n');
    fprintf('  ---------------------------------------------------------------\n');
    evt_type = EEG.event(start_idx).type;
    evt_time = (EEG.event(start_idx).latency - 1) / EEG.srate;
    evt_mins = floor(evt_time / 60);
    evt_secs = mod(evt_time, 60);
    fprintf('  %3d    %-20s  %d:%02d (%.2fs)   %d\n', start_idx, evt_type, evt_mins, evt_secs, evt_time, round(EEG.event(start_idx).latency));


    fprintf('\n===== "%s" =====\n', end_marker);
    fprintf('  Index   Type                 Time                       Latency\n');
    fprintf('  ---------------------------------------------------------------\n');
    
    evt_type = EEG.event(end_idx).type;
    evt_time = (EEG.event(end_idx).latency - 1) / EEG.srate;
    evt_mins = floor(evt_time / 60);
    evt_secs = mod(evt_time, 60);
    fprintf('  %3d    %-20s  %d:%02d (%.2fs)   %d\n', end_idx, evt_type, evt_mins, evt_secs, evt_time, round(EEG.event(end_idx).latency));
    
    s_time = (EEG.event(start_idx).latency - 1) / EEG.srate;
    e_time = (EEG.event(end_idx).latency - 1) / EEG.srate;
    duration = e_time - s_time;
    
    segment_to_keep.segment_id = 1;
    segment_to_keep.start_idx = start_idx;
    segment_to_keep.end_idx = end_idx;
    segment_to_keep.start_time = s_time;
    segment_to_keep.end_time = e_time;
    segment_to_keep.duration = duration;
    segment_to_keep.start_latency = EEG.event(start_idx).latency;
    segment_to_keep.end_latency = EEG.event(end_idx).latency;
    
    % Display segment that will be kept
    fprintf('\nThe following segment will be KEPT (all other data will be removed):\n');
    fprintf('  Seg#   Start Idx  End Idx     Start Time               End Time               Duration\n');
    fprintf('  --------------------------------------------------------------------------------------\n');
    
    s_time = segment_to_keep.start_time;
    e_time = segment_to_keep.end_time;
    s_mins = floor(s_time / 60);
    s_secs = mod(s_time, 60);
    e_mins = floor(e_time / 60);
    e_secs = mod(e_time, 60);
    
    fprintf('  %3d    %5d      %5d      %d:%02d (%.2fs)  %d:%02d (%.2fs)  %.2fs\n', ...
        segment_to_keep.segment_id, segment_to_keep.start_idx, segment_to_keep.end_idx, ...
        s_mins, s_secs, s_time, e_mins, e_secs, e_time, segment_to_keep.duration);

    % Determine IMU sampling rate
    if isfield(acc_data, 'srate')
        imu_fs = acc_data.srate;
    else
        imu_fs = 49.647537;
    end
    
    % Visualize full dataset before processing
    before_fig = plotEEGData(sub_idx, [task ' - Before Processing (Full Data)'], EEG, skip_channels, ...
                             acc_data, gyro_data, imu_fs, [], [], [], []);
    
    % Print original data length
    orig_duration = EEG.pnts / EEG.srate;
    fprintf('\nOriginal data length: %.2f seconds (%d samples)\n', orig_duration, EEG.pnts);
    
    % Calculate the range to keep
    start_sample = round(segment_to_keep.start_latency);
    end_sample = round(segment_to_keep.end_latency);
    
    % Ensure samples are within valid range
    start_sample = max(1, start_sample);
    end_sample = min(EEG.pnts, end_sample);
    
    fprintf('\nKeeping EEG samples %d sec (%d) to %d sec (%d)...\n', s_time, start_sample, e_time, end_sample);
    
    % First handle accelerometer data in EEG structure BEFORE using pop_select
    % This is crucial because pop_select might not handle accelerometer data correctly
    if isfield(EEG, 'accelerometer') && isfield(EEG.accelerometer, 'data')
        fprintf('Pre-processing accelerometer data in EEG structure...\n');
        
        % Get the original accelerometer data size
        orig_acc_eeg_samples = size(EEG.accelerometer.data, 2);
        fprintf('Original CGX accelerometer data length: %d samples\n', orig_acc_eeg_samples);
        
        % Calculate which accelerometer samples to keep based on EEG samples
        % Assuming accelerometer in EEG structure has same sampling rate as EEG
        if orig_acc_eeg_samples == EEG.pnts
            % Same sampling rate as EEG
            acc_start_eeg = start_sample;
            acc_end_eeg = end_sample;
        else
            % Different sampling rate - calculate proportionally
            acc_ratio = orig_acc_eeg_samples / EEG.pnts;
            acc_start_eeg = round((start_sample - 1) * acc_ratio) + 1;
            acc_end_eeg = round(end_sample * acc_ratio);
        end
        
        % Ensure samples are within valid range
        acc_start_eeg = max(1, acc_start_eeg);
        acc_end_eeg = min(orig_acc_eeg_samples, acc_end_eeg);
        
        fprintf('Keeping CGX accelerometer samples %d to %d (out of %d)...\n', acc_start_eeg, acc_end_eeg, orig_acc_eeg_samples);
        
        % Keep only the selected segment of accelerometer data
        EEG.accelerometer.data = EEG.accelerometer.data(:, acc_start_eeg:acc_end_eeg);
        EEG.accelerometer.x = EEG.accelerometer.data(1, :);
        EEG.accelerometer.y = EEG.accelerometer.data(2, :);
        EEG.accelerometer.z = EEG.accelerometer.data(3, :);
        
        % Update accelerometer time axis
        new_acc_samples = size(EEG.accelerometer.data, 2);
        EEG.accelerometer.times = (0:new_acc_samples-1) / EEG.srate;
        if isfield(EEG, 'xmin')
            EEG.accelerometer.times = EEG.accelerometer.times + EEG.xmin;
        end
        
        fprintf('MOVE CGX acc length: %d samples\n', new_acc_samples);
    end
    
    % Now use pop_select to keep only the selected segment
    EEG_kept = pop_select(EEG, 'point', [start_sample end_sample]);
    
    % Synchronously keep external IMU data
    if exist('acc_data', 'var') && ~isempty(acc_data)
        start_time = (start_sample - 1) / EEG.srate;
        end_time = (end_sample - 1) / EEG.srate;
        
        acc_start_sample = round(start_time * imu_fs) + 1;
        acc_end_sample = round(end_time * imu_fs) + 1;
        
        % Ensure sampling points are within valid range
        acc_start_sample = max(1, min(acc_start_sample, size(acc_data, 1)));
        acc_end_sample = max(1, min(acc_end_sample, size(acc_data, 1)));
        
        if acc_start_sample <= acc_end_sample
            fprintf('Keeping acc data %d to %d...\n', acc_start_sample, acc_end_sample);
            acc_data_kept = acc_data(acc_start_sample:acc_end_sample, :);
        else
            fprintf('Warning: Invalid external accelerometer sample range\n');
            acc_data_kept = [];
        end
    end
    
    if exist('gyro_data', 'var') && ~isempty(gyro_data)
        start_time = (start_sample - 1) / EEG.srate;
        end_time = (end_sample - 1) / EEG.srate;
        
        gyro_start_sample = round(start_time * imu_fs) + 1;
        gyro_end_sample = round(end_time * imu_fs) + 1;
        
        % Ensure sampling points are within valid range
        gyro_start_sample = max(1, min(gyro_start_sample, size(gyro_data, 1)));
        gyro_end_sample = max(1, min(gyro_end_sample, size(gyro_data, 1)));
        
        if gyro_start_sample <= gyro_end_sample
            fprintf('Keeping gyro data %d to %d...\n', gyro_start_sample, gyro_end_sample);
            gyro_data_kept = gyro_data(gyro_start_sample:gyro_end_sample, :);
        else
            fprintf('Warning: Invalid external gyroscope sample range\n');
            gyro_data_kept = [];
        end
    end
    
    % Verify accelerometer data in EEG structure after pop_select
    if isfield(EEG_kept, 'accelerometer') && isfield(EEG_kept.accelerometer, 'data')
        final_acc_samples = size(EEG_kept.accelerometer.data, 2);
        final_eeg_samples = size(EEG_kept.data, 2);
        fprintf('\n| EEG samples: %d\n| CGX accsamples: %d\n', final_eeg_samples, final_acc_samples);
        
    end
    
    % Display event statistics after processing
    fprintf('\n===== Event Statistics After Processing =====\n');
    if isfield(EEG_kept, 'event') && ~isempty(EEG_kept.event)
        event_types = unique({EEG_kept.event.type}, 'stable');
        for i = 1:length(event_types)
            count = sum(strcmp({EEG_kept.event.type}, event_types{i}));
            fprintf('  %s: %d\n', event_types{i}, count);
        end
        fprintf('Total events: %d\n', length(EEG_kept.event));
    else
        fprintf('No events in EEG data after processing\n');
    end
    
    % Visualize data after processing using plotEEGData
    after_fig = plotEEGData(sub_idx, [task ' - After Processing (MOVE Segment)'], EEG_kept, skip_channels, ...
                             acc_data_kept, gyro_data_kept, imu_fs, [], [], [], []);
    
    % Add length information
    new_duration = EEG_kept.pnts / EEG_kept.srate;
    kept_duration = segment_to_keep.duration;

    
    fprintf('Kept segment from %.2f to %.2f seconds (%.2f seconds duration)\n', ...
        s_time, e_time, segment_to_keep.duration);

    EEG = EEG_kept;
    acc_data = acc_data_kept;
    gyro_data = gyro_data_kept;

    saveEEG_dir  = fullfile('EEG_Data', sub_idx, 'Segments', [sub_idx,'_', task, '_', 'Move.mat']);
    [folder, ~, ~] = fileparts(saveEEG_dir);
    if ~exist(folder, 'dir')
            mkdir(folder);
    end
    save(saveEEG_dir, 'EEG');
    
    saveACC_dir  = fullfile('IMU_Data', sub_idx, 'Segments', [sub_idx,'_', task, '_', 'ACC_Move.mat']);
    [folder, ~, ~] = fileparts(saveACC_dir);
    if ~exist(folder, 'dir')
            mkdir(folder);
    end
    save(saveACC_dir, 'acc_data');
    
    
    saveGYRO_dir  = fullfile('IMU_Data', sub_idx, 'Segments', [sub_idx,'_', task, '_', 'GYRO_Move.mat']);
    save(saveGYRO_dir, 'gyro_data');

    fprintf('save Move.mat files to:\n%s\n%s\n%s', saveEEG_dir, saveACC_dir, saveGYRO_dir);

end