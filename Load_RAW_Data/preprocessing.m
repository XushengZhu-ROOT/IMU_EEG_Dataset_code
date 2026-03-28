%% ============================================================
%  preprocessing.m
%  EEG + IMU Preprocessing Pipeline
%
%  Steps:
%    1. Load raw EEG (.xdf) and IMU (.txt) data
%    2. Synchronize EEG and IMU timelines
%    3. Remove unnecessary event markers
%    4. Cut data between event markers
%    5. Segment into REST / MOVE
%    6. Truncate undesired segments
%    7. Save all outputs
% ============================================================

clear;
close all;

%% ---- Config ------------------------------------------------
sub_idx = 'Sub06';
task    = 'Sit';
device  = 20;
imu_fs  = 49.647537;   % IMU sampling rate (Hz)

EEG_loadpath = fullfile('EEG_Data', sub_idx, [sub_idx, '_', task, '.xdf']);
IMU_loadpath = fullfile('IMU_Data', sub_idx, [sub_idx, '_', task, '.txt']);

%% ---- 1. Load Raw Data --------------------------------------
[ALLEEG, EEG, CURRENTSET] = eeglab;
EEG_Raw = process_xdf(EEG_loadpath, device, 'low', 3);
[acc_data_raw, gyro_data_raw, imu_fs] = loadIMUData(IMU_loadpath);

[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG_Raw, CURRENTSET, ...
    'setname', 'EEG_Raw', 'overwrite', 'off', 'gui', 'off');
%%
time = max(EEG_Raw.xmax, size(acc_data_raw, 1)/imu_fs);
time_range = [0, time];
scaling_factor = 0.1; % adjust EEG signal amplitude
line_width = 0.5; % line_width
skip_channels = {};

plotEEGData(sub_idx, task, EEG_Raw, skip_channels, acc_data_raw, gyro_data_raw, imu_fs, time_range, scaling_factor, line_width);
%% ---- 2. EEG / IMU Time Synchronization ---------------------
% Manually identified sync peaks (update per subject/session)
cgx_time = 9.318;
imu_time = 2.9;

cut_time = calculate_cut_time_from_peaks(cgx_time, imu_time, 500, imu_fs);

EEG = EEG_Raw;

% Trim EEG from cut_time onward and remove boundary events
EEG = pop_select(EEG, 'point', [cut_time EEG.pnts]);
EEG.event(strcmp({EEG.event.type}, 'boundary')) = [];
EEG = eeg_checkset(EEG);

% Synchronize accelerometer data embedded in EEG struct (if present)
if isfield(EEG, 'accelerometer') && ~isempty(EEG.accelerometer) && cut_time > 0
    fprintf('Trimming embedded accelerometer data...\n');
    if isfield(EEG.accelerometer, 'data')
        EEG.accelerometer.data = EEG.accelerometer.data(:, cut_time:end);
    end
    for ax = {'x', 'y', 'z'}
        if isfield(EEG.accelerometer, ax{1})
            EEG.accelerometer.(ax{1}) = EEG.accelerometer.(ax{1})(cut_time:end);
        end
    end
    if isfield(EEG.accelerometer, 'times')
        EEG.accelerometer.times = linspace(EEG.xmin, EEG.xmax, EEG.pnts);
    end
    if size(EEG.accelerometer.data, 2) ~= EEG.pnts
        warning('Accelerometer length mismatch after trimming.');
    end

% Trim external IMU arrays if cut is negative (IMU starts later)
elseif cut_time < 0
    acc_data_raw  = acc_data_raw(-cut_time:end, :);
    gyro_data_raw = gyro_data_raw(-cut_time:end, :);
    fprintf('IMU data trimmed. New length: %d samples\n', size(acc_data_raw, 1));
end

acc_data  = acc_data_raw;
gyro_data = gyro_data_raw;

EEG_syn_imu = EEG;
fprintf('\nSync complete: EEG = EEG_syn_imu\n');

%% ---- 3. Remove Unnecessary Event Markers -------------------
EEG = remove_specific_events(EEG_syn_imu);
EEG_correctedmarked = EEG;

[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
    'setname', 'EEG_correctedmarked', 'overwrite', 'off', 'gui', 'off');

%% ---- 4. Cut Between Event Markers --------------------------
skip_channels = {};
[EEG_cut, acc_data_cut, gyro_data_cut] = cut_between_event_markers( ...
    EEG, acc_data, gyro_data, sub_idx, task, skip_channels);

EEG       = EEG_cut;
acc_data  = acc_data_cut;
gyro_data = gyro_data_cut;

[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
    'setname', 'EEG_cut', 'overwrite', 'off', 'gui', 'off');

%% ---- 5. Segment: REST and MOVE -----------------------------
% Segments (REST/MOVE) are saved automatically by keep_rest_segment_data() and keep_move_segment_data()
[EEG_rest, acc_data_rest, gyro_data_rest] = keep_rest_segment_data( ...
    EEG, acc_data, gyro_data, sub_idx, task);

[EEG_move, acc_data_move, gyro_data_move] = keep_move_segment_data( ...
    EEG, acc_data, gyro_data, sub_idx, task);
%% ---- 6. Truncate Undesired Segments ------------------------
[EEG_truncated, acc_data_truncated, gyro_data_truncated] = truncate_data( ...
    EEG, acc_data, gyro_data, sub_idx, task);

EEG       = EEG_truncated;
acc_data  = acc_data_truncated;
gyro_data = gyro_data_truncated;

[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
    'setname', 'EEG_truncated', 'overwrite', 'off', 'gui', 'off');

%% ---- 7. Save Outputs ---------------------------------------

speed     = 'slow';
task_name = [task, speed];

% Helper: ensure directory exists and save
save_mat = @(dir_path, filename, varname, data) deal( ...
    mkdir_safe(dir_path), save(fullfile(dir_path, filename), varname) );

% --- Synced (full) ---
save_to(fullfile('EEG_Data', sub_idx), [sub_idx,'_',task_name,'.mat'],      'EEG_syn_imu');
save_to(fullfile('IMU_Data', sub_idx), [sub_idx,'_',task_name,'_ACC.mat'],   'acc_data');
save_to(fullfile('IMU_Data', sub_idx), [sub_idx,'_',task_name,'_GYRO.mat'],  'gyro_data');

% --- Truncated ---
EEG       = EEG_truncated;
acc_data  = acc_data_truncated;
gyro_data = gyro_data_truncated;

save_to(fullfile('EEG_Data', sub_idx, 'Truncated'), [sub_idx,'_',task,'_Truncated.mat'],     'EEG');
save_to(fullfile('IMU_Data', sub_idx, 'Truncated'), [sub_idx,'_',task,'_ACC_Truncated.mat'],  'acc_data');
save_to(fullfile('IMU_Data', sub_idx, 'Truncated'), [sub_idx,'_',task,'_GYRO_Truncated.mat'], 'gyro_data');

fprintf('\nAll outputs saved.\n');

%% ---- Helper ------------------------------------------------
function save_to(folder, filename, varname)
    if ~exist(folder, 'dir'), mkdir(folder); end
    s.(varname) = evalin('caller', varname);
    save(fullfile(folder, filename), '-struct', 's');
end
