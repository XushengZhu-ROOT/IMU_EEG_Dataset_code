% Re-run specified Subjects and merge back into existing quality_metrics.mat
clear; clc;

subjects = arrayfun(@(n) sprintf('%02d', n), [1:11, 13:21], 'UniformOutput', false);

% =========================================================
% Configuration parameters (same as original)
% =========================================================
base_pathEEG = '/Users/pinocchio/Documents/SCCN/IMU+EEG/EEG_Data';
base_pathIMU = '/Users/pinocchio/Documents/SCCN/IMU+EEG/IMU_Data';

tasks = {'Walkslow','Walkmedium','Walkfast', ...
         '8slow','8medium','8fast', ...
         'Horizontalslow','Horizontalmedium','Horizontalfast', ...
         'Verticalslow','Verticalmedium','Verticalfast', ...
         'Pickslow','Pickmedium','Pickfast', ...
         'Stairslow','Stairmedium','Stairfast'};
k = 10;

muscle_band = [30 50];

% =========================================================
% Initialize results
% =========================================================
results    = struct('sub', {}, 'task', {}, 'muscle_reduction_dB', {}, 'muscle_reduction_pct', {});
result_idx = 0;

% =========================================================
% EEGLAB initialization
% =========================================================
addpath(genpath('/Users/pinocchio/Documents/SCCN/IMU+EEG/eeglab2025.1.0'));
eeglab nogui;
ALLEEG = []; CURRENTSET = 0;

% Some FieldTrip-lite compat folders shadow MATLAB's built-in flip(),
% which breaks legend(). Remove any non-MATLAB flip.m from the path.
fix_flip_path_conflict();

% Set true to visually compare cleaned vs. original data (opens an
% interactive vis_artifacts window for every subject/task - use with care
% in a full batch run).
show_vis_artifacts = false;

% =========================================================
% loop
% =========================================================
total_files  = length(subjects) * length(tasks);
current_file = 0;

for s = 1:length(subjects)
    sub_idx = "Sub" + subjects{s};

    for t = 1:length(tasks)
        task         = tasks{t};
        current_file = current_file + 1;

        fprintf('\n=== %d/%d: %s_%s ===\n', ...
                current_file, total_files, sub_idx, task);

        try
            % --------------------------------------------------
            % 1. Load data
            % --------------------------------------------------
            EEG_Data  = find_file_case_insensitive(base_pathEEG, sub_idx, "Segments", sub_idx + "_" + task + "_Move.mat");
            ACC_Data  = find_file_case_insensitive(base_pathIMU, sub_idx, "Segments", sub_idx + "_" + task + "_ACC_Move.mat");
            GYRO_Data = find_file_case_insensitive(base_pathIMU, sub_idx, "Segments", sub_idx + "_" + task + "_GYRO_Move.mat");

            if isempty(EEG_Data) || isempty(ACC_Data) || isempty(GYRO_Data)
                fprintf('File not found, skipping: %s_%s\n', sub_idx, task);
                continue;
            end

            ALLEEG = []; CURRENTSET = 0;

            S = load(EEG_Data);  fields = fieldnames(S);  EEG       = S.(fields{1});
            S = load(ACC_Data);  fields = fieldnames(S);  acc_data  = S.(fields{1});
            S = load(GYRO_Data); fields = fieldnames(S);  gyro_data = S.(fields{1});

            fprintf('Loaded EEG: %s\n', EEG_Data);

            % --------------------------------------------------
            % 2. Downsample + preprocessing
            % --------------------------------------------------
            EEG = pop_resample(EEG, 250);
            EEG = preprocess_eeg_forASR(EEG);

            % --------------------------------------------------
            % [Metrics BEFORE ASR]
            % --------------------------------------------------
            EEG_beforeASR    = EEG;
            muscle_power_pre = compute_bandpower(EEG_beforeASR, muscle_band);

            % --------------------------------------------------
            % 3. ASR
            % --------------------------------------------------
            EEG_asr         = apply_asr(EEG, k);
            EEG_asr.setname = 'EEG_ASR';
            EEG = EEG_asr;

            % --------------------------------------------------
            % Compare cleaned data to the original
            % --------------------------------------------------
            if show_vis_artifacts
                vis_artifacts(EEG_asr, EEG_beforeASR);
            end

            % --------------------------------------------------
            % [Metrics AFTER ASR]
            % --------------------------------------------------
            muscle_power_post    = compute_bandpower(EEG_asr, muscle_band);
            muscle_reduction_db  = 10*log10(muscle_power_pre / muscle_power_post);
            muscle_reduction_pct = (1 - muscle_power_post / muscle_power_pre) * 100;

            % --------------------------------------------------
            % New record
            % --------------------------------------------------
            new_record.sub                  = char(sub_idx);
            new_record.task                 = task;
            new_record.muscle_reduction_dB  = muscle_reduction_db;
            new_record.muscle_reduction_pct = muscle_reduction_pct;

            result_idx = result_idx + 1;
            results(result_idx) = new_record;

            fprintf('Done: %s_%s\n', sub_idx, task);

        catch ME
            fprintf('Failed: %s_%s\n', sub_idx, task);
            fprintf('   Error: %s\n', ME.message);
            continue;
        end
    end
end

% =========================================================
% Plot results
% =========================================================
plot_muscle_reduction(results, result_idx);
fprintf('\nAll re-runs complete!\n');


% =========================================================
% ==================  Helper functions  ==================
% =========================================================

function plot_muscle_reduction(results, result_idx)
    % Guard against the flip.m path conflict again in case a plugin
    % re-added its path after eeglab nogui ran.
    fix_flip_path_conflict();

    % --------------------------------------------------------------
    % Task category mapping: task_prefix -> display name
    % "medium" tasks are dropped; only slow/fast are plotted.
    % --------------------------------------------------------------
    category_map = {
        'Walk',       'Straight Walking';
        '8',          'Curved Walking';
        'Horizontal', 'Head Shaking';
        'Vertical',   'Head Nodding';
        'Pick',       'Picking up an Object';
        'Stair',      'Stair Climbing and Descending'
    };
    n_cat = size(category_map, 1);

    slow_mean = nan(1, n_cat);
    fast_mean = nan(1, n_cat);

    for c = 1:n_cat
        prefix = category_map{c, 1};

        slow_vals = arrayfun(@(r) r.muscle_reduction_pct, ...
            results(strcmp({results.task}, [prefix 'slow'])));
        fast_vals = arrayfun(@(r) r.muscle_reduction_pct, ...
            results(strcmp({results.task}, [prefix 'fast'])));

        slow_mean(c) = nanmean(slow_vals);
        fast_mean(c) = nanmean(fast_vals);
    end

    task_labels = category_map(:, 2)';

    % --------------------------------------------------------------
    % Per-subject means (averaged across all tasks for that subject)
    % --------------------------------------------------------------
    all_subs = unique({results(1:result_idx).sub}, 'stable');
    all_subs = sort(all_subs);
    n_subs   = length(all_subs);
    sub_mean = nan(1, n_subs);

    for si = 1:n_subs
        sub_vals = arrayfun(@(r) r.muscle_reduction_pct, ...
            results(strcmp({results.sub}, all_subs{si})));
        sub_mean(si) = nanmean(sub_vals);
    end

    % --------------------------------------------------------------
    % Figure
    % --------------------------------------------------------------
    figure('Name', 'Mean Muscle Reduction (%)', 'Color', 'w', ...
           'Position', [100 100 1300 500]);
    sgtitle('Mean Muscle Reduction (%)', 'FontWeight', 'bold', 'FontSize', 14);

    % ---- Left panel: Per Task (Slow vs Fast) ----
    subplot(1,2,1);
    b = bar([slow_mean' fast_mean'], 'grouped');
    b(1).FaceColor = [0.68 0.87 0.68];  % light green
    b(2).FaceColor = [0.18 0.44 0.24];  % dark green
    set(gca, 'XTick', 1:n_cat, 'XTickLabel', task_labels, ...
             'XTickLabelRotation', 30);
    ylabel('Muscle Reduction (%)');
    ylim([0 100]);
    title('Per Task');
    try
        legend({'Slow', 'Fast'}, 'Location', 'northwest');
    catch ME
        warning('Legend could not be created (%s). Skipping legend.', ME.message);
    end
    grid on; box on;
    add_bar_labels(b);

        % ---- Right panel: Per Subject ----
    subplot(1,2,2);
    b2 = bar(sub_mean, 'FaceColor', [0.18 0.44 0.24]);

    display_subs = all_subs;
    for si = 1:n_subs
        num = str2double(all_subs{si}(4:end));
        if num >= 13
            display_subs{si} = sprintf('Sub%02d', num - 1);
        end
    end

    set(gca, 'XTick', 1:n_subs, 'XTickLabel', display_subs, ...
             'XTickLabelRotation', 45);
    ylabel('Muscle Reduction (%)');
    ylim([0 100]);
    title('Per Subject');
    grid on; box on;
    add_bar_labels(b2);
end

function fix_flip_path_conflict()
    % Some EEGLAB/FieldTrip plugin folders (e.g. Fieldtrip-lite compat
    % folders) define their own flip.m that shadows MATLAB's built-in
    % flip(), which breaks legend(). Remove any non-MATLAB flip.m from
    % the path so the built-in one is used.
    flip_locs = which('flip', '-all');
    for i = 1:length(flip_locs)
        if ~startsWith(flip_locs{i}, matlabroot)
            conflict_dir = fileparts(flip_locs{i});
            rmpath(conflict_dir);
            fprintf('Removed conflicting path (shadows flip()): %s\n', conflict_dir);
        end
    end
end

function add_bar_labels(b)
    % Adds value labels above each bar (works for single or grouped bars)
    for k = 1:length(b)
        xd = b(k).XEndPoints;
        yd = b(k).YData;
        for i = 1:length(xd)
            if ~isnan(yd(i))
                text(xd(i), yd(i) + 2, sprintf('%.1f', yd(i)), ...
                     'HorizontalAlignment', 'center', ...
                     'FontSize', 10);
            end
        end
    end
end

function band_power = compute_bandpower(EEG, band)
    [psd_db, freqs] = spectopo(EEG.data, 0, EEG.srate, 'plot', 'off', 'verbose', 'off');
    psd_power = 10.^(psd_db / 10);
    band_idx = freqs >= band(1) & freqs <= band(2);
    if ~any(band_idx), band_power = NaN; return; end
    band_power = mean(psd_power(:, band_idx), 'all');
end

function filepath = find_file_case_insensitive(base_path, sub_idx, subfolder, target_filename)
    dir_path = fullfile(base_path, sub_idx, subfolder);
    if ~exist(dir_path, 'dir'), filepath = ''; return; end
    files        = dir(fullfile(dir_path, '*.mat'));
    target_lower = char(target_filename);
    for i = 1:length(files)
        if strcmpi(files(i).name, target_lower)
            filepath = fullfile(dir_path, files(i).name);
            return;
        end
    end
    filepath = '';
end

%%
task_labels = {'Straight Walking', 'Curved Walking', 'Head Shaking', ...
               'Head Nodding', 'Picking up an Object', 'Stair Climbing and Descending'};

slow_mean = [38.8, 36.8, 40.3, 46.9, 54.9, 55.7];
fast_mean = [37.9, 44.2, 39.2, 36.0, 60.7, 61.1];

sub_labels = arrayfun(@(n) sprintf('Sub%02d', n), 1:20, 'UniformOutput', false);
sub_mean = [57.9, 47.0, 47.1, 52.6, 41.7, 39.8, 54.9, 41.7, 36.3, 49.2, ...
            42.9, 60.8, 46.9, 42.7, 46.0, 35.5, 43.2, 37.7, 46.7, 48.3];

% ---- Mean across task conditions (12 個 task-level 值的平均) ----
overall_task_mean = mean([slow_mean, fast_mean], 'omitnan');
fprintf('Overall mean across task conditions: %.1f%%\n', overall_task_mean);

% ---- Mean across subjects (20 個 subject-level 值的平均) ----
overall_sub_mean = mean(sub_mean, 'omitnan');
fprintf('Overall mean across subjects: %.1f%%\n', overall_sub_mean);

% ---- 找 task 的 min/max，並標出是哪個 task + 哪個速度 ----
all_task_vals   = [slow_mean, fast_mean];
all_task_names  = [strcat(task_labels, ' Slow'), strcat(task_labels, ' Fast')];

[min_val, idx] = min(all_task_vals);
fprintf('Task min: %.1f%% (%s)\n', min_val, all_task_names{idx});

[max_val, idx] = max(all_task_vals);
fprintf('Task max: %.1f%% (%s)\n', max_val, all_task_names{idx});

% ---- 找 subject 的 min/max ----
[min_val, min_idx] = min(sub_mean);
fprintf('Subject min: %.1f%% (%s)\n', min_val, sub_labels{min_idx});

[max_val, max_idx] = max(sub_mean);
fprintf('Subject max: %.1f%% (%s)\n', max_val, sub_labels{max_idx});
%%
%% 直接用已有結果畫圖（不重跑 ASR pipeline）
clear; clc; close all;

% =========================================================
% 你的資料
% =========================================================
task_labels = {'Straight Walking', 'Curved Walking', 'Head Shaking', ...
               'Head Nodding', 'Picking up an Object', 'Stair Climbing and Descending'};

slow_mean = [38.8, 36.8, 40.3, 46.9, 54.9, 55.7];
fast_mean = [37.9, 44.2, 39.2, 36.0, 60.7, 61.1];

sub_labels = arrayfun(@(n) sprintf('Sub%02d', n), 1:20, 'UniformOutput', false);
sub_mean = [57.9, 47.0, 47.1, 52.6, 41.7, 39.8, 54.9, 41.7, 36.3, 49.2, ...
            42.9, 60.8, 46.9, 42.7, 46.0, 35.5, 43.2, 37.7, 46.7, 48.3];

n_cat  = length(task_labels);
n_subs = length(sub_labels);

% =========================================================
% Figure
% =========================================================
fix_flip_path_conflict();  % 避免 EEGLAB/FieldTrip 的 flip.m 蓋掉 MATLAB 內建的 flip()

figure('Name', 'Mean Muscle Reduction (%)', 'Color', 'w', ...
       'Position', [100 100 1300 500]);
sgtitle('Mean Muscle Reduction (%)', 'FontWeight', 'bold', 'FontSize', 24);

% ---- Left panel: Per Task (Slow vs Fast) ----
subplot(1,2,1);
b = bar([slow_mean' fast_mean'], 'grouped');
b(1).FaceColor = [0.68 0.87 0.68];  % light green
b(2).FaceColor = [0.18 0.44 0.24];  % dark green
set(gca, 'XTick', 1:n_cat, 'XTickLabel', task_labels, ...
         'XTickLabelRotation', 30, 'FontSize', 14);
ylabel('Muscle Reduction (%)',  'FontSize', 14);
ylim([0 100]);
title('Per Task',  'FontSize', 20, 'FontWeight', 'bold');
try
    legend({'Slow', 'Fast'}, 'Location', 'northwest',  'FontSize', 12);
catch ME
    warning('Legend could not be created (%s). Skipping legend.', ME.message);
end
grid on; box on;
add_bar_labels(b);

% ---- Right panel: Per Subject ----
subplot(1,2,2);
b2 = bar(sub_mean, 'FaceColor', [0.18 0.44 0.24]);

n_subs = length(sub_labels);

set(gca, 'XTick', 1:n_subs, 'XTickLabel', sub_labels, ...
         'XTickLabelRotation', 45,  'FontSize', 14);
ylabel('Muscle Reduction (%)',  'FontSize', 14);
ylim([0 100]);
title('Per Subject',  'FontSize', 20, 'FontWeight', 'bold');
grid on; box on;
add_bar_labels(b2);