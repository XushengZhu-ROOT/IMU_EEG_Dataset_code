%% ============================================================
%  analysis.m
%  EEG Analysis Pipeline: Preprocessing → ASR → ICA → ERP
%
%  Prerequisite: run preprocessing.m first to generate Segments data.
%
%  Steps:
%    1. Load segmented MOVE data
%    2. Downsample
%    3. Filter + Re-reference (prep for ASR)
%    4. ASR artifact removal
%    5. ICA + ICLabel automatic component rejection
%    6. Audio ERP analysis
%    7. Save clean EEG
% ============================================================

clear;
close all;
%% ---- Config ------------------------------------------------
sub_idx = 'Sub16';
task    = 'Stairfast';
device  = 20;
imu_fs  = 49.647537;   % IMU sampling rate (Hz)

target_fs = 250;       % Downsampling target (Hz)
asr_k     = 10;        % ASR burst rejection threshold

% ICLabel rejection thresholds  [Brain Muscle Eye Heart LineNoise ChanNoise Other]
ic_thresholds = struct( ...
    'Muscle',   0.4, ...
    'Eye',      0.4, ...
    'Heart',    0.2, ...
    'Line',     0.5, ...
    'Channel',  0.5  ...
);
%% ---- 1. Load Segmented MOVE Data ---------------------------
EEG_path  = fullfile('EEG_Data', sub_idx, 'Segments', [sub_idx,'_',task,'_Move.mat']);
ACC_path  = fullfile('IMU_Data', sub_idx, 'Segments', [sub_idx,'_',task,'_ACC_Move.mat']);
GYRO_path = fullfile('IMU_Data', sub_idx, 'Segments', [sub_idx,'_',task,'_GYRO_Move.mat']);

[ALLEEG, EEG, CURRENTSET] = eeglab;
load(EEG_path);   % loads EEG_move
load(ACC_path);   % loads acc_data_move
load(GYRO_path);  % loads gyro_data_move

% EEG       = EEG_move;
% acc_data  = acc_data_move;
% gyro_data = gyro_data_move;

%% ---- 2. Downsample -----------------------------------------
EEG = pop_resample(EEG, target_fs);
EEG.setname = 'EEG_resampled';
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, length(ALLEEG) + 1);
eeglab redraw;

%% ---- 3. Filter + Re-reference (pre-ASR) --------------------
EEG = preprocess_eeg_forASR(EEG);
EEG.setname = 'EEG_beforeASR';
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, length(ALLEEG) + 1);
eeglab redraw;

%% ---- 4. ASR Artifact Removal -------------------------------
EEG_asr = apply_asr(EEG, asr_k);
EEG_asr.setname = 'EEG_ASR';
checkpoint('ASR');
[ALLEEG, EEG_asr, CURRENTSET] = eeg_store(ALLEEG, EEG_asr, length(ALLEEG) + 1);
eeglab redraw;

%% ---- 5. ICA + ICLabel Automatic Component Rejection --------
EEG_ICA = pop_runica(EEG_asr, 'icatype', 'runica', 'extended', 1);
EEG_ICA = pop_iclabel(EEG_ICA);

ic_classes   = EEG_ICA.etc.ic_classification.ICLabel.classifications;
reject_comp  = false(1, size(ic_classes, 1));

reject_comp = reject_comp | (ic_classes(:, 2) > ic_thresholds.Muscle)';   % Muscle
reject_comp = reject_comp | (ic_classes(:, 3) > ic_thresholds.Eye)';      % Eye
reject_comp = reject_comp | (ic_classes(:, 4) > ic_thresholds.Heart)';    % Heart
reject_comp = reject_comp | (ic_classes(:, 5) > ic_thresholds.Line)';     % Line Noise
reject_comp = reject_comp | (ic_classes(:, 6) > ic_thresholds.Channel)';  % Channel Noise

reject_idx = find(reject_comp);
fprintf('Components rejected: ');
disp(reject_idx);

EEG_clean = pop_subcomp(EEG_ICA, reject_idx, 0);
EEG_clean.setname = 'EEG_ICA_clean';
checkpoint('ICA');
[ALLEEG, EEG_clean, CURRENTSET] = eeg_store(ALLEEG, EEG_clean, length(ALLEEG) + 1);
eeglab redraw;
fprintf('ICA cleaning complete.\n');

%% ---- 6. Audio ERP Analysis ---------------------------------
[EEG_standard, EEG_target, EEG_all_audio, ALLEEG, CURRENTSET] = ...
    find_audio_ERP(EEG_clean, ALLEEG, length(ALLEEG) + 1);

% ERP comparison plot
save_dir = fullfile('Results', sub_idx, task, 'ERPs', ...
    [sub_idx,'_',task,'_ERPs.png']);
mkdir_safe(fileparts(save_dir));
plot_erp_comparison(EEG_target, EEG_standard, ...
    [sub_idx, ' ', task, ' ERP Comparison'], save_dir);
%% ---- 9. Save Clean EEG -------------------------------------
EEG = EEG_clean;
clean_dir = fullfile('EEG_Data', sub_idx, 'Clean');
mkdir_safe(clean_dir);
save(fullfile(clean_dir, [sub_idx,'_',task,'_clean.mat']), 'EEG');
fprintf('\nClean EEG saved to: %s\n', clean_dir);

%% ---- Helper ------------------------------------------------
function mkdir_safe(folder)
    if ~exist(folder, 'dir'), mkdir(folder); end
end
