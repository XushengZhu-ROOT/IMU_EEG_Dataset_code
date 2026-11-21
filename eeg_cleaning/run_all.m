% Writes the cleaned Move EEG, ERP measures, and ERP and gyroscope figures.
% Set BIDS, OUT and EEGLAB below, then run:
%   run('eeg_cleaning/run_all.m')
% Needs the Signal Processing and Statistics toolboxes.
BIDS = '/path/to/bids_dataset';             % BIDS dataset root
OUT = '/path/to/output_folder';             % new or empty folder
EEGLAB = '/path/to/eeglab2025.1.0';         % EEGLAB folder
if any(startsWith({BIDS, OUT, EEGLAB}, '/path/to/')), error('Set BIDS, OUT and EEGLAB at the top of run_all.m'); end
if exist(OUT, 'dir') && numel(dir(OUT)) > 2, error('%s is not empty', OUT); end
HERE = fileparts(mfilename('fullpath'));
addpath(HERE, fullfile(fileparts(HERE), 'figures'));

% ERP figure: four cleaning versions
clean_paper_versions(BIDS, OUT, EEGLAB, 'hp_notch', 0);
clean_paper_versions(BIDS, OUT, EEGLAB, 'imureg', 0);
clean_paper_versions(BIDS, OUT, EEGLAB, 'asr', 30);
clean_paper_versions(BIDS, OUT, EEGLAB, 'imureg_asr', 30);
V = struct('label', {'before cleaning', 'IMU regression only', 'ASR only', 'IMU regression + ASR'}, ...
           'folder', fullfile(OUT, {'hp_notch/k0', 'imureg/k0', 'asr/k30', 'imureg_asr/k30'}));
erp_measures(V, BIDS, fullfile(OUT, 'erp_by_version'));
plot_erp_figures(fullfile(OUT, 'erp_by_version'), 'All movement (all subjects)', fullfile(OUT, 'fig1_erp_by_version'));

% ASR cutoff figure: ASR only with k = 10, 20 and 30
for k = [10 20]
    clean_paper_versions(BIDS, OUT, EEGLAB, 'asr', k);
end
V = struct('label', {'before cleaning', 'ASR k10', 'ASR k20', 'ASR k30'}, ...
           'folder', fullfile(OUT, {'hp_notch/k0', 'asr/k10', 'asr/k20', 'asr/k30'}));
erp_measures(V, BIDS, fullfile(OUT, 'erp_asr_k'));
plot_erp_figures(fullfile(OUT, 'erp_asr_k'), 'All movement, ASR cutoff k (all subjects)', ...
                 fullfile(OUT, 'fig_asr_k_comparison'));

% gyroscope figure
make_gyro_figure(BIDS, OUT);

fprintf('Next, the BIDS derivative of the cleaned EEG (version imureg_asr, k = 30):\n');
fprintf('  python eeg_cleaning/make_clean_erp.py --bids %s --mat %s --out %s\n', BIDS, ...
        fullfile(OUT, 'imureg_asr', 'k30'), fullfile(OUT, 'derivatives', 'clean'));
