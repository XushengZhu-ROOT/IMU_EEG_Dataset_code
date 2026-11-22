function EEG = apply_asr(EEG, k)
% Artifact subspace reconstruction (clean_rawdata) with cutoff k, keeping every channel and sample, then 40 Hz low-pass.

    EEG = clean_rawdata(EEG, ...
        'off', ...          % FlatlineCriterion
        [0.25 0.75], ...    % Highpass: drift high-pass transition band (Hz)
        'off', ...          % ChannelCriterion
        'off', ...          % LineNoiseCriterion
        k, ...              % BurstCriterion: ASR cutoff k (standard deviations)
        'off');             % WindowCriterion

    EEG = pop_eegfiltnew(EEG, [], 40);
    fprintf('k = %d\n', k);
end
