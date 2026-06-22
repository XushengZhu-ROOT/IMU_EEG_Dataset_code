function EEG = preprocess_eeg_forASR(EEG)
%     EEG.etc.icLabel.version = 'default';

%     channel_std = std(EEG.data, [], 2);
%     channel_median_std = median(channel_std);
%     scaling_factors = channel_median_std ./ channel_std;
%     % 对异常通道进行特殊处理
%     for ch = 1:EEG.nbchan
%         if channel_std(ch) > channel_median_std
%             EEG.data(ch,:) = EEG.data(ch,:) * scaling_factors(ch);
%             if channel_std(ch) > 2*channel_median_std
%                 channel_mean = mean(EEG.data(ch,:));
%                 EEG.data(ch,:) = ((EEG.data(ch,:) - channel_mean) * scaling_factors(ch)) + channel_mean;
%             end
%         end
%     end

%     EEG.data = improved_wavelet_denoise(EEG.data, EEG.srate);

    % 平均参考
    EEG = pop_reref(EEG, []);
    
    % Cleanline去除电源线噪声
    cleanline_params = struct(...
        'bandwidth', 2, ...
        'SignalType', 'Channels', ...
        'ChanCompIndices', 1:EEG.nbchan, ...
        'ScanForLines', 1, ...
        'NormalizeSpectrum', 1, ...
        'LineAlpha', 0.01, ...
        'PaddingFactor', 2, ...
        'PlotFigures', 0);
    
    for freq = [60 120]
        cleanline_params.LineFrequencies = freq;
        EEG = pop_cleanline(EEG, cleanline_params);
    end
    checkpoint('preprocess_eeg_forASR');
end