function [EEG, acc_data, gyro_data] = cut_between_event_markers(EEG, acc_data, gyro_data, sub_idx, task, skip_channels)

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

    %=== Event statistics before cutting ===
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

    pressed = {'S pressed', '1 pressed', '2 pressed', '3 pressed'};
    released = {'S released', '1 released', '2 released', '3 released'};

    if isfield(EEG, 'event') && ~isempty(EEG.event)
        for hehe = 1:length(pressed)

            %=============================
            % Case 1: S released → S pressed
            %=============================
            if strcmp(pressed{hehe}, 'S pressed')
                idx_S_released = find(strcmp({EEG.event.type}, 'S released'));
                idx_S_pressed  = find(strcmp({EEG.event.type}, 'S pressed'));

                if isempty(idx_S_released) || isempty(idx_S_pressed)
                    fprintf('No S released or S pressed found.\n');
                    continue;
                end

                forbidden = {'1 pressed','1 released','2 pressed','2 released','3 pressed','3 released'};

                % === find valid S released → S pressed pair ===
                valid_pairs = [];
                for s = 1:length(idx_S_released)
                    s_idx = idx_S_released(s);
                    next_Sp = find(idx_S_pressed > s_idx, 1, 'first');
                    if isempty(next_Sp), continue; end
                    e_idx = idx_S_pressed(next_Sp);

                    middle_types = {EEG.event(s_idx+1:e_idx-1).type};

                    if all(~ismember(middle_types, forbidden))
                        valid_pairs(end+1,:) = [s_idx, e_idx];
                        fprintf('✅ Found clean pair: S released (%d) → S pressed (%d)\n', s_idx, e_idx);
                    else
                        fprintf('⏩ Skipped pair (%d→%d): contains forbidden events (%s)\n', ...
                            s_idx, e_idx, strjoin(intersect(middle_types, forbidden), ', '));
                    end
                end

                if isempty(valid_pairs)
                    fprintf('No valid S released → S pressed segments to cut.\n');
                    continue;
                end

                cut_intervals = zeros(size(valid_pairs));
                for i = 1:size(valid_pairs,1)
                    cut_intervals(i,1) = EEG.event(valid_pairs(i,1)).latency;
                    cut_intervals(i,2) = EEG.event(valid_pairs(i,2)).latency;
                end
                cut_intervals = sortrows(cut_intervals,1);

                merged_intervals = [];
                merged_intervals(1,:) = cut_intervals(1,:);
                for i = 2:size(cut_intervals,1)
                    if cut_intervals(i,1) <= merged_intervals(end,2)
                        merged_intervals(end,2) = max(merged_intervals(end,2), cut_intervals(i,2));
                    else
                        merged_intervals(end+1,:) = cut_intervals(i,:);
                    end
                end

                fprintf('\nCutting %d valid intervals between S released → S pressed:\n', size(merged_intervals,1));
                EEG_temp = EEG;
                    for i = size(merged_intervals, 1):-1:1
                        start_sample = round(merged_intervals(i, 1));
                        end_sample = round(merged_intervals(i, 2));
                        end_sample = min(end_sample, EEG_temp.pnts);
                        
                        fprintf('\nCutting EEG samples %d to %d...\n', start_sample, end_sample);
                        
                        if isfield(EEG_temp, 'accelerometer') && isfield(EEG_temp.accelerometer, 'data')
                            
                            
                            orig_acc_size = size(EEG_temp.accelerometer.data, 2);
                            fprintf('Cutting EEG samples %d to %d...\n', start_sample, end_sample);
                            
                            
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
                        EEG = EEG_temp;
                        
                        if exist('acc_data', 'var') && ~isempty(acc_data)
                            start_time = (start_sample - 1) / EEG.srate;
                            end_time = (end_sample - 1) / EEG.srate;
                            
                            acc_start_sample = round(start_time * imu_fs) + 1;
                            acc_end_sample = round(end_time * imu_fs) + 1;

                            acc_start_sample = max(1, min(acc_start_sample, size(acc_data, 1)));
                            acc_end_sample = max(1, min(acc_end_sample, size(acc_data, 1)));
                            
                            if acc_start_sample <= acc_end_sample
                                fprintf('Cutting acc data %d to %d...\n', acc_start_sample, acc_end_sample);
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
                                fprintf('Cutting gyro data %d to %d...\n', gyro_start_sample, gyro_end_sample);
                                gyro_data_temp = [gyro_data(1:gyro_start_sample-1,:); gyro_data(gyro_end_sample+1:end,:)];
                                gyro_data = gyro_data_temp;
                            end
                        end
                    end
                EEG = EEG_temp;

            %=============================
            % Case 2: Regular 1/2/3 pressed/released
            %=============================
            else
                event_marker_pressed = pressed{hehe};
                event_marker_released = released{hehe};
                pressed_count = sum(strcmp({EEG.event.type}, event_marker_pressed));
                released_count = sum(strcmp({EEG.event.type}, event_marker_released));
                if pressed_count == released_count && pressed_count >1 && released_count >1
                    fprintf('found %d "%s" and %d "%s"\n', pressed_count, event_marker_pressed, released_count, event_marker_released);
                    fprintf('remove data inbetween %s and %s...\n', event_marker_released, event_marker_pressed);
                    
                    start_indices = find(strcmp({EEG.event.type}, event_marker_released));
                    end_indices = find(strcmp({EEG.event.type}, event_marker_pressed));
                    
                    fprintf('\n===== "%s" =====\n', event_marker_pressed);
                    fprintf('  Index   Type                 Time                       Latency\n');
                    fprintf('  ---------------------------------------------------------------\n');
                    for i = 1:length(end_indices)
                        idx = end_indices(i);
                        evt_type = EEG.event(idx).type;
                        evt_time = (EEG.event(idx).latency - 1) / EEG.srate;
                        evt_mins = floor(evt_time / 60);
                        evt_secs = mod(evt_time, 60);
                        fprintf('  %3d    %-20s  %d:%02d (%.2fs)   %d\n', idx, evt_type, evt_mins, evt_secs, evt_time, round(EEG.event(idx).latency));
                    end
    
                    fprintf('\n===== "%s" =====\n', event_marker_released);
                    fprintf('  Index   Type                 Time                       Latency\n');
                    fprintf('  ---------------------------------------------------------------\n');
                    for i = 1:length(start_indices)
                        idx = start_indices(i);
                        evt_type = EEG.event(idx).type;
                        evt_time = (EEG.event(idx).latency - 1) / EEG.srate;
                        evt_mins = floor(evt_time / 60);
                        evt_secs = mod(evt_time, 60);
                        fprintf('  %3d    %-20s  %d:%02d (%.2fs)   %d\n', idx, evt_type, evt_mins, evt_secs, evt_time, round(EEG.event(idx).latency));
                    end
                    
    
                    valid_segments = [];
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
                            fprintf('\n  Seg#   Start Idx  End Idx     Start Time                 End Time                 Duration\n');
                            fprintf('  ------------------------------------------------------------------------------------------\n');
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
                    
                    cut_intervals = [];
                    for i = 1:length(valid_segments)
                        segment = valid_segments(i);
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
                        
                        fprintf('\nCutting EEG samples %d to %d...\n', start_sample, end_sample);
                        
                        if isfield(EEG_temp, 'accelerometer') && isfield(EEG_temp.accelerometer, 'data')
                            
                            
                            orig_acc_size = size(EEG_temp.accelerometer.data, 2);
                            fprintf('Cutting EEG samples %d to %d...\n', start_sample, end_sample);
                            
                            
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
                        EEG = EEG_temp;

                        if exist('acc_data', 'var') && ~isempty(acc_data)
                            start_time = (start_sample - 1) / EEG.srate;
                            end_time = (end_sample - 1) / EEG.srate;
                            
                            acc_start_sample = round(start_time * imu_fs) + 1;
                            acc_end_sample = round(end_time * imu_fs) + 1;
                            
                            acc_start_sample = max(1, min(acc_start_sample, size(acc_data, 1)));
                            acc_end_sample = max(1, min(acc_end_sample, size(acc_data, 1)));
                            
                            if acc_start_sample <= acc_end_sample
                                fprintf('Cutting acc data %d to %d...\n', acc_start_sample, acc_end_sample);
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
                                fprintf('Cutting gyro data %d to %d...\n', gyro_start_sample, gyro_end_sample);
                                gyro_data_temp = [gyro_data(1:gyro_start_sample-1,:); gyro_data(gyro_end_sample+1:end,:)];
                                gyro_data = gyro_data_temp;
                            end
                        end
                    end
                end
            end % <-- else
        end % <-- for hehe loop
    end % <-- if EEG.event

    fprintf('\n===== Event Statistics After Cutting =====\n');
    if isfield(EEG, 'event') && ~isempty(EEG.event)
        event_types = unique({EEG.event.type}, 'stable');
        for i = 1:length(event_types)
            count = sum(strcmp({EEG.event.type}, event_types{i}));
            fprintf('  %s: %d\n', event_types{i}, count);
        end
        fprintf('Total events: %d\n', length(EEG.event));
    else
        fprintf('No events in EEG data after cutting\n');
    end
    
    % Visualize after cutting
    after_fig = plotEEGData(sub_idx, [task ' - After Cutting'], EEG, skip_channels, ...
                             acc_data, gyro_data, imu_fs, [], [], [], []);
    
    % Show reduction info
    new_duration = EEG.pnts / EEG.srate;
    fprintf('\n| Data length before cutting: %.2f seconds (%d samples)\n', orig_duration, EEG.pnts);
    fprintf('| Data length after cutting: %.2f seconds (%d samples)\n', new_duration, EEG.pnts);
    fprintf('| Reduced by: %.2f seconds (%.2f%%)\n', orig_duration - new_duration, ...
        100 * (orig_duration - new_duration) / orig_duration);
    
end