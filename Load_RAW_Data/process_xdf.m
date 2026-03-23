function EEG = process_xdf(filename, device, trimToLastRelease, low)
% Process XDF file: extract EEG, add keyboard/audio markers via LSL sync.
% EEG = process_xdf(filename, device, trimToLastRelease, low)
% trimToLastRelease: if true, trim to last released event

if nargin < 3
    trimToLastRelease = true;
end
if nargin < 4
    low = 1;
end

if ~exist('ALLEEG', 'var')
    [ALLEEG, EEG, CURRENTSET] = eeglab;
else
    eeglab redraw;
end

fprintf('...XDF file loading: %s\n', filename);
[streams, fileheader] = load_xdf(filename);

num_streams = length(streams);
fprintf('Found %d streams\n', num_streams);

for i = 1:length(streams)
    fprintf('Stream %d:\n', i);
    fprintf('  Name: %s\n', streams{i}.info.name);
    fprintf('  Type: %s\n', streams{i}.info.type);
    fprintf('  Channels: %d\n', streams{i}.info.channel_count);
    fprintf('  Sample rate: %s Hz\n', streams{i}.info.nominal_srate);
    fprintf('  Data points: %d\n', size(streams{i}.time_series, 2));
    fprintf('\n');
end

% EEG stream: CGX Quick-20r Q20r-0162 (20ch) or CGX Quick-32r Q32r-0584 (30ch)
if device == 20
    eegStreamIdx = find_stream_by_name(streams, 'CGX Quick-20r Q20r-0162');
elseif device == 30
    eegStreamIdx = find_stream_by_name(streams, 'CGX Quick-32r Q32r-0584');
end

if eegStreamIdx == -1
    error('EEG data stream not found.');
end

eegStream = streams{eegStreamIdx};

keyboardStreamIdx = find_stream_by_name(streams, 'Keyboard');
if keyboardStreamIdx == -1
    warning('Keyboard stream not found, proceeding without keyboard markers');
    keyboardStream = [];
else
    keyboardStream = streams{keyboardStreamIdx};
    fprintf('Found Keyboard stream: %d events\n', length(keyboardStream.time_stamps));
end

audioStreamIdx = find_stream_by_name(streams, 'AudioMarkers');
if audioStreamIdx == -1
    warning('AudioMarkers stream not found, proceeding without audio markers');
    audioStream = [];
else
    audioStream = streams{audioStreamIdx};
    fprintf('Found AudioMarkers stream: %d events\n', length(audioStream.time_stamps));
end

timeSyncIdx = find_stream_by_name(streams, 'TimeSync');
if timeSyncIdx ~= -1
    timeSyncStream = streams{timeSyncIdx};
    fprintf('Found TimeSync stream for precise sync\n');
    try
        if ~isempty(timeSyncStream.time_series)
            fprintf('Analyzing TimeSync...\n');
            if iscell(timeSyncStream.time_series)
                syncData = timeSyncStream.time_series{1};
            else
                syncData = timeSyncStream.time_series;
            end
            if ischar(syncData)
                try
                    syncInfo = jsondecode(syncData);
                    fprintf('Sync: LSL=%.4f, sys_time=%s, offset=%.4f\n', ...
                            syncInfo.lsl_time, syncInfo.readable_time, syncInfo.offset);
                catch
                    fprintf('Could not parse TimeSync: %s\n', syncData);
                end
            end
        end
    catch e
        fprintf('TimeSync error: %s\n', e.message);
    end
end

srate = 500;
eegData = eegStream.time_series(1:device, :);
fprintf('\nEEG channels: %d\n', size(eegStream.time_series, 1));

eegTimestamps = eegStream.time_stamps;

scale_factor = 1000000;
if device == 20
    accelerometer_data = eegStream.time_series(22:24, :)/scale_factor;
elseif device == 30
    accelerometer_data = eegStream.time_series(33:35, :)/scale_factor;
end

fprintf('Accelerometer range (g):\n');
for i = 1:3
    axis_names = {'X', 'Y', 'Z'};
    fprintf('  %s: %.3f to %.3f\n', axis_names{i}, ...
            min(accelerometer_data(i,:)), max(accelerometer_data(i,:)));
end

gravity_magnitude = sqrt(mean(accelerometer_data(1,:))^2 + ...
                        mean(accelerometer_data(2,:))^2 + ...
                        mean(accelerometer_data(3,:))^2);
fprintf('Mean gravity: %.3f g\n', gravity_magnitude);

EEG = create_eeg_struct(eegData, eegTimestamps, srate, eegStream, accelerometer_data);
EEG = load_channel_locations(EEG);

if ~isempty(keyboardStream)
    EEG = add_keyboard_events_lsl_sync(EEG, keyboardStream, eegStream);
end

if ~isempty(audioStream)
    EEG = add_audio_events_lsl_sync(EEG, audioStream, eegStream);
end

[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0, 'setname', 'EEG_data', 'gui', 'on');
eeglab redraw;

fprintf('Applying filter...\n');
EEG = filter_eeg(EEG, low);

[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'setname', 'EEG_filtered', 'overwrite', 'off', 'gui', 'on');
eeglab redraw;

if trimToLastRelease
    fprintf('\n==== Trimming to last released event ====\n');
    EEG = trim_eeg_to_last_release(EEG);
    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'setname', 'EEG_trimmed', 'overwrite', 'off', 'gui', 'on');
    eeglab redraw;
end

if isfield(EEG, 'event') && ~isempty(EEG.event)
    fprintf('\n==== Event summary ====\n');
    eventTypes = {EEG.event.type};
    uniqueTypes = unique(eventTypes);
    fprintf('Total events: %d\n', length(EEG.event));
    for i = 1:length(uniqueTypes)
        count = sum(strcmp(eventTypes, uniqueTypes{i}));
        fprintf('  %s: %d\n', uniqueTypes{i}, count);
    end
end

fprintf('Done.\n');
end

function EEG = add_audio_events_lsl_sync(EEG, audioStream, eegStream)
if ~isfield(audioStream, 'time_stamps') || isempty(audioStream.time_stamps)
    warning('Audio stream has no timestamps, skipping');
    return;
end

if ~isfield(EEG, 'event') || isempty(EEG.event)
    EEG.event = struct('type', {}, 'latency', {}, 'urevent', {});
end

numEvents = length(audioStream.time_stamps);
fprintf('Processing %d audio events...\n', numEvents);

if ~iscell(audioStream.time_series) && numEvents == 1
    audioStream.time_series = {audioStream.time_series};
elseif ~iscell(audioStream.time_series)
    temp = cell(1, size(audioStream.time_series, 2));
    for i = 1:size(audioStream.time_series, 2)
        temp{i} = audioStream.time_series(:, i);
    end
    audioStream.time_series = temp;
end

eeg_lsl_start = min(eegStream.time_stamps);
eeg_lsl_end = max(eegStream.time_stamps);
audio_lsl_start = min(audioStream.time_stamps);
audio_lsl_end = max(audioStream.time_stamps);
time_diff = audio_lsl_start - eeg_lsl_start;
fprintf('Audio stream starts %.4fs after EEG\n', time_diff);

samples_per_second = EEG.srate;
if isfield(EEG, 'event') && ~isempty(EEG.event)
    eventCount = length(EEG.event);
else
    eventCount = 0;
end

if ~isempty(EEG.event)
    existingFields = fieldnames(EEG.event);
else
    existingFields = {'type', 'latency', 'urevent'};
end

for i = 1:numEvents
    audio_lsl_time = audioStream.time_stamps(i);
    relative_lsl_time = audio_lsl_time - eeg_lsl_start;
    eventSample = round(relative_lsl_time * samples_per_second) + 1;

    if eventSample < 1
        warning('Audio event #%d before EEG start (%.4fs), set to sample 1', i, relative_lsl_time);
        eventSample = 1;
    elseif eventSample > EEG.pnts
        warning('Audio event #%d after EEG end (%.4fs), set to last sample', i, relative_lsl_time);
        eventSample = EEG.pnts;
    end

    if iscell(audioStream.time_series) && i <= length(audioStream.time_series)
        eventData = audioStream.time_series{i};
        if ischar(eventData)
            try
                markerParts = strsplit(eventData, ',');
                if length(markerParts) >= 1
                    stimType = markerParts{1};
                    if strcmpi(stimType, 'standard')
                        eventType = 'standard_audio';
                    elseif strcmpi(stimType, 'target')
                        eventType = 'target_audio';
                    else
                        eventType = ['audio_', stimType];
                    end
                    if length(markerParts) >= 2
                        trialNum = str2double(markerParts{2});
                    else
                        trialNum = i;
                    end
                    if length(markerParts) >= 3
                        stimId = str2double(markerParts{3});
                    else
                        stimId = -1;
                    end
                else
                    eventType = ['AudioEvent_', num2str(i)];
                    trialNum = i;
                    stimId = -1;
                end
            catch
                eventType = ['AudioEvent_', num2str(i)];
                trialNum = i;
                stimId = -1;
            end
        else
            eventType = ['AudioEvent_', num2str(i)];
            trialNum = i;
            stimId = -1;
        end
    else
        eventType = ['AudioEvent_', num2str(i)];
        trialNum = i;
        stimId = -1;
    end

    newEvent = struct();
    for f = 1:length(existingFields)
        fieldName = existingFields{f};
        if strcmp(fieldName, 'type')
            newEvent.(fieldName) = eventType;
        elseif strcmp(fieldName, 'latency')
            newEvent.(fieldName) = eventSample;
        elseif strcmp(fieldName, 'urevent')
            newEvent.(fieldName) = eventCount + i;
        else
            if strcmp(fieldName, 'trial') && exist('trialNum', 'var')
                newEvent.(fieldName) = trialNum;
            elseif strcmp(fieldName, 'stimId') && exist('stimId', 'var')
                newEvent.(fieldName) = stimId;
            else
                if ~isempty(EEG.event)
                    firstValue = EEG.event(1).(fieldName);
                    if isnumeric(firstValue)
                        newEvent.(fieldName) = 0;
                    elseif ischar(firstValue)
                        newEvent.(fieldName) = '';
                    elseif islogical(firstValue)
                        newEvent.(fieldName) = false;
                    else
                        newEvent.(fieldName) = [];
                    end
                else
                    newEvent.(fieldName) = [];
                end
            end
        end
    end

    if isempty(EEG.event)
        EEG.event = newEvent;
    else
        EEG.event(end+1) = newEvent;
    end

    fprintf('Added audio: %s, LSL=%.4f, sample=%d\n', eventType, audio_lsl_time, eventSample);
end

if ~isempty(EEG.event)
    [~, sortIdx] = sort([EEG.event.latency]);
    EEG.event = EEG.event(sortIdx);
    for i = 1:length(EEG.event)
        EEG.event(i).urevent = i;
    end

    eventTypes = {EEG.event.type};
    standard_count = sum(strcmp(eventTypes, 'standard_audio'));
    target_count = sum(strcmp(eventTypes, 'target_audio'));
    audioEvents = find(startsWith(eventTypes, 'standard_audio') | ...
                       startsWith(eventTypes, 'target_audio'));

    if ~isempty(audioEvents)
        audioLatencies = [EEG.event(audioEvents).latency];
        fprintf('Audio event sample range: %d - %d (total: %d)\n', ...
                min(audioLatencies), max(audioLatencies), EEG.pnts);
        start_pct = min(audioLatencies) / EEG.pnts * 100;
        end_pct = max(audioLatencies) / EEG.pnts * 100;
        fprintf('Distribution: %.2f%% - %.2f%%\n', start_pct, end_pct);
    end

    fprintf('Added %d audio events (standard: %d, target: %d)\n', ...
            standard_count + target_count, standard_count, target_count);
end

return;
end

function idx = find_stream_by_name(streams, name)
idx = -1;
for i = 1:length(streams)
    if isfield(streams{i}, 'info') && isfield(streams{i}.info, 'name')
        if strcmp(streams{i}.info.name, name)
            idx = i;
            return;
        end
    end
end
end

function srate = get_sampling_rate(stream)
if isfield(stream.info, 'effective_srate') && stream.info.effective_srate > 0
    srate = stream.info.effective_srate;
elseif isfield(stream.info, 'nominal_srate') && stream.info.nominal_srate > 0
    srate = stream.info.nominal_srate;
else
    srate = round(length(stream.time_stamps) / (stream.time_stamps(end) - stream.time_stamps(1)));
end
fprintf('EEG sample rate: %.2f Hz\n', srate);
end

function EEG = create_eeg_struct(eegData, timestamps, srate, stream, accelerometer_data)
EEG = eeg_emptyset;
EEG.setname = 'EEG_data';
EEG.data = eegData;
EEG.nbchan = size(eegData, 1);
EEG.pnts = size(eegData, 2);
EEG.trials = 1;
EEG.srate = srate;
EEG.xmin = 0;
EEG.xmax = (EEG.pnts-1)/EEG.srate;
EEG.times = linspace(EEG.xmin, EEG.xmax, EEG.pnts) * 1000;
EEG.ref = '';
EEG.accelerometer.data = double(accelerometer_data);
EEG.accelerometer.labels = {'ACC_X', 'ACC_Y', 'ACC_Z'};
EEG.accelerometer.times = linspace(EEG.xmin, EEG.xmax, EEG.pnts);
EEG.accelerometer.x = accelerometer_data(1, :);
EEG.accelerometer.y = accelerometer_data(2, :);
EEG.accelerometer.z = accelerometer_data(3, :);

if isfield(stream, 'info') && isfield(stream.info, 'desc') && ...
   isfield(stream.info.desc, 'channels') && isfield(stream.info.desc.channels, 'channel')
    channels = stream.info.desc.channels.channel;
    for i = 1:min(EEG.nbchan, length(channels))
        if isfield(channels{i}, 'label')
            EEG.chanlocs(i).labels = channels{i}.label;
        else
            EEG.chanlocs(i).labels = ['Chan' num2str(i)];
        end
    end
end

return;
end

function EEG = add_keyboard_events_lsl_sync(EEG, keyboardStream, eegStream)
if ~isfield(keyboardStream, 'time_stamps') || isempty(keyboardStream.time_stamps)
    warning('Keyboard stream has no timestamps, skipping');
    return;
end

if ~isfield(EEG, 'event') || isempty(EEG.event)
    EEG.event = struct('type', {}, 'latency', {}, 'urevent', {});
end

numEvents = length(keyboardStream.time_stamps);
fprintf('Processing %d keyboard events...\n', numEvents);

if ~iscell(keyboardStream.time_series) && numEvents == 1
    keyboardStream.time_series = {keyboardStream.time_series};
elseif ~iscell(keyboardStream.time_series)
    temp = cell(1, size(keyboardStream.time_series, 2));
    for i = 1:size(keyboardStream.time_series, 2)
        temp{i} = keyboardStream.time_series(:, i);
    end
    keyboardStream.time_series = temp;
end

eeg_lsl_start = min(eegStream.time_stamps);
eeg_lsl_end = max(eegStream.time_stamps);
kb_lsl_start = min(keyboardStream.time_stamps);
kb_lsl_end = max(keyboardStream.time_stamps);
time_diff = kb_lsl_start - eeg_lsl_start;
fprintf('Keyboard stream starts %.4fs after EEG\n', time_diff);

samples_per_second = EEG.srate;

for i = 1:numEvents
    kb_lsl_time = keyboardStream.time_stamps(i);
    relative_lsl_time = kb_lsl_time - eeg_lsl_start;
    eventSample = round(relative_lsl_time * samples_per_second) + 1;

    if eventSample < 1
        warning('Event #%d before EEG start (%.4fs), set to sample 1', i, relative_lsl_time);
        eventSample = 1;
    elseif eventSample > EEG.pnts
        warning('Event #%d after EEG end (%.4fs), set to last sample', i, relative_lsl_time);
        eventSample = EEG.pnts;
    end

    if iscell(keyboardStream.time_series) && i <= length(keyboardStream.time_series)
        eventData = keyboardStream.time_series{i};
        if ischar(eventData)
            eventType = eventData;
        elseif isnumeric(eventData) && length(eventData) == 1
            eventType = ['KEY_', num2str(eventData)];
        elseif isstruct(eventData) && isfield(eventData, 'key')
            eventType = ['KEY_', eventData.key];
        else
            try
                eventType = char(eventData);
            catch
                eventType = ['KeyEvent_', num2str(i)];
            end
        end
    else
        eventType = ['KeyEvent_', num2str(i)];
    end

    newEvent = struct('type', eventType, 'latency', eventSample, 'urevent', i);
    if isempty(EEG.event)
        EEG.event = newEvent;
    else
        EEG.event(end+1) = newEvent;
    end

    fprintf('Added event: %s, LSL=%.4f, sample=%d\n', eventType, kb_lsl_time, eventSample);
end

if ~isempty(EEG.event)
    [~, sortIdx] = sort([EEG.event.latency]);
    EEG.event = EEG.event(sortIdx);
    for i = 1:length(EEG.event)
        EEG.event(i).urevent = i;
    end

    latencies = [EEG.event.latency];
    fprintf('Event sample range: %d - %d (total: %d)\n', min(latencies), max(latencies), EEG.pnts);
    start_pct = min(latencies) / EEG.pnts * 100;
    end_pct = max(latencies) / EEG.pnts * 100;
    fprintf('Distribution: %.2f%% - %.2f%%\n', start_pct, end_pct);
    fprintf('Added %d keyboard events\n', length(EEG.event));
end

return;
end

function EEG = load_channel_locations(EEG)
try
    EEG = pop_chanedit(EEG, 'lookup', 'standard_1005.elc');
catch
    try
        EEG = pop_chanedit(EEG, 'lookup', fullfile(fileparts(which('eeglab')), 'plugins', 'dipfit', 'standard_BEM', 'elec', 'standard_1005.elc'));
    catch
        warning('Could not load channel locations');
    end
end
return;
end

function EEG = filter_eeg(EEG, low)
EEG = pop_eegfiltnew(EEG, 'locutoff', low);
EEG = pop_eegfiltnew(EEG, 'hicutoff', 50);
return;
end

function EEG = trim_eeg_to_last_release(EEG)
% Trim EEG to last released event; also trim accelerometer data.

if ~isfield(EEG, 'event') || isempty(EEG.event)
    warning('No events in EEG, cannot trim');
    return;
end

releasedIndices = [];
for i = 1:length(EEG.event)
    if ischar(EEG.event(i).type) && contains(lower(EEG.event(i).type), 'released')
        releasedIndices(end+1) = i;
    end
end

if isempty(releasedIndices)
    warning('No released events found, cannot trim');
    return;
end

lastReleaseIdx = releasedIndices(end);
lastReleaseSample = EEG.event(lastReleaseIdx).latency;
bufferSamples = round(EEG.srate * 1);
endSample = round(lastReleaseSample) + bufferSamples;
endSample = min(endSample, EEG.pnts);

fprintf('Original EEG: %.2fs (%d samples)\n', EEG.xmax, EEG.pnts);
fprintf('Last released: %s at %.2fs (sample %d)\n', ...
        EEG.event(lastReleaseIdx).type, ...
        (lastReleaseSample-1)/EEG.srate, round(lastReleaseSample));
fprintf('Trimming to %.2fs (sample %d) with 1s buffer\n', (endSample-1)/EEG.srate, endSample);

if isfield(EEG, 'accelerometer') && isfield(EEG.accelerometer, 'data')
    fprintf('Trimming accelerometer data...\n');
    orig_acc_size = size(EEG.accelerometer.data, 2);
    EEG.accelerometer.data = EEG.accelerometer.data(:, 1:endSample);
    EEG.accelerometer.x = EEG.accelerometer.data(1, :);
    EEG.accelerometer.y = EEG.accelerometer.data(2, :);
    EEG.accelerometer.z = EEG.accelerometer.data(3, :);
    EEG.accelerometer.times = linspace(EEG.xmin, (endSample-1)/EEG.srate, endSample);
    fprintf('Accelerometer: %d -> %d samples\n', orig_acc_size, size(EEG.accelerometer.data, 2));
end

EEG = pop_select(EEG, 'point', [1 endSample]);
EEG.xmax = (EEG.pnts-1) / EEG.srate;
EEG.times = linspace(EEG.xmin, EEG.xmax, EEG.pnts) * 1000;

if isfield(EEG, 'accelerometer')
    if size(EEG.accelerometer.data, 2) == EEG.pnts
        fprintf('Accelerometer and EEG lengths match\n');
    else
        warning('Accelerometer (%d) and EEG (%d) length mismatch', ...
                size(EEG.accelerometer.data, 2), EEG.pnts);
    end
end

fprintf('Trimmed EEG: %.2fs (%d samples)\n', EEG.xmax, EEG.pnts);
return;
end
