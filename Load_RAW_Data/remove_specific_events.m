function EEG_cleaned = remove_specific_events(EEG)
% REMOVE_SPECIFIC_EVENTS Remove events near specific time points in EEG data
%   EEG_cleaned = REMOVE_SPECIFIC_EVENTS(EEG, target_times, search_window)
%   Searches for events near given time points, lets the user select events
%   to delete, then returns the EEG data with those events removed
%
%   Inputs:
%       EEG - EEGLAB EEG structure
%       target_times - Array of time points to search (in seconds), e.g. [168, 337]
%       search_window - Search window size (seconds), default is 10 seconds
%
%   Output:
%       EEG_cleaned - EEG structure with selected events removed

% Set default parameters
if nargin < 3 || isempty(search_window)
    search_window = 10;  % Default search window is ±10 seconds
end

fprintf('======================================================\n')
fprintf('removed unnecessary event markers\n');
event_types = unique({EEG.event.type}, 'stable');

keep_events = {'S pressed', 'S released', '1 pressed', '1 released', ...
               '2 pressed', '2 released', '3 pressed', '3 released', ...
               'standard_audio', 'target_audio'};

indices_to_remove = [];
fprintf('"""\n')
for i = 1:length(event_types)
    if ~ismember(event_types{i}, keep_events)
        
        redundant_event_indices = find(strcmp({EEG.event.type}, event_types{i}));
        count = length(redundant_event_indices);
        fprintf('\nfound %d %s', count, event_types{i});
        
        indices_to_remove = [indices_to_remove, redundant_event_indices];
    end
end

indices_to_remove = sort(indices_to_remove, 'descend');
for idx = indices_to_remove
    EEG.event(idx) = [];
end

fprintf('  --->  removed %d events\n', length(indices_to_remove));
fprintf('"""\n')
fprintf('======================================================\n')
% Count events after removal
fprintf('Event summary after removal:\n');
event_types = unique({EEG.event.type}, 'stable');
for i = 1:length(event_types)
    count = sum(strcmp({EEG.event.type}, event_types{i}));
    fprintf('  %s: %d\n', event_types{i}, count);
end
fprintf('Total event count: %d\n\n', length(EEG.event));

end