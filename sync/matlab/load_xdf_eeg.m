function [ts, stream] = load_xdf_eeg(xdf_path, stream_name)
%LOAD_XDF_EEG  EEG stream of an xdf file as [rows x samples]; column s = sample s of the cut tables.
%
%   [ts, stream] = load_xdf_eeg(xdf_path)               the first EEG stream
%   [ts, stream] = load_xdf_eeg(xdf_path, stream_name)  the EEG stream with this name (eeg_stream in segments.csv)
%
%   Needs load_xdf (EEGLAB plugin xdfimport, or xdf-Matlab). Headset accelerometer (g): ts(eeg_acc_rows, :) / 1e6.

xdf_path = char(xdf_path);
streams = load_xdf(xdf_path, 'HandleClockSynchronization', false, 'HandleClockResets', false, ...
    'HandleJitterRemoval', false);
for k = 1:numel(streams)
    info = streams{k}.info;
    if strcmpi(info.type, 'EEG') && (nargin < 2 || isempty(stream_name) || strcmp(info.name, char(stream_name)))
        stream = streams{k};
        ts = stream.time_series;
        return
    end
end
if nargin >= 2 && ~isempty(stream_name)
    error('load_xdf_eeg:noStream', 'No EEG stream named "%s" in %s.', char(stream_name), xdf_path);
end
error('load_xdf_eeg:noStream', 'No EEG stream in %s.', xdf_path);
end
