function [ts, stream] = load_xdf_eeg(xdf_path, stream_name)
%LOAD_XDF_EEG  EEG stream of an xdf file as [rows x samples]; column s = sample s of the cut tables.
%
%   [ts, stream] = load_xdf_eeg(xdf_path)               the first stream of type 'EEG'
%   [ts, stream] = load_xdf_eeg(xdf_path, stream_name)  the EEG stream with this name (segments.csv, column
%                                                       eeg_stream)
%
%   Needs load_xdf: EEGLAB plugin xdfimport (folder plugins/xdfimport*/xdf), or xdf-Matlab.
%   The cut tables count the samples of this stream in file order, as load_xdf returns them. Time stamps are not
%   used, so clock synchronization, clock-reset handling and jitter removal are off.
%
%   ts is single (the CGX stream is float32). The CGX headset's accelerometer is in the rows eeg_acc_rows of
%   segments.csv (22-24 with the 20-channel headset, 33-35 with the 30-channel headset): acc (g) = ts(rows, :) / 1e6.

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
