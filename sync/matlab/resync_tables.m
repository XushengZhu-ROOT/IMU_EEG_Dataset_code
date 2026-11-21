function T = resync_tables(tables_dir)
%RESYNC_TABLES  Read the cut tables segments.csv and pieces.csv into T.segments and T.pieces.
%
%   T = resync_tables()             tables/ of this repository
%   T = resync_tables(tables_dir)

if nargin < 1 || isempty(tables_dir)
    tables_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'tables');
end
T.dir = tables_dir;
T.segments = read_csv(fullfile(tables_dir, 'segments.csv'), ...
    {'eeg_samples', 'imu_slots', 'imu_empty', 'rho', 'check_imu_sum'});
T.pieces = read_csv(fullfile(tables_dir, 'pieces.csv'), {'piece', 'first', 'last', 'n', 'seg_first', 'seg_last'});
end


function t = read_csv(file, numeric)
% every column as text first; then the listed columns become doubles (empty = NaN)
opts = detectImportOptions(file, 'FileType', 'text', 'Delimiter', ',');
opts = setvartype(opts, 'string');
opts = setvaropts(opts, opts.VariableNames, 'FillValue', "");
t = readtable(file, opts);
for k = 1:numel(numeric)
    t.(numeric{k}) = str2double(t.(numeric{k}));
end
end
