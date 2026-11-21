function X = read_rabboni_txt(path)
%READ_RABBONI_TXT  Raw rabboni IMU txt -> [N x 6] double: accX accY accZ (g), gyroX gyroY gyroZ.
%
%   X = read_rabboni_txt(path)
%
%   Row i of X is txt row i of the cut tables: the i-th line of the file that starts with a number (leading spaces
%   and a minus sign allowed) and has at least 6 comma-separated numbers; the first 6 are kept. Header and trailer
%   lines are skipped, however many there are (13 header lines in most files; Sub12_all.txt has 19, plus 3 trailing
%   lines). The txt has no time stamps; its rows are read at 49.647537 Hz.

txt = fileread(char(path));
lines = regexp(txt, '\r\n|\n|\r', 'split')';
lines = lines(~cellfun('isempty', regexp(lines, '^\s*-?\d', 'once')));
fields = regexp(lines, ',', 'split');
fields = fields(cellfun('length', fields) >= 6);
F = cellfun(@(c) c(1:6), fields, 'UniformOutput', false);
V = str2double(vertcat(F{:}));
X = V(all(~isnan(V), 2), :);
end
