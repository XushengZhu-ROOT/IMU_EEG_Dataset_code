function X = read_rabboni_txt(path)
%READ_RABBONI_TXT  Raw rabboni IMU txt -> [N x 6] double: accX accY accZ (g), gyroX gyroY gyroZ.
%
%   X = read_rabboni_txt(path)
%
%   Row i of X is txt row i of the cut tables (header and trailer lines skipped).

txt = fileread(char(path));
lines = regexp(txt, '\r\n|\n|\r', 'split')';
lines = lines(~cellfun('isempty', regexp(lines, '^\s*-?\d', 'once')));
fields = regexp(lines, ',', 'split');
fields = fields(cellfun('length', fields) >= 6);
F = cellfun(@(c) c(1:6), fields, 'UniformOutput', false);
V = str2double(vertcat(F{:}));
X = V(all(~isnan(V), 2), :);
end
