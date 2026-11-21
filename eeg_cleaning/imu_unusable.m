function u = imu_unusable(bids, sub, task, acq)
% True if anomalies.tsv marks this segment's IMU unusable.
%   u = imu_unusable(bids, sub, task, acq)    e.g. imu_unusable(bids, '19', 'headshake', 'slowmove')
A = readtable(fullfile(bids, 'anomalies.tsv'), 'FileType', 'text', 'Delimiter', '\t', 'TextType', 'char');
tasks = {'walkstraight', 'walkcurved', 'headshake', 'headnod', 'pickup', 'stairs'}; speeds = {'slow', 'fast'};
u = false;
for r = 1:height(A)
    if ~strcmp(A.imu_usable{r}, 'no') || str2double(A.sub{r}(4:end)) ~= str2double(sub), continue, end
    m = regexp(A.cond{r}, 'movement(\d)_speed(\d)', 'tokens', 'once');
    if isempty(m), tk = 'rest'; aq = lower(A.cond{r});
    else, tk = tasks{str2double(m{1}) + 1}; aq = [speeds{str2double(m{2}) + 1} lower(A.seg{r})]; end
    if strcmp(tk, task) && strcmp(aq, acq), u = true; return, end
end
end
