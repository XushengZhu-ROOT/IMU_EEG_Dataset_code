function make_gyro_figure(bids, outdir)
% Plot clipped head-IMU gyroZ and its restoration by declip_gyro.m (must be on the path), 10 s of the Move segment
% with the most clipped samples.
%   make_gyro_figure(bids, outdir)
% Writes fig_gyro_declip.png, fig_gyro_declip.pdf and fig_gyro_declip_values.csv to outdir.
F = dir(fullfile(bids, 'sub-*', 'motion', '*move_motion.tsv'));
best = 0;
for i = 1:numel(F)
    M = readmatrix(fullfile(F(i).folder, F(i).name), 'FileType', 'text', 'Delimiter', '\t', 'TreatAsMissing', 'n/a');
    n = sum(abs(M(:, 6)) >= 990);                       % columns: accX accY accZ gyroX gyroY gyroZ latency
    if n > best, best = n; pick = i; end
end
name = F(pick).name(1:end - 11);                        % without _motion.tsv
M = readmatrix(fullfile(F(pick).folder, F(pick).name), 'FileType', 'text', 'Delimiter', '\t', 'TreatAsMissing', 'n/a');
t = M(:, 7); g = M(:, 6); y = declip_gyro(g);

% runs of clipped samples with the same sign; the 10-s window starts 5 s before the longest run that declip_gyro changed
s = sign(g) .* (abs(g) >= 990); s(isnan(s)) = 0;
runs = zeros(0, 2); k = 1;
while k <= numel(s)
    if s(k) == 0, k = k + 1; continue, end
    j = k; while j < numel(s) && s(j + 1) == s(k), j = j + 1; end
    runs(end + 1, :) = [k j]; k = j + 1; %#ok<AGROW>
end
len = runs(:, 2) - runs(:, 1) + 1;
fixed = arrayfun(@(r) any(y(runs(r, 1):runs(r, 2)) ~= g(runs(r, 1):runs(r, 2))), (1:size(runs, 1))');
[~, r] = max(len .* fixed);
s0 = t(runs(r, 1)) - 5; w = find(t >= s0 & t < s0 + 10);
fprintf('%s: %d clipped gyroZ samples in %d runs (%d changed); window %.3f-%.3f s of the segment, %d samples\n', ...
    name, best, size(runs, 1), sum(fixed), s0, s0 + 10, numel(w));

% title from the file name
tok = regexp(name, '^sub-(\d+)_task-([a-z]+)_tracksys-imu_acq-([a-z]+)move$', 'tokens', 'once');
words = containers.Map({'walkstraight', 'walkcurved', 'headshake', 'headnod', 'pickup', 'stairs'}, ...
    {'walk straight', 'walk curved', 'head shake', 'head nod', 'pick up', 'stairs'});
figtitle = sprintf('%s%s %s (Sub%s)', upper(tok{3}(1)), tok{3}(2:end), words(tok{2}), tok{1});
cBlue = [42 120 214] / 255;
fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'centimeters', 'Position', [1 1 18 6.2]);
ax = axes(fig, 'Units', 'normalized', 'Position', [0.085 0.15 0.905 0.61]); hold(ax, 'on');
plot(ax, t(w) - s0, y(w), '-', 'Color', cBlue, 'LineWidth', 1);
plot(ax, t(w) - s0, g(w), '-', 'Color', 'k', 'LineWidth', 1);
yline(ax, [992.2 -1000], ':', 'Color', 'k', 'LineWidth', 0.6);
set(ax, 'FontName', 'Helvetica', 'FontSize', 8, 'TickDir', 'out', 'Box', 'off', 'LineWidth', 0.6, ...
    'XColor', 'k', 'YColor', 'k', 'XLim', [0 10], 'XTick', 0:10, 'YLim', [-1500 1500], 'YTick', -1500:500:1500, ...
    'Layer', 'top');
ylabel(ax, 'gyroZ (deg/s)', 'Color', 'k');
xlabel(ax, '(s)', 'FontSize', 7, 'Color', 'k'); lb = ax.XLabel; lb.Units = 'normalized'; lb.Position(2) = 0.8 * lb.Position(2);
annotation(fig, 'textbox', [0 0.9 1 0.09], 'String', figtitle, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', 'EdgeColor', 'none', 'FontName', 'Helvetica', 'FontSize', 11, ...
    'FontWeight', 'bold', 'Color', 'k', 'Interpreter', 'none');
fig_legend(fig, {'recorded', 'restored (cubic spline)'}, {'k', cBlue});
exportgraphics(fig, fullfile(outdir, 'fig_gyro_declip.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(outdir, 'fig_gyro_declip.pdf'), 'ContentType', 'vector');
close(fig);

T = table(t(w), t(w) - s0, g(w), y(w), 'VariableNames', {'latency_s', 'time_in_figure_s', 'gyroZ_recorded', ...
    'gyroZ_restored'});
writetable(T, fullfile(outdir, 'fig_gyro_declip_values.csv'));
fprintf('restored samples in the window: %d; largest |value| recorded %.1f, restored %.1f\n', ...
    sum(y(w) ~= g(w)), max(abs(g(w))), max(abs(y(w))));
end

function fig_legend(fig, labels, cols)
% legend above the plot: a line sample, then the label
ax = axes(fig, 'Units', 'normalized', 'Position', [0.02 0.8 0.96 0.07], 'Visible', 'off'); hold(ax, 'on');
xlim(ax, [0 1]); ylim(ax, [0 1]); x = 0.005;
for i = 1:numel(labels)
    plot(ax, [x x + 0.03], [0.5 0.5], '-', 'Color', cols{i}, 'LineWidth', 1.4);
    tx = text(ax, x + 0.037, 0.5, labels{i}, 'Color', 'k', 'FontName', 'Helvetica', 'FontSize', 8, ...
        'VerticalAlignment', 'middle', 'Interpreter', 'none');
    e = tx.Extent; x = e(1) + e(3) + 0.025;
end
end
