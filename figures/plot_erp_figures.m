function plot_erp_figures(resdir, figtitle, outname)
% ERPs at Pz to standard and target tones over all movement, one panel per cleaned version.
%   plot_erp_figures(resdir, figtitle, outname)
%   resdir    folder with the erp_averages.mat of erp_measures.m
%   figtitle  title above the panels
%   outname   output file without extension: writes <outname>.png (300 dpi) and <outname>.pdf
% Curves: mean over subjects; bands: +-1 SEM over subjects; grey: 300-500 ms. All panels share the y-axis.
S = load(fullfile(resdir, 'erp_averages.mat'), 'G', 'V', 't', 'chans');
G = S.G; t = S.t; pz = find(strcmp(S.chans, 'Pz'));
cS = [42 120 214] / 255; cT = [235 104 52] / 255;           % standard (blue), target (orange)
fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'centimeters', 'Position', [1 1 18 7], ...
    'DefaultAxesFontName', 'Helvetica', 'DefaultAxesFontSize', 8, 'DefaultAxesTickDir', 'out', ...
    'DefaultAxesBox', 'off', 'DefaultAxesLineWidth', 0.6, 'DefaultAxesXColor', 'k', 'DefaultAxesYColor', 'k', ...
    'DefaultTextFontSize', 8, 'DefaultTextColor', 'k');
tl = tiledlayout(fig, 1, numel(G), 'TileSpacing', 'compact', 'Padding', 'compact');
tl.OuterPosition = [0 0.04 1 0.85];
X = {};
for v = 1:numel(G), A = G{v}.allMovement; X = [X, {A.standard(:, pz, :), A.target(:, pz, :)}]; end %#ok<AGROW>
yl = shared_lim(X);
for v = 1:numel(G)
    ax = nexttile(tl); hold(ax, 'on');
    A = G{v}.allMovement;
    patch(ax, [300 500 500 300], [yl(1) yl(1) yl(2) yl(2)], [0.92 0.92 0.92], 'EdgeColor', 'none');
    band(ax, t, A.standard(:, pz, :), cS); band(ax, t, A.target(:, pz, :), cT);
    xline(ax, 0, ':', 'Color', 'k'); yline(ax, 0, 'Color', [0.8 0.8 0.8]);
    xlim(ax, [-200 800]); ylim(ax, yl);
    set(ax, 'XTick', [-200 0 300 500 800], 'XTickLabelRotation', 0, 'Layer', 'top');
    title(ax, S.V(v).label, 'FontWeight', 'normal');
    if v == 1, ylabel(ax, 'Pz (\muV)'); end
    xlabel(ax, '(ms)', 'FontSize', 7); lb = ax.XLabel; lb.Units = 'normalized'; lb.Position(2) = 0.8 * lb.Position(2);
end
annotation(fig, 'textbox', [0 0.945 1 0.05], 'String', figtitle, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', 'EdgeColor', 'none', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k', ...
    'Interpreter', 'none');
fig_legend(fig, {'standard', 'target'}, {cS, cT});
exportgraphics(fig, [outname '.png'], 'Resolution', 300);
exportgraphics(fig, [outname '.pdf'], 'ContentType', 'vector');
close(fig);
end

function band(ax, t, X, col)
% mean over subjects (rows) with a +-1 SEM band; X is subjects x 1 x time
X = reshape(X, size(X, 1), []);
m = mean(X, 1); se = std(X, 0, 1) / sqrt(size(X, 1));
fill(ax, [t fliplr(t)], [m + se fliplr(m - se)], col, 'FaceAlpha', 0.18, 'EdgeColor', 'none');
plot(ax, t, m, '-', 'Color', col, 'LineWidth', 1.1);
end

function yl = shared_lim(X)
% y-limits covering mean +-1 SEM of every array in X (each subjects x 1 x time), with 5% margin
lo = inf; hi = -inf;
for i = 1:numel(X)
    Y = reshape(X{i}, size(X{i}, 1), []); m = mean(Y, 1); se = std(Y, 0, 1) / sqrt(size(Y, 1));
    lo = min(lo, min(m - se)); hi = max(hi, max(m + se));
end
yl = [lo hi] + [-1 1] * 0.05 * (hi - lo);
end

function fig_legend(fig, labels, cols)
% legend in the top-left corner of the figure, outside the panels: a line sample, then the label
ax = axes(fig, 'Units', 'normalized', 'Position', [0.02 0.905 0.96 0.035], 'Visible', 'off'); hold(ax, 'on');
xlim(ax, [0 1]); ylim(ax, [0 1]); x = 0.005;
for i = 1:numel(labels)
    plot(ax, [x x + 0.03], [0.5 0.5], '-', 'Color', cols{i}, 'LineWidth', 1.4);
    tx = text(ax, x + 0.037, 0.5, labels{i}, 'Color', 'k', 'FontSize', 8, 'VerticalAlignment', 'middle', ...
        'Interpreter', 'none');
    e = tx.Extent; x = e(1) + e(3) + 0.025;
end
end
