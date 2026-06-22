function plot_erp_comparison(EEG_A, EEG_B, title_text, save_path, varargin)
%PLOT_ERP_COMPARISON Plot ERP overlay of two EEG datasets with topographic layout
%   With grid lines and confidence interval shading
%
%   INPUTS:
%     EEG_A, EEG_B  - EEGLAB EEG structures (same channels, sampling rate)
%     title_text    - Title string for the figure
%     save_path     - File path to save the resulting figure (PNG)
%     varargin      - Optional parameters in 'key', 'val' format:
%                     'timerange'   - [min max] time range to plot in ms
%                     'gridcolor'   - Color of grid lines [r g b] (default [0.8 0.8 0.8])
%                     'gridalpha'   - Transparency of grid (default 0.5)
%                     'cialpha'     - Transparency of CI shading (default 0.3)
%
% Example:
%   plot_erp_comparison(EEG_target, EEG_standard, 'Eye Shifting', 'results/eye_shifting.png', ...
%                      'timerange', [-100 600]);

% Parse input parameters with a simpler approach
p = struct(...
    'timerange', [], ...
    'gridcolor', [0.8 0.8 0.8], ...
    'gridalpha', 0.5, ...
    'cialpha', 0.1, ...
    'linewidth', 0.7 ...
);

% Process varargin
if nargin > 4
    for i = 1:2:length(varargin)
        if i+1 <= length(varargin)
            field = varargin{i};
            val = varargin{i+1};
            if isfield(p, field)
                p.(field) = val;
            else
                warning(['Unknown parameter: ' field]);
            end
        end
    end
end

% Set timerange if not provided
if isempty(p.timerange)
    p.timerange = [EEG_A.xmin*1000 EEG_A.xmax*1000];
end

% Validate inputs
if EEG_A.nbchan ~= EEG_B.nbchan || EEG_A.srate ~= EEG_B.srate
    error('EEG_A and EEG_B must have the same number of channels and sampling rate');
end

% Get ERP data
if ndims(EEG_A.data) == 3  % Has epochs
    % Calculate ERP
    erp_A = mean(EEG_A.data, 3);
    
    % Calculate standard error of the mean for confidence intervals
    sem_A = std(EEG_A.data, 0, 3) / sqrt(size(EEG_A.data, 3));
    
    % 95% confidence intervals
    ci_A_lower = erp_A - 1.96 * sem_A;
    ci_A_upper = erp_A + 1.96 * sem_A;
else
    % Already averaged data
    erp_A = EEG_A.data;
    ci_A_lower = [];
    ci_A_upper = [];
end

if ndims(EEG_B.data) == 3  % Has epochs
    % Calculate ERP
    erp_B = mean(EEG_B.data, 3);
    
    % Calculate standard error of the mean for confidence intervals
    sem_B = std(EEG_B.data, 0, 3) / sqrt(size(EEG_B.data, 3));
    
    % 95% confidence intervals
    ci_B_lower = erp_B - 1.96 * sem_B;
    ci_B_upper = erp_B + 1.96 * sem_B;
else
    % Already averaged data
    erp_B = EEG_B.data;
    ci_B_lower = [];
    ci_B_upper = [];
end

% Get time vector in milliseconds
times = linspace(EEG_A.xmin*1000, EEG_A.xmax*1000, EEG_A.pnts);

% Get data indices for the specified time range
timeIdx = (times >= p.timerange(1) & times <= p.timerange(2));
plotTimes = times(timeIdx);
plotData_A = erp_A(:, timeIdx);
plotData_B = erp_B(:, timeIdx);

if ~isempty(ci_A_lower)
    plotCI_A_lower = ci_A_lower(:, timeIdx);
    plotCI_A_upper = ci_A_upper(:, timeIdx);
end

if ~isempty(ci_B_lower)
    plotCI_B_lower = ci_B_lower(:, timeIdx);
    plotCI_B_upper = ci_B_upper(:, timeIdx);
end

% Set width and height parameters for plots based on channel count
disp(['Number of channels: ', num2str(size(EEG_A.data, 1))]);
if size(EEG_A.data, 1) >= 30
    w = 0.055;
    h = 0.04;
elseif size(EEG_A.data, 1) >= 20
    w = 0.07;
    h = 0.05;
else
    w = 0.09;  % For fewer channels, make plots larger
    h = 0.07;
end

% Get topographic layout positions
try
    % Try with standard EEGLAB format
    positions = get_standard_layout(EEG_A.chanlocs, 1.5);
catch
    try
        % Try with different parameters
        positions = get_standard_layout(EEG_A.chanlocs, 1.5, 0.07);
    catch
        % Try with all parameters
        positions = get_standard_layout(EEG_A.chanlocs, 1.5, w, h);
    end
end

% Create figure
figure('Color', 'w', 'Position', [100, 100, 800, 800]);
p1 = []; p2 = []; fill1 = []; fill2 = [];

% Plot each channel
for ch = 1:size(positions, 1)
    ax = axes('Position', positions(ch,:));
    hold on;
    
    % Add grid first so it's in background
    % grid on;
    % set(ax, 'GridColor', p.gridcolor, 'GridAlpha', p.gridalpha);
    
    % % Plot confidence intervals if available
    % if ~isempty(ci_A_lower)
    %     % Create fill area for confidence interval
    %     fill1 = fill([plotTimes, fliplr(plotTimes)], ...
    %          [plotCI_A_lower(ch,:), fliplr(plotCI_A_upper(ch,:))], ...
    %          'r', 'FaceAlpha', p.cialpha, 'EdgeColor', 'none');
    % end
    % 
    % if ~isempty(ci_B_lower)
    %     % Create fill area for confidence interval
    %     fill2 = fill([plotTimes, fliplr(plotTimes)], ...
    %          [plotCI_B_lower(ch,:), fliplr(plotCI_B_upper(ch,:))], ...
    %          'b', 'FaceAlpha', p.cialpha, 'EdgeColor', 'none');
    % end
    
    % Plot mean ERP lines
    h1 = plot(plotTimes, plotData_A(ch,:), 'r', 'LineWidth', p.linewidth);
    h2 = plot(plotTimes, plotData_B(ch,:), 'b', 'LineWidth', p.linewidth);
    
    % Draw reference lines
    yline(0, 'Color', [0.6 0.6 0.6], 'LineStyle', '-', 'LineWidth', 0.5); 
    xline(0, 'Color', [0.6 0.6 0.6], 'LineStyle', '-', 'LineWidth', 0.5);
    
    % Set channel title
    title(EEG_A.chanlocs(ch).labels, 'FontSize', 6);
    
    % Set axis limits
    xlim(p.timerange);
    
    % Keep grid but turn off box and axis labels for cleaner look
    % box off;
    % set(ax, 'XTickLabel', [], 'YTickLabel', []);
    axis off;
    
    % Store handles for legend
    if ch == 1
        p1 = h1; p2 = h2;
    end
    
    % Create and save individual channel plots if save_path provided
    if nargin > 3 && ~isempty(save_path)
        mini_fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 400, 150]);
        hold on;
        
        % Add grid to mini figure
        grid on;
        set(gca, 'GridColor', p.gridcolor, 'GridAlpha', p.gridalpha);
        
        % Add confidence intervals to mini figure
        if ~isempty(ci_A_lower)
            fill([plotTimes, fliplr(plotTimes)], ...
                 [plotCI_A_lower(ch,:), fliplr(plotCI_A_upper(ch,:))], ...
                 'r', 'FaceAlpha', p.cialpha, 'EdgeColor', 'none');
        end
        
        if ~isempty(ci_B_lower)
            fill([plotTimes, fliplr(plotTimes)], ...
                 [plotCI_B_lower(ch,:), fliplr(plotCI_B_upper(ch,:))], ...
                 'b', 'FaceAlpha', p.cialpha, 'EdgeColor', 'none');
        end
        
        % Plot ERP lines
        plot(plotTimes, plotData_A(ch,:), 'r', 'LineWidth', 0.5);
        plot(plotTimes, plotData_B(ch,:), 'b', 'LineWidth', 0.5);
        
        % Reference lines
        yline(0, 'Color', [0.6 0.6 0.6], 'LineStyle', '-', 'LineWidth', 0.5);
        xline(0, 'Color', [0.6 0.6 0.6], 'LineStyle', '-', 'LineWidth', 0.5);
        
        title(EEG_A.chanlocs(ch).labels, 'FontWeight', 'bold');
        xlim(p.timerange);
        
        % Save mini figure
        [folder, ~, ~] = fileparts(save_path);
        if ~exist(folder, 'dir')
            mkdir(folder);
        end
        filename = sprintf('%s/%s_%s.png', folder, title_text, EEG_A.chanlocs(ch).labels);
        exportgraphics(mini_fig, filename, 'Resolution', 300);
        close(mini_fig);
    end
end

% Create legend
legend_elements = [p1, p2];
legend_labels = {'target audio', 'standard audio'};

% Add confidence interval elements to legend if available
% if ~isempty(ci_A_lower) && ~isempty(ci_B_lower)
%     legend_elements = [p1, fill1, p2, fill2];
%     legend_labels = {'target audio', 'target CI (95%)', 'standard audio', 'standard CI (95%)'};
% end

lgd = legend(legend_elements, legend_labels, 'Position', [0.85 0.92 0.1 0.05]);
sgtitle(title_text, 'FontWeight', 'bold', 'FontSize', 12);

% Mini axis in bottom right (scale bar)
axes('Position', [0.88 0.05 0.1 0.1]);
hold on;

% Plot reference lines
plot(p.timerange, [0 0], 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
plot([0 0], [-10 15], 'Color', [0.5 0.5 0.5], 'LineWidth', 1);

% Add grid to scale bar
grid on;
set(gca, 'GridColor', p.gridcolor, 'GridAlpha', p.gridalpha);

% Set axis properties
xlim(p.timerange); 
ylim([-10 15]);
set(gca, 'XTick', [p.timerange(1) p.timerange(2)], 'YTick', [-10 15], 'FontSize', 6);
xlabel('Time (ms)', 'FontSize', 6);
ylabel('\muV', 'FontSize', 6);
axis on; 
box on;

% Save main figure
if nargin > 3 && ~isempty(save_path)
    exportgraphics(gcf, save_path, 'Resolution', 300);
    fprintf('圖形已保存至: %s\n', save_path);
end

end