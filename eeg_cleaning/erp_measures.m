function erp_measures(V, bids, outdir)
% Compute the ERPs of the Move segments for each cleaned version, and their measures.
%   erp_measures(V, bids, outdir)
%   V       struct array, one per version: label and folder (outroot/<version>/k<k> of clean_paper_versions)
%   bids    BIDS dataset root, for the events
% Epochs -200 to 800 ms, baseline -200 to 0 ms, averaged per subject and condition. Writes to outdir:
%   erp_averages.mat   subject averages at Fz, Cz and Pz
%   erp_measures.csv   N1, P3, target - standard and residual noise per version and condition
%   erp_latency.csv    peak of target - standard per version, condition and channel
%   erp_windows.csv    Pz target peak and mean |target - standard| at -100 to 100 and 600 to 800 ms
if ~exist(outdir, 'dir'), mkdir(outdir); end
fs = 500; k = -100:400; t = k / fs * 1000; base = t < 0; win = t >= 300 & t < 500;
chans = {'Fz', 'Cz', 'Pz'};
moves = {'walkstraight', 'walkcurved', 'headshake', 'headnod', 'pickup', 'stairs'};
conds = [{'all movement'}, moves, {'slow', 'fast'}];
G = cell(1, numel(V));
for v = 1:numel(V)
    G{v} = load_version(V(v).folder, bids, chans, k, base, moves, conds);
    fprintf('%s: %d Move files\n', V(v).label, G{v}.nfiles);
end
V = struct('label', {V.label});
save(fullfile(outdir, 'erp_averages.mat'), 'G', 'V', 't', 'chans', 'conds', '-v7.3');

rows = {};
for v = 1:numel(V)
    for c = 1:numel(conds)
        A = G{v}.(fld(conds{c}));
        d = mean(A.target(:, 3, win), 3) - mean(A.standard(:, 3, win), 3);
        rows(end + 1, :) = {V(v).label, conds{c}, A.n, round(mean(A.standard(:, 2, t >= 80 & t <= 120), 'all'), 2), ...
            round(mean(A.target(:, 3, win), 'all'), 2), round(mean(d), 2), round(std(d) / sqrt(numel(d)), 2), ...
            round(median(std(squeeze(A.standard(:, 3, base)), 0, 2)), 2), sum(d > 0)}; %#ok<AGROW>
    end
end
M = cell2table(rows, 'VariableNames', {'version', 'condition', 'subjects', 'N1_Cz_80_120_uV', ...
    'P3_Pz_target_300_500_uV', 'Pz_target_minus_standard_uV', 'Pz_target_minus_standard_SEM_uV', ...
    'residual_noise_Pz_uV', 'subjects_target_minus_standard_positive'});
writetable(M, fullfile(outdir, 'erp_measures.csv'));

lw = t >= 200 & t <= 600; tl = t(lw); rows = {};
for v = 1:numel(V)
    for c = 1:numel(conds)
        A = G{v}.(fld(conds{c})); D = A.target - A.standard;
        for ch = 1:numel(chans)
            g = reshape(mean(D(:, ch, lw), 1), 1, []); [pk, i] = max(g);
            ps = arrayfun(@(s) tl(find(reshape(D(s, ch, lw), 1, []) == max(D(s, ch, lw)), 1)), 1:size(D, 1));
            rows(end + 1, :) = {V(v).label, conds{c}, chans{ch}, tl(i), round(pk, 2), median(ps)}; %#ok<AGROW>
        end
    end
end
writetable(cell2table(rows, 'VariableNames', {'version', 'condition', 'channel', 'grand_average_peak_ms', ...
    'grand_average_peak_uV', 'median_subject_peak_ms'}), fullfile(outdir, 'erp_latency.csv'));

pw = t >= 250 & t <= 450; tw = t(pw); w1 = t >= -100 & t <= 100; w2 = t >= 600 & t <= 800; rows = {};
for v = 1:numel(V)
    A = G{v}.(fld('all movement'));
    m = reshape(mean(A.target(:, 3, :), 1), 1, []); [pk, i] = max(m(pw));
    dm = reshape(mean(A.target(:, 3, :) - A.standard(:, 3, :), 1), 1, []);
    rows(end + 1, :) = {V(v).label, round(pk, 2), tw(i), round(mean(abs(dm(w1))), 2), round(mean(abs(dm(w2))), 2)}; %#ok<AGROW>
end
writetable(cell2table(rows, 'VariableNames', {'version', 'Pz_target_peak_250_450_uV', 'Pz_target_peak_ms', ...
    'Pz_abs_target_minus_standard_m100_100_uV', 'Pz_abs_target_minus_standard_600_800_uV'}), ...
    fullfile(outdir, 'erp_windows.csv'));
end

function R = load_version(folder, bids, chans, k, base, moves, conds)
% per condition: subject means of the standard and target trials (subjects x channels x time)
fs = 500;
F = dir(fullfile(folder, 'sub-*', '*move.mat'));
E = struct('sub', {}, 'move', {}, 'speed', {}, 'S', {}, 'nS', {}, 'T', {}, 'nT', {});
for i = 1:numel(F)
    d = load(fullfile(F(i).folder, F(i).name), 'data', 'labels');
    lab = cellstr(d.labels);
    x = double(d.data(cellfun(@(c) find(strcmp(lab, c)), chans), :));
    stem = F(i).name(1:end - 4); sub = stem(1:6);
    ev = readtable(fullfile(bids, sub, 'eeg', [stem '_events.tsv']), 'FileType', 'text', 'Delimiter', '\t', ...
                   'TextType', 'char');
    on = round(ev.onset * fs) + 1; bnd = on(strcmp(ev.trial_type, 'boundary'));
    sm = cell(1, 2); n = zeros(1, 2); tones = {'standard', 'target'};
    for j = 1:2
        c = on(strcmp(ev.trial_type, tones{j}));
        ok = c + k(1) >= 1 & c + k(end) <= size(x, 2) & arrayfun(@(q) ~any(bnd > q + k(1) & bnd <= q + k(end)), c);
        c = c(ok); s = zeros(numel(chans), numel(k));
        for q = c(:)', e = x(:, q + k); s = s + (e - mean(e(:, base), 2)); end      % baseline, then sum
        sm{j} = s; n(j) = numel(c);
    end
    mv = regexp(stem, 'task-([a-z]+)_acq-([a-z]+)move', 'tokens', 'once');
    E(end + 1) = struct('sub', sub, 'move', mv{1}, 'speed', mv{2}, 'S', sm{1}, 'nS', n(1), 'T', sm{2}, 'nT', n(2)); %#ok<AGROW>
end
R.nfiles = numel(F);
for ci = 1:numel(conds)
    nm = conds{ci};
    if strcmp(nm, 'all movement'), sel = true(1, numel(E));
    elseif any(strcmp(moves, nm)), sel = strcmp({E.move}, nm);
    else, sel = strcmp({E.speed}, nm); end
    subs = unique({E(sel).sub});
    St = zeros(0, numel(chans), numel(k)); Tt = St;
    for s = 1:numel(subs)
        es = E(sel & strcmp({E.sub}, subs{s}));
        nS = sum([es.nS]); nT = sum([es.nT]);
        if nS == 0 || nT == 0, continue, end
        St(end + 1, :, :) = reshape(sum(cat(3, es.S), 3) / nS, [1 numel(chans) numel(k)]);   % mean over all trials
        Tt(end + 1, :, :) = reshape(sum(cat(3, es.T), 3) / nT, [1 numel(chans) numel(k)]);
    end
    R.(fld(nm)) = struct('standard', St, 'target', Tt, 'n', size(St, 1));
end
end

function f = fld(nm)
% condition name as a field name ('all movement' -> 'allMovement')
f = matlab.lang.makeValidName(nm);
end
