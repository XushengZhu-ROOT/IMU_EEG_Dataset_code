function y = declip_gyro(x)
% Restore the clipped peaks of one gyroscope axis of the head IMU (rabboni: the values stop at +992.2 and -1000).
%   y = declip_gyro(x)      x = one gyroscope axis at the IMU samples (column; NaN = no data)
% |x| >= 990 counts as clipped. A run = consecutive clipped samples of the same sign. Up to 5 unclipped, non-NaN
% samples right before and right after the run are taken; a run with at least 3 on each side is replaced by a cubic
% spline (MATLAB spline, not-a-knot) through those samples, and other runs keep their values. The restored value is
% at least as far from zero as the clipped value (the true value was at least that large).
level = 990; K = 5;
x = x(:); y = x; n = numel(x);
c = abs(x) >= level;                                   % NaN gives false
ok = ~c & ~isnan(x);                                   % samples the spline may use
i = 1;
while i <= n
    if ~c(i), i = i + 1; continue, end
    s = sign(x(i)); j = i;
    while j < n && c(j + 1) && sign(x(j + 1)) == s, j = j + 1; end
    L = []; q = i - 1;
    while q >= 1 && numel(L) < K && ok(q), L(end + 1) = q; q = q - 1; end %#ok<AGROW>
    R = []; q = j + 1;
    while q <= n && numel(R) < K && ok(q), R(end + 1) = q; q = q + 1; end %#ok<AGROW>
    if numel(L) >= 3 && numel(R) >= 3
        tk = [fliplr(L) R]'; tr = (i:j)';
        f = spline(tk, x(tk), tr);
        y(tr) = s * max(s * f, s * x(tr));
    end
    i = j + 1;
end
end
