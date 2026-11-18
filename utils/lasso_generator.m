function [A, b, x_star] = lasso_generator(m, d, Lmax, sparseness, seed)
% Generate least squares instance with specified condition number

rng(seed);

% Generate orthogonal matrices (economy size to avoid shape mismatch)
[U, ~] = qr(randn(m, m), 0);  % m x d
[V, ~] = qr(randn(d, d), 0);  % d x d

singular_vals = linspace(sqrt(Lmax), 0, min(d,m));
S = zeros(m, d);
S(1:min(d,m),1:min(d,m)) = diag(singular_vals);

A = U * S * V';

% Generate true solution
x_star = sprandn(d, 1, sparseness);
x_star = x_star / norm(x_star);

% Generate noisy observations
b = A * x_star + 0.1 * randn(m, 1);
end

