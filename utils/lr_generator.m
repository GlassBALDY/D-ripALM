function [X, y] = lr_generator(m, d, Lmax, seed)
% Generate logistic regression instance with specified condition number for X

rng(seed);

% Generate orthogonal matrices (economy size to avoid shape mismatch)
[U, ~] = qr(randn(m, d), 0);  % m x d
[V, ~] = qr(randn(d, d), 0);  % d x d

% Singular values spaced to achieve condition number kappa
singular_vals = linspace(sqrt(Lmax), 0, min(d, m));
S = zeros(m, d);
S(1:min(d, m), 1:min(d, m)) = diag(singular_vals);

X = U * S * V';

% Generate true solution
w_true = randn(d, 1);
w_true = w_true / norm(w_true);

% Generate noisy observations
logits = X * w_true + 0.1 * randn(m, 1);
y = sign(logits);
y(y == 0) = 1;
end

