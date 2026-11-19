
clear; clc;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'solver')));
addpath(fullfile(fileparts(mfilename('fullpath')), 'utils'));

%% Configuration
run_DripALMre_FISTA = 1;
run_NIDS = 1;
run_PG_EXTRA = 1;
saveyes = 1;

%% Experiment parameters
floder_name = 'Benchmark_LASSO_LIBSVM_shuffle_v1_test';

% Bundled LIBSVM datasets (original scale).
dataset_name = 'abalone_scale';
% Require n_agents <= m and nonzero local Ai.
lam_modulus = 1e-1;
n_agents = 20;
ranseed = 2025;
if ~isnumeric(ranseed) || ~isscalar(ranseed) || ~isreal(ranseed) || ...
        ~isfinite(ranseed) || ranseed < 0 || ranseed > 2^32-1 || ranseed ~= fix(ranseed)
    error('realdata:InvalidSeeds', 'ranseed must be an integer scalar in [0, 2^32-1].');
end
ranseed = double(ranseed);
rng(ranseed);
data_protocol = 'shuffle-before-split-v1';
shuffle_generator = 'mt19937ar';

% Network topologies
graph = 'erdos_renyi'; % {'ring', 'erdos_renyi', 'geometric', 'grid'}

%% Settings: ripALM options
maxiter = 30000;
maxiterFOMs = 30000;
tau0fista = 1e-3;
rho = 0.99;
stoptol = 1e-6;
options.stoptol = stoptol;
options.maxiter = maxiter;
options.maxapgiter = maxiterFOMs;
options.maxtotaliter = maxiterFOMs;
options.sigfun = @(iter) min(1.5^(iter-1), 1e4);

%% Load and split LIBSVM data (original scale)
global KKTres_fun
load(fullfile(fileparts(mfilename('fullpath')), 'data', 'LIBSVM', [dataset_name, '.mat']), 'A', 'b');
[m, d] = size(A);
data_stream = RandStream(shuffle_generator, 'Seed', ranseed);
sample_order = randperm(data_stream, m);
A = A(sample_order, :);
b = b(sample_order, :);
ATA = A' * A; ATb = A' * b;

% Shuffle paired rows before splitting; the last agent receives the remainder.
[A_list, b_list] = agent_split(A, b, n_agents);
sample_counts = cellfun(@(Ai) size(Ai, 1), A_list).';
lam_global = lam_modulus*norm(ATb, 'inf');
lam_local = lam_global / n_agents;

% Create local functions
local_objs = cell(n_agents, 1);
local_grads = cell(n_agents, 1);
local_proxs = cell(n_agents, 1);
L_f_local = zeros(n_agents, 1);

for i = 1:n_agents
    Ai = A_list{i};
    bi = b_list{i};
    local_objs{i} = @(x) 0.5 * norm(Ai * x - bi)^2 + lam_local * sum(abs(x));
    local_grads{i} = @(x) Ai' * (Ai * x - bi);
    local_proxs{i} = @(x, t) prox_l1_GPU(x, lam_local * t);
    L_f_local(i) = max(eig(Ai' * Ai));
end
KKTres_fun = @(x) norm( x - prox_l1_GPU(x - ATA * x + ATb, lam_global)) / (1 + norm(A*x-b)+norm(x));

L_f = max(L_f_local);
fprintf('Local L_f values: %s \n', mat2str(L_f_local, 4));
fprintf('Global L_f (max): %.6f \n', L_f);

%% Generate Network
network = NetworkGraph(n_agents, graph, ranseed);
fprintf('Network spectral gap: %.4f\n', network.compute_spectral_gap());

eigs_W = sort(eig(network.W));
lambda_max_W = max(eigs_W);
lambda_min_W = min(eigs_W);
lambda_max_Z = 1 - lambda_min_W;
lambda_minp_Z = 1 - eigs_W(end-1);

%% save results
if saveyes == 1
    dir_save = fullfile(fileparts(mfilename('fullpath')), 'results', floder_name, dataset_name, ...
        ['lambda_modu', num2str(lam_modulus), '_graph_', graph]);
    if exist(dir_save, 'dir') ~= 7, mkdir(dir_save); end
end

%% Run D-ripALM-re (FISTA) with proximal operator
if run_DripALMre_FISTA == 1
    alg_name = 'DripALMre_FISTA';

    options_DripALMre_FISTA = options;
    options_DripALMre_FISTA.taufun = @(iter) tau0fista;
    options_DripALMre_FISTA.rhofun = @(iter) rho;
    options_DripALMre_FISTA.inexacttype = 2;
    options_DripALMre_FISTA.extrapo = 1;
    options_DripALMre_FISTA.theta_choice = 1;
    options_DripALMre_FISTA.restart_w_strategy = 'default';

    dripalm_re_fista = DripALM(n_agents, d, network.W, ...
        'local_objectives', local_objs, ...
        'local_gradients', local_grads, ...
        'local_proximals', local_proxs, ...
        'L_f', L_f, 'sc_f', 0);

    result_dripalm_re_fista = dripalm_re_fista.main_iters(options_DripALMre_FISTA);

    iters = result_dripalm_re_fista.iterations;
    comms = result_dripalm_re_fista.comm_rounds;
    feasp = result_dripalm_re_fista.consensus_violation;
    feasd = result_dripalm_re_fista.stational_violation;
    time_cpu_total = result_dripalm_re_fista.wall_time;

    fprintf('    Iter: %d, Comm: %d, Cons: %.3e, Sta: %.3e \n', iters, comms, feasp, feasd);
    fprintf('(%s, %.2e) \n', num2str(iters), max(feasp,feasd));

    %% save results
    if saveyes == 1
        file_name = fullfile(dir_save, sprintf('%s_lam_modu%s_data_%s_d%d_n%d_samp%d_graph_%s_seed%d_rho%s_tau%s.mat', ...
            alg_name, num2str(lam_modulus), dataset_name, d, n_agents, m, graph, ranseed, num2str(rho), num2str(tau0fista)));
        save(file_name, 'iters', 'comms', 'feasp', 'feasd', 'time_cpu_total', 'dataset_name', 'sample_counts', 'ranseed', 'data_protocol', 'shuffle_generator');
    end
    
end

%% Run NIDS (Yan version) with proximal operator
if run_NIDS == 1
    alg_name = 'NIDS';

    alpha_vec = 1 ./ L_f_local;
    c_param = 0.5 / max(alpha_vec);

    x0_mat = zeros(n_agents, d);

    nids = NIDS();
    nids.getGradS = @(X) cell2mat(arrayfun(@(row) local_grads{row}(X(row,:)'), 1:n_agents, 'UniformOutput', false))';
    nids.getProxR = @(X, a) cell2mat(arrayfun(@(row) local_proxs{row}(X(row,:)', a(row)), 1:n_agents, 'UniformOutput', false))';
    nids.getS     = @(X) sum(cellfun(@(Ai,bi,xi) 0.5 * norm(Ai*xi(:) - bi)^2, A_list, b_list, num2cell(X,2)));
    nids.getR     = @(X) lam_local * sum(abs(X), 'all');

    paras = struct();
    paras.tol    = stoptol;
    paras.n      = n_agents;
    paras.p      = d;
    paras.x_star = zeros(n_agents, d);  % use zeros as a placeholder
    paras.iter   = maxiterFOMs;
    paras.x0     = x0_mat;
    paras.alpha  = alpha_vec;
    paras.method = 'NIDS';
    paras.W      = network.W;
    paras.c      = c_param;
    paras.verbose= true;

    t_start = tic;
    result_nids = nids.minimize(paras);
    time_cpu_total = toc(t_start);

    iters = max(0, numel(result_nids.eng) - 2);
    comms = iters;
    feasp = result_nids.err_con(end);
    feasd = result_nids.kkt_res(end);

    fprintf('    Iter: %d, Comm: %d, Cons: %.3e, Sta: %.3e \n', iters, comms, feasp, feasd);
    fprintf('(%s, %.2e) \n', num2str(iters), max(feasp,feasd));

    %% save results
    if saveyes == 1
        file_name = fullfile(dir_save, sprintf('%s_lam_modu%s_data_%s_d%d_n%d_samp%d_graph_%s_seed%d.mat', ...
            alg_name, num2str(lam_modulus), dataset_name, d, n_agents, m, graph, ranseed));
        save(file_name, 'iters', 'comms', 'feasp', 'feasd', 'time_cpu_total', 'dataset_name', 'sample_counts', 'ranseed', 'data_protocol', 'shuffle_generator');
    end

end

%% Run PG-EXTRA (Yan version) with proximal operator
if run_PG_EXTRA == 1
    alg_name = 'PG_EXTRA';

    alpha_vec = 0.99 * (1 + lambda_min_W) ./ L_f * ones(n_agents, 1);
    
    x0_mat = zeros(n_agents, d);

    pgextra = NIDS();
    pgextra.getGradS = @(X) reshape(cell2mat(cellfun(@(f,xi) f(xi(:))', local_grads, num2cell(X,2), 'UniformOutput', false)), n_agents, d);
    pgextra.getProxR = @(Z, a) reshape(cell2mat(cellfun(@(prox,zi,ai) prox(zi(:), ai)', local_proxs, num2cell(Z,2), num2cell(a), 'UniformOutput', false)), n_agents, d);
    pgextra.getS     = @(X) sum(cellfun(@(Ai,bi,xi) 0.5 * norm(Ai*xi(:) - bi)^2, A_list, b_list, num2cell(X,2)));
    pgextra.getR     = @(X) lam_local * sum(abs(X), 'all');

    paras = struct();
    paras.tol    = stoptol;
    paras.n      = n_agents;
    paras.p      = d;
    paras.x_star = zeros(n_agents, d);
    paras.iter   = maxiterFOMs;
    paras.x0     = x0_mat;
    paras.alpha  = alpha_vec;
    paras.method = 'PG-EXTRA';
    paras.W      = network.W;
    paras.verbose= true;

    t_start = tic;
    result_pgextra = pgextra.minimize(paras);
    time_cpu_total = toc(t_start);

    iters = max(0, numel(result_pgextra.eng) - 2);
    comms = iters;
    feasp = result_pgextra.err_con(end);
    feasd = result_pgextra.kkt_res(end);

    fprintf('    Iter: %d, Comm: %d, Cons: %.3e, Sta: %.3e \n', iters, comms, feasp, feasd);
    fprintf('(%s, %.2e) \n', num2str(iters), max(feasp,feasd));

    %% save results
    if saveyes == 1
        file_name = fullfile(dir_save, sprintf('%s_lam_modu%s_data_%s_d%d_n%d_samp%d_graph_%s_seed%d.mat', ...
            alg_name, num2str(lam_modulus), dataset_name, d, n_agents, m, graph, ranseed));
        save(file_name, 'iters', 'comms', 'feasp', 'feasd', 'time_cpu_total', 'dataset_name', 'sample_counts', 'ranseed', 'data_protocol', 'shuffle_generator');
    end

end

fprintf('Dataset: %s; (m, d, n, Lam) = (%s, %s, %s, %s) \n', ...
    dataset_name, num2str(m), num2str(d), num2str(n_agents), num2str(lam_global));
fprintf('Samples per agent: %s\n', mat2str(sample_counts));

if run_DripALMre_FISTA, fprintf('DripALMre_FISTA: (%s (%s), %.2e) \n', ...
        num2str(result_dripalm_re_fista.iterations), num2str(result_dripalm_re_fista.comm_rounds), max(result_dripalm_re_fista.consensus_violation,result_dripalm_re_fista.stational_violation)); end
if run_PG_EXTRA, fprintf('PG-EXTRA: (%s, %.2e) \n', ...
        num2str(max(0, numel(result_pgextra.eng) - 2)), max(result_pgextra.err_con(end),result_pgextra.kkt_res(end))); end
if run_NIDS, fprintf('NIDS: (%s, %.2e) \n', ...
        num2str(max(0, numel(result_nids.eng) - 2)), max(result_nids.err_con(end),result_nids.kkt_res(end))); end

