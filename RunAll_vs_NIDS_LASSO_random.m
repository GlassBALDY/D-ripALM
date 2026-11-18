
clear; clear global; clc;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'solver')));
addpath(fullfile(fileparts(mfilename('fullpath')), 'utils'));

%% Configuration
run_DripALMre_FISTA = 1;
run_NIDS = 1;
run_PG_EXTRA = 1;
saveyes = 1;

%% Experiment parameters
% D-ripALM-re uses the default restart strategy.
floder_name = 'Benchmark_LASSO_random';

Lmaxs = [1e2];
lam_moduli = [1e-1, 1e-2, 1e-3];
n_agents = 20;
ds = [1000];
ms = [200];
ranseeds = 0:9;

% Network topologies
graphs = {'ring', 'erdos_renyi', 'geometric'};


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



%% Main experiment loop
for Lmax = Lmaxs
    for lam_modulus = lam_moduli
        for d = ds
            for m = ms
                %% Generate Lasso instance
                % global ATA ATb normATA lam_global KKTres
                for ranseed = ranseeds
                    rng(ranseed);
                    global KKTres_fun
                    [A, b] = lasso_generator(m, d, Lmax, 0.1, ranseed);
                    ATA = A' * A; ATb = A' * b; normA = sqrt(norm(ATA, 2));
                    [A_list, b_list] = agent_split(A, b, n_agents);
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
                    for graph_ = graphs
                        graph = graph_{1};
                        network = NetworkGraph(n_agents, graph, ranseed);
                        fprintf('Network spectral gap: %.4f\n', network.compute_spectral_gap());

                        eigs_W = sort(eig(network.W));
                        lambda_max_W = max(eigs_W);
                        lambda_min_W = min(eigs_W);
                        lambda_max_Z = 1 - lambda_min_W;
                        lambda_minp_Z = 1 - eigs_W(end-1);

                        %% save results
                        if saveyes == 1
                            dir_save = fullfile(fileparts(mfilename('fullpath')), 'results', floder_name, ...
                                ['lambda_modu', num2str(lam_modulus), '_Lmax', num2str(Lmax), '_graph_', graph]);
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
                                file_name = fullfile(dir_save, sprintf('%s_lam_modu%s_Lmax%s_d%d_n%d_samp%d_graph_%s_seed%d_rho%s_tau%s.mat', ...
                                    alg_name, num2str(lam_modulus), num2str(Lmax), d, n_agents, m, graph, ranseed, num2str(rho), num2str(tau0fista)));
                                save(file_name, 'iters', 'comms', 'feasp', 'feasd', 'time_cpu_total');
                            end

                        end

                        %% Run NIDS (Yan version) with proximal operator
                        if run_NIDS == 1
                            alg_name = 'NIDS';
                            
                            alpha_vec = 1 ./ L_f_local;
                            c_param = 0.5 / max(alpha_vec);

                            x0_mat = zeros(n_agents, d);

                            nids = NIDS();
                            nids.getGradS = @(X) reshape(cell2mat(cellfun(@(f,xi) f(xi(:))', local_grads, num2cell(X,2), 'UniformOutput', false)), n_agents, d);
                            nids.getProxR = @(Z, a) reshape(cell2mat(cellfun(@(prox,zi,ai) prox(zi(:), ai)', local_proxs, num2cell(Z,2), num2cell(a), 'UniformOutput', false)), n_agents, d);
                            nids.getS     = @(X) sum(cellfun(@(Ai,bi,xi) 0.5 * norm(Ai*xi(:) - bi)^2, A_list, b_list, num2cell(X,2)));
                            nids.getR     = @(X) lam_local * sum(abs(X), 'all');

                            paras = struct();
                            paras.tol    = stoptol;
                            paras.n      = n_agents;
                            paras.p      = d;
                            paras.x_star = zeros(n_agents, d);
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
                                file_name = fullfile(dir_save, sprintf('%s_lam_modu%s_Lmax%s_d%d_n%d_samp%d_graph_%s_seed%d.mat', ...
                                    alg_name, num2str(lam_modulus), num2str(Lmax), d, n_agents, m, graph, ranseed));
                                save(file_name, 'iters', 'comms', 'feasp', 'feasd', 'time_cpu_total');
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
                            result_pg_extra = pgextra.minimize(paras);
                            time_cpu_total = toc(t_start);

                            iters = max(0, numel(result_pg_extra.eng) - 2);
                            comms = iters;
                            feasp = result_pg_extra.err_con(end);
                            feasd = result_pg_extra.kkt_res(end);

                            fprintf('    Iter: %d, Comm: %d, Cons: %.3e, Sta: %.3e \n', iters, comms, feasp, feasd);
                            fprintf('(%s, %.2e) \n', num2str(iters), max(feasp,feasd));

                            %% save results
                            if saveyes == 1
                                file_name = fullfile(dir_save, sprintf('%s_lam_modu%s_Lmax%s_d%d_n%d_samp%d_graph_%s_seed%d.mat', ...
                                    alg_name, num2str(lam_modulus), num2str(Lmax), d, n_agents, m, graph, ranseed));
                                save(file_name, 'iters', 'comms', 'feasp', 'feasd', 'time_cpu_total');
                            end

                        end

                    end
                end
            end
        end
    end
end

