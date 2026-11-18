
clear; clear global; clc;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'solver')));
addpath(fullfile(fileparts(mfilename('fullpath')), 'utils'));

%% Configuration
run_DripALM_FISTA = 0;
run_DripALMre_FISTA = 1;
run_IDEAL = 1;
saveyes = 1;


%% Experiment parameters
% D-ripALM-re uses the default restart strategy.
floder_name = 'DiffTolPara_log_reg_random';

Lmaxs = [1e3];  % Global data scale: norm(X, 2)^2 = Lmax.
lambdas = [1e-2];
n_agents = 10;
d = 1000;
max_samples = 400;
ranseeds = 0:9;

% Network topologies
graphs = {'erdos_renyi'};


%% Settings: (p)ALMs options
maxiter = 30000;
maxiterFOMs = 30000;
tau0fista = 1e-3;
rhos = [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01]; % D-ripALM tuning parameters
c_values = [1e-2, 1e-1, 1e0, 1e1];  % IDEAL tuning parameters: eps0
abs_factors = [0.2, 0.4, 0.6, 0.8];  % IDEAL tuning parameters: alpha0
stoptol = 1e-6;
options.stoptol = stoptol;
options.maxiter = maxiter;
options.maxapgiter = maxiterFOMs;
options.maxtotaliter = maxiterFOMs;
options.sigfun = @(iter) min(1.5^(iter-1), 1e4);

%% Main experiment loop
for graph_ = graphs
    graph = graph_{1};
    for Lmax = Lmaxs
        for lam = lambdas
            for ranseed = ranseeds
                rng(ranseed);
                %% Generate data
                [X, y] = lr_generator(max_samples, d, Lmax, ranseed);
                nt = 10;
                [X_test, y_test] = lr_generator(nt, d, Lmax, ranseed+1);
                [X_list, y_list, data_ix] = agent_split(X, y, n_agents);

                % Estimate L_f for each local function (node)
                local_objs = cell(n_agents, 1);
                local_grads = cell(n_agents, 1);
                L_f_local = zeros(n_agents, 1);
                for i = 1:n_agents
                    Xi = X_list{i};
                    yi = y_list{i};
                    local_objs{i} = @(w) logistic_obj(w, Xi, yi, lam);
                    local_grads{i} = @(w) logistic_grad(w, Xi, yi, lam);
                    L_f_local(i) = 0.25 * svds(Xi, 1)^2 + lam;
                end
                L_f = max(L_f_local);  % Global L_f is the maximum among all nodes
                fprintf('Local L_f values: %s\n', mat2str(L_f_local, 4));
                fprintf('Global L_f (max): %.6f\n', L_f);

                % Define objective and gradient functions (for IDEAL and MIDEAL)
                obj_fn = @(w, ix) logistic_obj(w, X(ix,:), y(ix), lam);
                % Reuse cached full-agent gradients through the existing IDEAL interface.
                sample_agent = zeros(size(X, 1), 1);
                for p_ix = 1:n_agents
                    sample_agent(data_ix{p_ix}) = p_ix;
                end
                grad_fn = @(w, ix) local_grads{sample_agent(ix(1))}(w);

                %% Generate Network
                network = NetworkGraph(n_agents, graph, ranseed);
                fprintf('Network spectral gap: %.4f\n', network.compute_spectral_gap());

                eigs_W = sort(eig(network.W));
                Z = eye(n_agents) - network.W;
                lambda_max_W = max(eigs_W);
                lambda_min_W = min(eigs_W);
                lambda_max_Z = 1 - lambda_min_W;
                lambda_minp_Z = 1 - eigs_W(end-1);

                %% save results
                if saveyes == 1
                    dir_save = fullfile(fileparts(mfilename('fullpath')), 'results', floder_name, ...
                        ['lambda', num2str(lam), '_Lmax', num2str(Lmax)]);
                    if exist(dir_save, 'dir') ~= 7, mkdir(dir_save); end
                    dir_save = [dir_save, filesep];
                end

                %% Run D-ripALM with different rho
                if run_DripALM_FISTA == 1
                    alg_name = 'DripALM_FISTA';
                    res_DripALM_FISTA = [];
                    for i = 1:length(rhos)
                        rho = rhos(i);
                        options_DripALM_FISTA = options;
                        options_DripALM_FISTA.taufun = @(iter) tau0fista;
                        options_DripALM_FISTA.rhofun = @(iter) rho;
                        options_DripALM_FISTA.inexacttype = 2;
                        options_DripALM_FISTA.extrapo = 1;
                        options_DripALM_FISTA.theta_choice = 1;
                        options_DripALM_FISTA.restart_w_strategy = 'none';

                        dripalm_fista = DripALM(n_agents, d, network.W, ...
                            'local_objectives', local_objs, ...
                            'local_gradients', local_grads, ...
                            'L_f', L_f, 'sc_f', lam);

                        result_fista = dripalm_fista.main_iters(options_DripALM_FISTA);
                        iters = result_fista.iterations;
                        comms = result_fista.comm_rounds;
                        feasp = result_fista.consensus_violation;
                        feasd = result_fista.stational_violation;
                        time_cpu_total = result_fista.wall_time;

                        fprintf('    Iter: %d, Comm: %d, Cons: %.3e, Sta: %.3e \n', iters, comms, feasp, feasd);

                        % store results
                        res_DripALM_FISTA = [res_DripALM_FISTA; rho, iters, comms, feasp, feasd, time_cpu_total];

                        %% save results
                        if saveyes == 1
                            file_name = sprintf('%s%s_lam%s_Lmax%s_d%d_n%d_samp%d_graph_%s_seed%d_rho%s_tau%s.mat', ...
                                dir_save, alg_name, num2str(lam), num2str(Lmax), d, n_agents, max_samples, graph, ranseed, num2str(rho), num2str(tau0fista));
                            save(file_name, 'iters', 'comms', 'feasp', 'feasd', 'time_cpu_total');
                        end
                    end
                end

                %% Run DripALMre_FISTA
                if run_DripALMre_FISTA == 1
                    alg_name = 'DripALMre_FISTA';
                    res_DripALMre_FISTA = [];
                    for i = 1:length(rhos)
                        rho = rhos(i);
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
                            'L_f', L_f, 'sc_f', lam);

                        result_fista = dripalm_re_fista.main_iters(options_DripALMre_FISTA);
                        iters = result_fista.iterations;
                        comms = result_fista.comm_rounds;
                        feasp = result_fista.consensus_violation;
                        feasd = result_fista.stational_violation;
                        time_cpu_total = result_fista.wall_time;

                        fprintf('    Iter: %d, Comm: %d, Cons: %.3e, Sta: %.3e \n', iters, comms, feasp, feasd);

                        % store results
                        res_DripALMre_FISTA = [res_DripALMre_FISTA; rho, iters, comms, feasp, feasd, time_cpu_total];

                        %% save results
                        if saveyes == 1
                            file_name = sprintf('%s%s_lam%s_Lmax%s_d%d_n%d_samp%d_graph_%s_seed%d_rho%s_tau%s.mat', ...
                                dir_save, alg_name, num2str(lam), num2str(Lmax), d, n_agents, max_samples, graph, ranseed, num2str(rho), num2str(tau0fista));
                            save(file_name, 'iters', 'comms', 'feasp', 'feasd', 'time_cpu_total');
                        end
                    end
                end

                %% Run IDEAL with different abstol
                if run_IDEAL == 1
                    x0 = zeros(d, n_agents);  % Random initialization for each agent
                    alg_name = 'IDEAL';
                    res_IDEAL = [];
                    for ii = 1:length(abs_factors)
                        for jj = 1:length(c_values)
                            abs_fac = abs_factors(ii);
                            c_val = c_values(jj);
                            sigma0 = L_f/lambda_max_Z;
                            eta = 1./(lambda_max_Z*sigma0+L_f);
                            eta2 = sigma0 + (lam/lambda_max_Z);
                            kap_out= (L_f+sigma0*lambda_minp_Z)./(lam+sigma0*lambda_max_Z)* lambda_max_Z/lambda_minp_Z;
                            kap_in = ( L_f+sigma0*lambda_max_Z)./lam;
                            beta_in = (sqrt(kap_in)-1)./(sqrt(kap_in)+1);
                            beta_out = (sqrt(kap_out)-1)./(sqrt(kap_out)+1);
                        
                            options_IDEAL = options;
                            options_IDEAL.sigfun = @(iter) sigma0;
                            options_IDEAL.inexacttype = 1;
                            options_IDEAL.tolfun = @(iter) [c_val * abs_fac^iter, c_val * abs_fac^iter];


                            T_agd = []; n_agd = [];
                            data_full = (1:max_samples);
                            [iter_AGD,err_AGD,AGD_end,test_acc_AGD, result_ideal] = AccGDSolver(obj_fn,grad_fn,n_agents,n_agd,x0,Z,T_agd,L_f,lam, ...
                                eta,eta2,beta_in,beta_out,...
                                lambda_minp_Z,lambda_max_Z,sigma0, ...
                                data_ix,zeros(d, 1),0,data_full,X_test,y_test,nt, options_IDEAL);

                            iters = result_ideal.iterations;
                            comms = result_ideal.comm_rounds;
                            feasp = result_ideal.consensus_violation;
                            feasd = result_ideal.stational_violation;  % IDEAL may not have stationarity violation
                            time_cpu_total = result_ideal.wall_time;

                            fprintf('    Iter: %d, Comm: %d, Cons: %.3e, Sta: %.3e \n', iters, comms, feasp, feasd);

                            % store results
                            res_IDEAL = [res_IDEAL; abs_fac, c_val, iters, comms, feasp, feasd, time_cpu_total];

                            %% save results
                            if saveyes == 1
                                file_name = sprintf('%s%s_lam%s_Lmax%s_d%d_n%d_samp%d_graph_%s_seed%d_cval%s_abs_fac%s.mat', ...
                                    dir_save, alg_name, num2str(lam), num2str(Lmax), d, n_agents, max_samples, graph, ranseed, num2str(c_val), num2str(abs_fac));
                                save(file_name, 'iters', 'comms', 'feasp', 'feasd', 'time_cpu_total');
                            end
                        end
                    end
                end

            end
        end
    end
end

%% Helper functions
function val = logistic_obj(w, X, y, lam)
z = y .* (X * w);
val = sum(log1p(exp(-z))) + 0.5 * lam * norm(w)^2;
end

function g = logistic_grad(w, X, y, lam)
z = y .* (X * w);
coeff = -y .* (1 - sigmoid(z));
g = (X' * coeff) + lam * w;
end

function s = sigmoid(z)
s = 1 ./ (1 + exp(-z));
end
