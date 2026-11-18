classdef DripALM < handle

    properties
        % Problem dimensions
        n_agents        % Number of agents
        d               % Dimension of local decision variable
        N               % Total dimension (n_agents × d)

        % Local objective and first-order oracles
        f_local         % Cell array of local objective functions f_i(x_i)
        grad_f_local    % Cell array of gradient functions ∇f_i(x_i)
        prox_f_local    % Cell array of proximal operators prox_{t f_i}(·, t)
        L_f             % Lipschitz constant of ∇F (optional)
        sc_f

        % Network structure
        W               % Mixing matrix W (n × n)
        Z               % Constraint matrix Z = (I_n - W) ⊗ I_d
        neighbors       % Cell array of neighbor lists for each agent
        weights         % Cell array of edge weights for each agent
        lambda_max_Z    % Maximum eigenvalue of Z
        lambda_minp_Z


        % Algorithm variables (Algorithm 1, line 322)
        x               % Stacked primal variable x ∈ R^{nd}
        Omega           % Transformed dual variable Ω ∈ R^{nd}
        w               % Auxiliary variable w ∈ R^{nd}
        comm_rounds     % Accumulated communication rounds

        % History tracking
        history         % Struct tracking convergence metrics

        % Iteration counter
        iteration       % Current iteration number

    end

    methods
        function obj = DripALM(n_agents, d, mixing_matrix, varargin)

            % Parse optional arguments
            p = inputParser;
            addParameter(p, 'local_objectives', []);
            addParameter(p, 'local_gradients', []);
            addParameter(p, 'local_proximals', []);
            addParameter(p, 'L_f', 1.0);
            addParameter(p, 'sc_f', 0);
            parse(p, varargin{:});

            % Problem dimensions
            obj.n_agents = n_agents;
            obj.d = d;
            obj.N = n_agents * d;

            % Local objectives and first-order oracles
            obj.f_local = p.Results.local_objectives;

            % Handle gradients
            if isempty(p.Results.local_gradients)
                error('Provide local_gradients or local_gradients for first-order information.');
            end

            obj.grad_f_local = p.Results.local_gradients;
            obj.L_f = p.Results.L_f;
            obj.sc_f = p.Results.sc_f;

            % Proximal operators (identity if not provided)
            if ~isempty(p.Results.local_proximals)
                obj.prox_f_local = p.Results.local_proximals;
            else
                % Default: identity operator
                obj.prox_f_local = cell(n_agents, 1);
                for i = 1:n_agents
                    obj.prox_f_local{i} = @(x, t) x;
                end
            end

            % Network structure
            obj.W = mixing_matrix;
            obj.Z = obj.create_constraint_matrix();

            % Build neighbor lists for sparse operations
            obj.neighbors = cell(n_agents, 1);
            obj.weights = cell(n_agents, 1);
            for i = 1:n_agents
                idx = find(mixing_matrix(i, :));
                obj.neighbors{i} = idx;
                obj.weights{i} = mixing_matrix(i, idx);
            end

            % Spectral information
            eigs_W = eig(obj.W);
            lambda_max_W = max(eigs_W);
            lambda_min_W = min(eigs_W);
            obj.lambda_max_Z = 1 - lambda_min_W;
            obj.lambda_minp_Z = 1 - eigs_W(end-1);

            % Initialize variables
            obj.x = zeros(obj.N, 1);
            obj.Omega = zeros(obj.N, 1);
            obj.comm_rounds = 0.0;

            % Initialize history
            obj.history = struct(...
                'objective', [], ...
                'consensus_violation', [], ...
                'stational_violation', [], ...
                'dual_residual', [], ...
                'relative_error', [], ...
                'criterion_satisfied', [], ...
                'sigma', [], ...
                'tau', [], ...
                'comm_rounds' ,[]);

            % Iteration counter
            obj.iteration = 0;

        end

        function Z = create_constraint_matrix(obj)

            I_n = speye(obj.n_agents);
            I_d = speye(obj.d);
            Z = kron(I_n - obj.W, I_d);
        end

        function obj_val = compute_global_objective(obj, x)

            obj_val = 0.0;
            for i = 1:obj.n_agents
                idx = (i-1)*obj.d + (1:obj.d);
                x_i = x(idx);
                obj_val = obj_val + obj.f_local{i}(x_i);
            end
        end

        function grad = compute_global_gradient(obj, x)

            grad = zeros(obj.N, 1);
            for i = 1:obj.n_agents
                idx = (i-1)*obj.d + (1:obj.d);
                x_i = x(idx);
                grad(idx) = obj.grad_f_local{i}(x_i);
            end
        end

        function x_out = apply_prox(obj, x, alpha)

            x_out = x;
            for i = 1:obj.n_agents
                idx = (i-1)*obj.d + (1:obj.d);
                xi = x(idx);
                prox_i = obj.prox_f_local{i};
                x_out(idx) = prox_i(xi, alpha);
            end
        end

        function [step_size, lipschitz_subproblem] = compute_subproblem_step_size(obj, sigma, tau)
            % Compute a safe first-order step size for the inner subproblem

            if isempty(obj.L_f)
                smooth_part = 0.0;
            else
                smooth_part = obj.L_f;
            end

            lipschitz_subproblem = smooth_part + sigma * obj.lambda_max_Z + tau / max(sigma, 1e-12);

            % Guard against pathological zero denominators
            lipschitz_subproblem = max(lipschitz_subproblem, 1e-12);
            step_size = 1 / lipschitz_subproblem;
        end

        function out = Z_mul(obj, x, count_comm)

            out = zeros(size(x));
            for i = 1:obj.n_agents
                idx = (i-1)*obj.d + (1:obj.d);
                xi = x(idx);
                weighted = zeros(obj.d, 1);
                neigh_idx = obj.neighbors{i};
                neigh_w = obj.weights{i};
                for k = 1:length(neigh_idx)
                    j = neigh_idx(k);
                    w_ij = neigh_w(k);
                    j_idx = (j-1)*obj.d + (1:obj.d);
                    weighted = weighted + w_ij * x(j_idx);
                end
                out(idx) = xi - weighted;
            end

            % Count communication round
            if count_comm
                obj.comm_rounds = obj.comm_rounds + 1.0;
            end
        end

        function result = main_iters(obj, options)
            % Solve the consensus optimization problem
            %
            % Iterates D-ripALM until convergence or maximum iterations reached.
            % Convergence: ||Zx^k|| < tol AND ||\nabla Fx^{k+1} - Z\lambda^{k+1}|| < tol
            %
            % Args:
            %   max_iter: Maximum outer iterations (default: 100)
            %   tol: Convergence tolerance (default: 1e-6)
            %   verbose: Print progress every 10 iterations (default: true)
            %
            % Returns:
            %   result: Struct containing solution and convergence info

            % if isfield(options, 'scaledata'),   scaledata = options.scaledata;      else,   scaledata = 1;          end
            if isfield(options, 'sigfun'), 		sigfun = options.sigfun;            else, sigfun = @(iter) min(1e4, max(1e-4, 1.5^(iter-1)));	end
            if isfield(options, 'taufun'), 		taufun = options.taufun;            else, taufun = @(iter) 5;       end
            if isfield(options, 'maxiter'),     maxiter = options.maxiter;          else, maxiter = 1000;           end
            if isfield(options, 'maxapgiter'),  maxapgiter = options.maxapgiter;    else, maxapgiter = 500;         end
            if isfield(options, 'maxtotaliter'),maxtotaliter = options.maxtotaliter;else, maxtotaliter = 15000;     end
            if isfield(options, 'maxtime'),     maxtime = options.maxtime;          else, maxtime = 3600;           end
            if isfield(options, 'stoptol'),     stoptol = options.stoptol;          else, stoptol = 1e-6;           end
            if isfield(options, 'display'),     display = options.display;          else, display = 1;              end
            if isfield(options, 'warm_start_inner'), warm_start_inner = options.warm_start_inner; else, warm_start_inner = 1; end

            %% choose inexact error criterion
            % 0: absolute type;
            % 1: relative type with correction;
            % 2: pure relative type;
            % 3: heuristic relative type
            if isfield(options, 'inexacttype'), inexacttype = options.inexacttype;  else, inexacttype = 2;          end
            if isfield(options, 'tolfun'),      tolfun = options.tolfun;            else, tolfun = @(iter) [1/(iter)^1.1, 1/(iter)^1.1]; end
            if isfield(options, 'rhofun'),      rhofun = options.rhofun;            else, rhofun = @(iter) 0.99;    end
            % add an extra error varibale w for the pure relative type
            obj.w = zeros(obj.N, 1);
            % Auxiliary-variable restart: none (used if omitted) or default.
            if isfield(options, 'restart_w_strategy')
                restart_w_strategy = options.restart_w_strategy;
            else
                restart_w_strategy = 'none';
            end
            assert(any(strcmp(restart_w_strategy, {'none', 'default'})), ...
                'Invalid restart_w_strategy: choose none or default.');

            %% choose APG
            % extrapo: 0. DG; 1. FISTA; 2. NAG-SC scheme I; 3. NAG-SC scheme II;
            if isfield(options, 'extrapo'),     extrapo = options.extrapo;          else, extrapo = 1;              end
            % theta_choice: 1. FISTA; 2. Yang's expansion
            if isfield(options, 'theta_choice'),theta_choice = options.theta_choice;else, theta_choice = 1;         end
            if isfield(options, 'q_cond'),      q_cond = options.q_cond;            else, q_cond = 0;               end
            if isfield(options, 'restart_APG'), restart_APG = options.restart_APG;  else, restart_APG = 0;          end
            if isfield(options, 're_freq'),     re_freq = options.re_freq;          else, re_freq = 500;            end

            global KKTres_fun

            %% main loop (initialization)
            breakyes = 0;
            msg = [];
            grad_F = obj.compute_global_gradient(obj.x);
            Zxt = obj.Z_mul(obj.x, false);
            error_ratio = 0;
            comm_count = 0;
            totaliter = 0;
            num_restart_w = 0;

            % Scheduled restart flag
            do_w_restart = 0;

            % Metrics (validate by God)
            obj_val = obj.compute_global_objective(obj.x);
            consensus_viol = norm(Zxt);
            primal_res = norm(grad_F + obj.Omega);

            if display
                fprintf('%-6s %-15s %-15s %-15s %-15s %-15s \n', 'Iter', 'Objective', 'Consensus', 'Primal Res', 'Error Ratio', 'Comms.');
                fprintf('%-6d %-15.6e %-15.6e %-15.6e %-15.6e %-15d \n', 0, obj_val, consensus_viol, primal_res, error_ratio, comm_count);
            end

            t_start = tic;
            for iterk = 1:maxiter
                % Restart w according to the default schedule.
                restart_now = strcmp(restart_w_strategy, 'default') && do_w_restart == 1;
                do_w_restart = 0;

                % Execute restart
                if restart_now
                    obj.w = obj.x;
                    num_restart_w = num_restart_w + 1;
                    do_w_restart = 0;
                end

                % Outer iteration index is obj.iteration
                x_prev = obj.x;
                Omegak = obj.Omega;
                % Choose sigma and tau for this outer iteration
                abstol = tolfun(iterk);
                sigmak = sigfun(iterk);
                tauk = taufun(iterk);
                rho = rhofun(iterk);

                if warm_start_inner == 1, xt = x_prev; else, xt = zeros(obj.N, 1); end

                xdiff = xt - x_prev;
                % grad_L = grad_F + Omegak + sigmak * Zxt;
                % Delta = grad_L + tauk/sigmak*xdiff;
                sc_modulu = obj.sc_f + tauk/sigmak;
                [step_size, Lip_at_k] = obj.compute_subproblem_step_size(sigmak, tauk);
                if sc_modulu > 0
                    q_cond = sc_modulu / Lip_at_k;
                end
                criterion_satisfied = false;

                % extrapolation settings
                if extrapo == 1
                    theta = 1; theta_old = 1;
                    % 1: use EQ; 2: (alpha-1)/(k+alpha-1)
                    if theta_choice == 2, aa = 5; end
                    extrapo_modulu = theta*(1/theta_old-1);
                elseif extrapo == 2 && sc_modulu > 0
                    theta = 1; theta_old = 1;
                    extrapo_modulu = 0;
                else
                    extrapo_modulu = 0;
                end

                xt = x_prev;
                xet = xt;
                breakyessub = 0;
                msgsub = [];
                re_freq_in = re_freq;
                num_restart_APG = 0;

                if display
                    fprintf('    %6s | %-15s | %-9s %-9s %-9s %-9s | %-9s %-9s \n', ...
                        'itert', 'gradL', 'errorL', 'errorR', 'Deltap', 'Deltad', 'step-size', 'comms.');
                end

                for itert = 0:maxapgiter

                    % Validate the error criterion (We just theoretically check it)
                    if (rem(itert, 10) == 0 || itert == maxapgiter) && itert > 0

                        % compute residuals and errors at xt;
                        Zxt = obj.Z_mul(xt, false);
                        xdiff = xt - x_prev;
                        grad_F_xt = obj.compute_global_gradient(xt);
                        subgrad_F_xt = (xt_tmp - xt)/step_size;
                        Delta = subgrad_F_xt + grad_F_xt + Omegak + sigmak * Zxt + tauk/sigmak*xdiff;

                        normxdiff = norm(xdiff);

                        % compute residuals and errors;
                        norm_Delta  = norm(Delta);
                        Deltad = sigmak*sqrt(abs(xt'*Zxt));
                        Deltap = normxdiff;

                        % choose inexact error criterion
                        if inexacttype == 0  % absolute type
                            tolconst = min(1, sqrt(tauk)) / sigmak;
                            rule_lhs_1 = 0;
                            rule_lhs_2 = norm_Delta;
                            errorL = norm_Delta;
                            errorR = tolconst * min(abstol(2)*sqrt(Deltad^2+tauk*Deltap^2), abstol(1));
                        elseif inexacttype == 1  % relative type with correction
                            tolconst = min(1, sqrt(tauk)) / sigmak;
                            rule_lhs_1 = 0;
                            rule_lhs_2 = norm_Delta;
                            errorL = norm_Delta;
                            errorR = rho * tolconst * sqrt(Deltad^2+tauk*Deltap^2);
                        elseif inexacttype == 2  % pure relative type
                            if strcmp(restart_w_strategy, 'default') && iterk <= 3
                                % Omit the cross term during the first three outer iterations.
                                rule_lhs_1 = 0;
                            else
                                rule_lhs_1 = (2/sigmak) * abs((obj.w - xt)' * Delta);
                            end
                            rule_lhs_2 = norm_Delta^2;
                            rule_lhs = rule_lhs_1 + rule_lhs_2;
                            rule_rhs = rho/sigmak^2 * (Deltad^2 + tauk*Deltap^2);
                            errorL = sqrt(rule_lhs);
                            errorR = sqrt(rule_rhs);
                        elseif inexacttype == 3  % heuristic relative type
                            rule_lhs_1 = 0;
                            rule_lhs_2 = norm_Delta;
                            rule_rhs = rho/sigmak^2 * (Deltad^2 + tauk*Deltap^2);
                            errorL = norm_Delta;
                            errorR = sqrt(rule_rhs);
                        end

                        % transfer only scalar
                        if isgpuarray(norm_Delta) == 1
                            norm_Delta   = gather(norm_Delta);
                            normgradL = gather(normgradL);
                            errorL  = gather(errorL);
                            errorR  = gather(errorR);
                            Deltad  = gather(Deltad);
                            Deltap  = gather(Deltap);
                        end

                        % record the start status
                        if itert == 0
                            errorLs = errorL;
                            errorRs = errorR;
                        end

                        %% check for termination
                        if errorL < errorR
                            criterion_satisfied = true;
                            breakyessub = 1;
                            if inexacttype == 0
                                msgsub = "Absolute error criterion is satisfied. (SNIPAL)";
                            elseif inexacttype == 1
                                msgsub = "Relative error criterion with correction is satisfied. (ciPALM)";
                            elseif inexacttype == 2
                                msgsub = "Pure relative error criterion is satisfied. (ripALM)";
                                if tauk == 0
                                    msgsub = "Pure relative error criterion is satisfied. (iALM [Eckstein-Silva-2013])";
                                end
                            elseif inexacttype == 3
                                msgsub = "Heuristic relative error criterion is satisfied.";
                            elseif inexacttype == 4
                                msgsub = "Heuristic relative error criterion is satisfied.";
                            end
                        end

                        % safeguard to break
                        if norm_Delta < 1e-12 && breakyessub == 0
                            breakyessub = 1;
                            msgsub = "pALM subproblem is nearly exactly solved.";
                        end

                        if max(errorL,errorR) < 1e-12 && breakyessub == 0
                            breakyessub = 1;
                            msgsub = "Errors are too small to compare.";
                        end

                        if (itert == maxapgiter) && breakyessub == 0
                            breakyessub = 2;
                            msgsub = "maximum inner iterations reached.";
                        end

                        if (totaliter + itert >= maxtotaliter) && breakyessub == 0
                            breakyessub = 3;
                            msgsub = "maximum total iterations reached.";
                        end

                        if display && (rem(itert, 1) == 0 || itert == 0 || breakyessub)
                            fprintf('    %6d | %-15.8e | %-9.2e %-9.2e %-9.2e %-9.2e | %-9.2e %-9d \n', ...
                                itert, norm_Delta, errorL, errorR, Deltad, Deltap, step_size, obj.comm_rounds);
                        end
                    end

                    %% check to early stop
                    if breakyessub > 0
                        fprintf("\n        %s\n", msgsub);
                        break;
                    end

                    x0 = xt;
                    % local grad at extrapolation point
                    grad_F_xet = obj.compute_global_gradient(xet);
                    % proximal term
                    xetdiff = xet - x_prev;
                    % coupling term
                    Zxet = obj.Z_mul(xet, true);
                    % gradient at extrapolation point
                    grad_L_xet = grad_F_xet + Omegak + sigmak * Zxet + tauk/sigmak * xetdiff;

                    % gradient descent step
                    xt_tmp = xet - step_size * grad_L_xet;
                    % proximals block-wise
                    xt = obj.apply_prox(xt_tmp, step_size);

                    % update successive change
                    succhgx = xt - x0;

                    % update theta & restart
                    if extrapo == 1
                        if restart_APG == 1
                            if mod(itert, re_freq_in) == 0
                                if display == 1, fprintf(' *************** restart **************** \n'); end
                                theta = 1; theta_old = 1; num_restart_APG = num_restart_APG + 1;

                            else
                                theta_old = theta;
                                if theta_choice == 1
                                    theta = (sqrt(theta^4+4*theta^2)-theta^2)/2;
                                elseif theta_choice == 2
                                    theta = (aa-1)/(itert+aa);
                                end
                            end
                        else
                            theta_old = theta;
                            if theta_choice == 1
                                theta = (sqrt(theta^4+4*theta^2)-theta^2)/2;
                            elseif theta_choice == 2
                                theta = (aa-1)/(itert+aa);
                            end
                        end
                        extrapo_modulu = theta*(1/theta_old-1);
                    elseif extrapo == 2 && sc_modulu > 0
                        if restart_APG == 1
                            if mod(itert, re_freq_in) == 0 %|| (itert>0 && sum(sum([(ue-u).*succhgu; (ve-v).*succhgv])) > 0)
                                if display == 1, fprintf(' *************** restart **************** \n'); end
                                theta = 1; theta_old = 1; num_restart_APG = num_restart_APG + 1;

                            else
                                theta_old = theta;
                                theta = (sqrt((q_cond - theta_old^2)^2 + 4*theta_old^2) + q_cond - theta_old^2)/2;
                            end
                        else
                            theta_old = theta;
                            theta = (sqrt((q_cond - theta_old^2)^2 + 4*theta_old^2) + q_cond - theta_old^2)/2;
                        end
                        extrapo_modulu = theta_old * (1 - theta_old) / (theta_old^2 + theta);
                    elseif extrapo == 3 && sc_modulu > 0
                        extrapo_modulu = (1 - sqrt(q_cond)) / (1 + sqrt(q_cond));
                    else
                        extrapo_modulu = 0;
                    end

                    % extrapolation
                    xet = xt + extrapo_modulu*succhgx;

                end

                totaliter = totaliter + itert;

                %% update dual variable
                % Zxt = obj.Z_mul(xt, true);
                Omega_new = Omegak + sigmak * Zxt;
                w_new = obj.w - sigmak * Delta;

                % Metrics (validate by God)
                obj_val = obj.compute_global_objective(xt);
                consensus_viol = sqrt(abs(xt'*Zxt));
                if ~isempty(KKTres_fun)
                    x_mean = zeros(obj.d, 1);
                    for i = 1:obj.n_agents
                        idx = (i-1)*obj.d + (1:obj.d);
                        x_i = xt(idx);
                        x_mean = x_mean + x_i;
                    end
                    x_mean = x_mean / obj.n_agents;
                    x_mean(abs(x_mean)/norm(x_mean, 'inf') < 1e-8,:) = 0;
                    primal_res = KKTres_fun(x_mean);
                else
                    primal_res = norm(xt - obj.apply_prox(xt-grad_F_xt-Omega_new, 1));
                end
                error_ratio = errorL / errorR;

                % Update state
                obj.x = xt;
                obj.Omega = Omega_new;
                obj.w = w_new;

                % Record history
                obj.history.objective = [obj.history.objective; obj_val];
                obj.history.consensus_violation = [obj.history.consensus_violation; consensus_viol];
                obj.history.stational_violation = [obj.history.stational_violation; primal_res];
                obj.history.relative_error = [obj.history.relative_error; error_ratio];
                obj.history.criterion_satisfied = [obj.history.criterion_satisfied; criterion_satisfied];
                obj.history.sigma = [obj.history.sigma; sigmak];
                obj.history.tau = [obj.history.tau; tauk];
                obj.history.comm_rounds = [obj.history.comm_rounds; obj.comm_rounds];

                % Advance iteration counter
                obj.iteration = obj.iteration + 1;

                time_now = toc(t_start);
                %% check for termination
                if max(consensus_viol, primal_res) < stoptol
                    breakyes = 1;
                    msg = sprintf('\nConverged at iteration %d\n', iterk);
                elseif time_now > maxtime
                    breakyes = 3;
                    msg = "Maximum time reached.";
                elseif totaliter >= maxtotaliter
                    breakyes = 4;
                    msg = "Maximum APGs reached.";
                end

                if display
                    fprintf('%-6d %-15.6e %-15.6e %-15.6e %-15.6e %-15d\n', ...
                        iterk, obj_val, consensus_viol, primal_res, error_ratio, obj.comm_rounds);
                end

                if breakyes > 0
                    fprintf("\n %s \n", msg);
                    break;
                end

                if strcmp(restart_w_strategy, 'default')
                    if iterk <= 3
                        do_w_restart = 1;
                    elseif iterk <= 10
                        if rem(iterk, 2) == 0
                            do_w_restart = 1;
                        end
                    else
                        if rem(iterk, 3) == 0
                            do_w_restart = 1;
                        end
                    end
                end
            end
            wall_time = toc(t_start);

            % Compute consensus solution (average across agents)
            x_mean = zeros(obj.d, 1);
            for i = 1:obj.n_agents
                idx = (i-1)*obj.d + (1:obj.d);
                x_i = obj.x(idx);
                x_mean = x_mean + x_i;
            end
            x_mean = x_mean / obj.n_agents;
            x_mean(abs(x_mean)/norm(x_mean, 'inf') < 1e-8,:) = 0;

            result = struct(...
                'x_consensus', x_mean, ...
                'x_all', obj.x, ...
                'iterations', iterk, ...
                'objective', obj_val, ...
                'consensus_violation', consensus_viol, ...
                'stational_violation', primal_res, ...
                'comm_rounds', obj.comm_rounds, ...
                'wall_time', wall_time, ...
                'history', obj.history);
        end
    end
end
