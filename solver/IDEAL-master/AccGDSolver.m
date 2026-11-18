function [iter,err,x_end,test_acc,result]=AccGDSolver(obj_fn,grad_fn,n_agents,maxiter,x0,W,maxapgiter,L,mu,eta,eta2,beta_in,beta_out,lambda_2,lambda_n,sigma0,data_ix,x_optim,batch_size,data_full,X_test,label_test,n_test, options)
err = zeros(1,maxiter);
test_acc = zeros(1,maxiter);
[m,m1]=size(x0);
w=zeros(m,n_agents);
gamma=zeros(m,n_agents);
x=x0;

% added at 20260117
if isfield(options, 'sigfun'), 		sigfun = options.sigfun;            else, sigfun = @(iter) min(1e4, max(1e-4, 1.5^(iter-1)));	end
if isfield(options, 'maxiter'),     maxiter = options.maxiter;          else, maxiter = 1000;           end
if isfield(options, 'maxapgiter'),  maxapgiter = options.maxapgiter;    else, maxapgiter = 500;         end
if isfield(options, 'maxtotaliter'),maxtotaliter = options.maxtotaliter;else, maxtotaliter = 15000;     end
if isfield(options, 'maxtime'),     maxtime = options.maxtime;          else, maxtime = 3600;           end
if isfield(options, 'stoptol'),     stoptol = options.stoptol;          else, stoptol = 1e-6;           end
if isfield(options, 'display'),     display = options.display;          else, display = 1;              end

if isfield(options, 'inexacttype'), inexacttype = options.inexacttype;  else, inexacttype = 1;          end
if isfield(options, 'tolfun'),      tolfun = options.tolfun;            else, tolfun = @(iter) [0.8^iter, 0.8^iter]; end


totaliter = 0;
breakyes = 0;
msg = [];

err(1,1) = mean((sum((x-x_optim).^2)));
Ytest_pred = linear_prediction(X_test,mean(x,2));
test_acc(1,1) =  sum(Ytest_pred == label_test)/n_test;
val=obj_fn(mean(x,2),data_full);
iter=[val];
t_start = tic;

for iterk=1:maxiter
    xk = x;
    yk = xk;
    bar_yk = yk*W;
    abstol = tolfun(iterk);
    sigma0 = sigfun(iterk);

    %%%%%%%%%%%%% Inner Iteration %%%%%%%%%%%%%%%%%
    breakyessub = 0;
    msgsub = [];
    
    errorL = 1; errorR = 0;

    if display
        fprintf('    %6s | %-15s | %-9s %-9s %-9s %-9s | %-9s %-9s \n', ...
            'itert', 'gradL', 'errorL', 'errorR', 'Deltap', 'Deltad', 'step-size', 'comms.');
    end
    
    for itert=0:maxapgiter
        % Validate the error criterion (We just theoretically check it)
        if (rem(itert, 10) == 0 || itert == maxapgiter) && itert > 0
            
            % compute residuals and errors at xt;
            Zxk = xk*W;
            xdiff = xk - x;
            grad_xk  = zeros(m,n_agents);
            % Compute local iterates
            for p_ix=1:n_agents
                xk_local = xk(:,p_ix);
                data_local = data_ix{p_ix};
                grad_xk(:,p_ix) = grad_fn(xk_local,data_local);
            end
            Delta = gamma + grad_xk + sigma0*Zxk;
            
            normxdiff = norm(xdiff, 'fro');
            
            % compute residuals and errors;
            norm_Delta  = norm(Delta, 'fro');
            Deltad = sigma0*sqrt(abs(sum(xk.*Zxk, 'all')));
            Deltap = normxdiff;
            
            if inexacttype == 1
                errorL = (norm_Delta / mu)^2;
                errorR = abstol(1);
            end
            
            %% check for termination
            if errorL < errorR
                breakyessub = 1;
                msgsub = "Absolute error criterion is satisfied. (IDEAL)";
            end
            
            % safeguard to break
            if norm_Delta < 1e-12 && breakyessub == 0
                breakyessub = 1;
                msgsub = "ALM subproblem is nearly exactly solved.";
            end
            
            if max(errorL,errorR) < 1e-12 && breakyessub == 0
                breakyessub = 1;
                msgsub = "Errors are too small to compare.";
            end
            
            if (itert == maxapgiter) && breakyessub == 0
                breakyessub = 2;
                msgsub = "maximum inner iterations reached.";
            end
            
            if (totaliter >= maxtotaliter) && breakyessub == 0
                breakyessub = 3;
                msgsub = "maximum total iterations reached.";
            end
            
            if display 
                fprintf('    %6d | %-15.8e | %-9.2e %-9.2e %-9.2e %-9.2e | %-9.2e %-9d \n', ...
                    itert, norm_Delta, errorL, errorR, Deltad, Deltap, eta, totaliter);
            end
        end
        
        %% check to early stop
        if breakyessub > 0
            fprintf("\n        %s\n", msgsub);
            break;
        end
        
        xk_temp=xk;
        grad  = zeros(m,n_agents);
        % Compute local iterates
        for p_ix=1:n_agents
            y_local = yk(:,p_ix);
            data_local = data_ix{p_ix};
            grad(:,p_ix) = grad_fn(y_local,data_local);
        end
        grad_AL = gamma + grad + sigma0*bar_yk;
        xk= yk-eta*grad_AL;
        yk = xk + beta_in*(xk - xk_temp);
        bar_yk = yk*W;
        totaliter = totaliter + 1;
        % fprintf('out loop k = %d inner loop t = %d inner suboptimality = %.4e \n', iterk, itert, norm(grad_AL,'fro')^2)
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%% Outer Iteration %%%%%%%%%%%%%%%%%
    x=xk;
    bar_x = x*W;
    w_old=w;
    w = gamma + eta2* bar_x;
    gamma = w + beta_out*(w - w_old);
    
    consensus_viol = sqrt(abs(sum(x.*bar_x,'all')));
    primal_res = norm(grad_xk+w, 'fro');
    
    val=obj_fn(mean(x,2),data_full);
    iter=[iter,val];
    err(1,iterk+1) = mean((sum((x-x_optim).^2)));
    Ytest_pred = linear_prediction(X_test,mean(x,2));
    test_acc(1,iterk+1) =  sum(Ytest_pred == label_test)/n_test;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
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
            iterk, val, consensus_viol, primal_res, -1, totaliter);
    end
    
    if breakyes > 0
        fprintf("\n %s \n", msg);
        break;
    end
    
    
end
wall_time = toc(t_start);

x_end=x;
result = struct(...
    'x_consensus', mean(x_end,2), ...
    'x_all', x_end, ...
    'iterations', iterk, ...
    'objective', val, ...
    'consensus_violation', consensus_viol, ...
    'stational_violation', primal_res, ...
    'comm_rounds', totaliter, ...
    'wall_time', wall_time);

end
