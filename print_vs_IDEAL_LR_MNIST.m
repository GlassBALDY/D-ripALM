% Print comparison tables from saved results only.
clear; clc;

%% User configuration
results_base_dir = fullfile(fileparts(mfilename('fullpath')), 'results', 'DiffTolPara_log_reg_MNIST_shuffle_v1');
lambdas = [0.01];
topologies = {'erdos_renyi'};
tasks = {[0,8]}; % separate table per task; never average across digit pairs
ranseeds = 0:9;
alg_names = {'IDEAL', 'DripALMre_FISTA'};

d = 784;
n_agents = 10;
tau0 = 1e-3;
stoptol = 1e-6;
rhos = [0.99,0.9,0.7,0.5,0.3,0.1,0.05,0.01];
c_values = [1e-2,1e-1,1,10];
abs_factors = [0.2,0.4,0.6,0.8];

time_field = 'time_cpu_total';
time_kind = 'elapsed'; % saved time_cpu_total uses tic/toc

if ~isnumeric(ranseeds) || isempty(ranseeds) || ~isvector(ranseeds) || ...
        ~isreal(ranseeds) || any(~isfinite(ranseeds(:))) || ...
        any(ranseeds(:)<0 | ranseeds(:)>2^32-1 | ranseeds(:)~=fix(ranseeds(:))) || ...
        numel(unique(ranseeds))~=numel(ranseeds)
    error('logreg:Configuration','ranseeds must be a nonempty vector of distinct integers in [0, 2^32-1].');
end
ranseeds=double(ranseeds(:).');

for task_index=1:numel(tasks)
    task = tasks{task_index};
    for graph_ = topologies
        graph = graph_{1};
        for lam = lambdas
            stats = cell(1,numel(alg_names));
            for a=1:numel(alg_names)
                if strcmp(alg_names{a},'IDEAL')
                    params = zeros(numel(c_values)*numel(abs_factors),2);
                    p=0;
                    for c=c_values
                        for b=abs_factors, p=p+1; params(p,:)=[c,b]; end
                    end
                else
                    params = rhos(:);
                end
                for p=1:size(params,1)
                    files=cell(1,numel(ranseeds));
                    for k=1:numel(ranseeds)
                        files{k}=build_filename(results_base_dir,alg_names{a},lam,task,d, ...
                            n_agents,graph,ranseeds(k),params(p,:),tau0);
                    end
                    stat=collect_run_stats(files,ranseeds,time_field,stoptol);
                    stat.params=params(p,:);
                    if p==1, stats{a}=stat; else, stats{a}(p)=stat; end
                end
            end
            % The archived table packs the first/second halves of IDEAL beside rho.
            % Validate the full setting before emitting any LaTeX.
            caption=sprintf('MNIST %dvs%d logistic regression, $\\lambda=%s$, %d shuffled sample allocations (%d agents).', ...
                task(1),task(2),num2str(lam),numel(ranseeds),n_agents);
            label=sprintf('logreg_MNIST_%dvs%d_lam%s_%s',task(1),task(2),num2str(lam),graph);
            print_comparison(stats,caption,label,numel(ranseeds)>1,time_kind,stoptol);
        end
    end
end

%% Local helpers
function print_comparison(stats,caption,label,show_std,time_kind,stoptol)
fprintf('\\begin{table}[t]\n\\centering\n\\footnotesize\n');
fprintf('\\setlength{\\tabcolsep}{3pt}\n');
fprintf('\\caption{%s}\n\\label{tab:%s}\n',caption,label);
fprintf('\\resizebox{\\linewidth}{!}{%%\n');
fprintf('%s\n','\begin{tabular}{llll llll|llll}','\toprule', ...
    '\multicolumn{8}{c}{IDEAL} & \multicolumn{4}{c}{D-ripALM-re} \\', ...
    '\cmidrule(lr){1-8}\cmidrule(lr){9-12}');
headers={'$(\varepsilon_0,\alpha)$','\texttt{iter.(\#)}','\texttt{time(s)}','\texttt{KKTres}'};
fprintf('%s\n',[strjoin([headers,headers,{'$\rho$'},headers(2:4)],' & ') ' \\'], ...
    '\cmidrule(lr){1-4}\cmidrule(lr){5-8}\cmidrule(lr){9-12}');
half=ceil(numel(stats{1})/2);
for row=1:max(half,numel(stats{2}))
    cells={};
    for p=[row,row+half]
        if row<=half && p<=numel(stats{1})
            s=stats{1}(p);
            cells=[cells,{sprintf('(%s, %s)',num2str(s.params(1)),num2str(s.params(2)))},metric_cells(s,show_std)];
        else
            cells=[cells,{'---','---','---','---'}];
        end
    end
    if row<=numel(stats{2})
        s=stats{2}(row);
        cells=[cells,{num2str(s.params)},metric_cells(s,show_std)];
    else
        cells=[cells,{'---','---','---','---'}];
    end
    fprintf('%s\n',[strjoin(cells,' & ') ' \\']);
end
fprintf('%s\n','\bottomrule','\end{tabular}}','\par\smallskip');
fprintf('\\parbox{\\linewidth}{\\scriptsize ');
if show_std
    fprintf('Counts and times: mean $\\pm$ sample standard deviation over %d seeds ($N-1$ denominator). ',stats{1}(1).n_runs);
    fprintf('Parentheses: mean $\\pm$ sample standard deviation of outer iterations; KKTres: mean of final per-run $\\max(\\mathrm{feasp},\\mathrm{feasd})$. ');
else
    fprintf('Single run per setting; parentheses: outer iterations; KKTres: final $\\max(\\mathrm{feasp},\\mathrm{feasd})$. ');
end
fprintf('Samples are shuffled and repartitioned for each seed. ');
if strcmp(time_kind,'cpu'), description='CPU time'; else, description='elapsed time'; end
fprintf('Times are %s in seconds. All termination costs, including unsuccessful runs, are retained. ',description);
fprintf('Iter. counts saved inner updates; additional global stopping-check communication is excluded.\\par\n');
fprintf('Not reaching tolerance (final KKTres $\\geq$ \\texttt{%.0e}), failed/$N$: IDEAL ',stoptol);
for p=1:numel(stats{1})
    s=stats{1}(p);
    fprintf('(%s, %s): %d/%d',num2str(s.params(1)),num2str(s.params(2)),s.failed_count,s.n_runs);
    if p<numel(stats{1}), fprintf('; '); end
end
fprintf('.\\par\nD-ripALM-re (default): ');
for p=1:numel(stats{2})
    s=stats{2}(p);
    fprintf('$\\rho=%s$: %d/%d',num2str(s.params),s.failed_count,s.n_runs);
    if p<numel(stats{2}), fprintf('; '); end
end
fprintf('.}\n\\end{table}\n\n');
end

function cells=metric_cells(stat,show_std)
if show_std
    iter=sprintf('%.0f $\\pm$ %.0f (%.0f $\\pm$ %.0f)',stat.iter_mean,stat.iter_std,stat.outer_mean,stat.outer_std);
    time=sprintf('%.2f $\\pm$ %.2f',stat.time_mean,stat.time_std);
else
    iter=sprintf('%.0f (%.0f)',stat.iter_mean,stat.outer_mean);
    time=sprintf('%.2f',stat.time_mean);
end
cells={iter,time,sprintf('%.1e',stat.residual_mean)};
end

function file=build_filename(base_dir,alg_name,lam,digits_pair,d,n_agents,graph,seed,params,tau0)
dataset_tag=sprintf('MNIST_%dvs%d',digits_pair(1),digits_pair(2));
folder=sprintf('lambda%s_%s',num2str(lam),dataset_tag);
% Sample counts differ across binary cases; resolve from saved filenames.
name=sprintf('%s_%s_lam%s_d%d_n%d_samp*_graph_%s_seed%d', ...
    alg_name,dataset_tag,num2str(lam),d,n_agents,graph,seed);
switch alg_name
    case 'IDEAL'
        suffix=sprintf('_cval%s_abs_fac%s',num2str(params(1)),num2str(params(2)));
    case {'DripALM_FISTA','DripALMre_FISTA'}
        suffix=sprintf('_rho%s_tau%s',num2str(params),num2str(tau0));
    otherwise
        error('logreg:Configuration','Unknown algorithm: %s',alg_name);
end
pattern=fullfile(base_dir,folder,[name suffix '.mat']);
matches=dir(pattern);
matches=matches(~[matches.isdir]);
if isempty(matches)
    error('logreg:MissingFile','Missing input file matching: %s',pattern);
elseif numel(matches)~=1
    error('logreg:AmbiguousFile','Expected one input file, found %d matching: %s',numel(matches),pattern);
end
file=fullfile(matches(1).folder,matches(1).name);
end

function stat=collect_run_stats(files,seeds,time_field,stoptol)
seeds=seeds(:).';
stat=struct('seeds',seeds,'files',{files},'iters',zeros(size(seeds)), ...
    'comms',zeros(size(seeds)),'times',zeros(size(seeds)),'residuals',zeros(size(seeds)));
for k=1:numel(files)
    file=files{k};
    if ~isfile(file), error('logreg:MissingFile','Missing input file: %s',file); end
    try
        data=load(file);
    catch err
        error('logreg:InvalidFile','Cannot load input file %s: %s',file,err.message);
    end
    if ~isfield(data,'ranseed') || ~isnumeric(data.ranseed) || ...
            ~isscalar(data.ranseed) || ~isreal(data.ranseed) || ...
            ~isfinite(data.ranseed) || data.ranseed~=seeds(k) || ...
            ~isfield(data,'data_protocol') || ~isequal(data.data_protocol,'shuffle-before-split-v1') || ...
            ~isfield(data,'shuffle_generator') || ~isequal(data.shuffle_generator,'mt19937ar')
        error('logreg:InvalidProtocol','File %s: expected shuffled-data protocol and seed %d.',file,seeds(k));
    end
    stat.iters(k)=checked_value(data,'iters',file,true);
    stat.comms(k)=checked_value(data,'comms',file,true);
    stat.times(k)=checked_value(data,time_field,file,true);
    stat.residuals(k)=max(checked_value(data,'feasp',file,false),checked_value(data,'feasd',file,false));
end
stat.iter_mean=mean(stat.comms); stat.iter_std=std(stat.comms,0);
stat.outer_mean=mean(stat.iters); stat.outer_std=std(stat.iters,0);
stat.time_mean=mean(stat.times); stat.time_std=std(stat.times,0);
stat.residual_mean=mean(stat.residuals);
stat.failed_count=sum(stat.residuals>=stoptol); stat.n_runs=numel(seeds);
end

function value=checked_value(data,field,file,scalar_required)
if ~isfield(data,field), error('logreg:InvalidField','File %s: missing field %s.',file,field); end
value=data.(field);
if ~isnumeric(value) || ~isreal(value) || isempty(value) || ~isvector(value) || ...
        (scalar_required && ~isscalar(value)) || any(~isfinite(value(:))) || any(value(:)<0)
    error('logreg:InvalidField','File %s: invalid field %s (expected finite nonnegative real scalar/vector).',file,field);
end
value=double(value(end));
end
