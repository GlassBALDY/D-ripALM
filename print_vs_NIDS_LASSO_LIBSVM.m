% One horizontal three-lambda table per LIBSVM dataset; read saved results only.
clear; clc;

%% User configuration
script_dir = fileparts(mfilename('fullpath'));
results_base_dir = fullfile(script_dir, 'results', 'Benchmark_LASSO_LIBSVM');
% Report the three bundled LIBSVM datasets by default.
dataset_names = {'abalone_scale', 'bodyfat_scale', 'space_ga_scale'};
ranseeds = 0; % fixed instance: no artificial +/-0 or cross-dataset averages
lam_moduli = [1e-1, 1e-2, 1e-3];
n_agents = 20;
rho = 0.99;
tau0 = 1e-3;
graphs = {'ring', 'erdos_renyi', 'geometric'};
graphs_latex = {'Ring', 'Erd\H{o}s--R\''enyi', 'Geometric'};
alg_names = {'DripALMre_FISTA', 'NIDS', 'PG_EXTRA'};
alg_display = {'D-ripALM-re', 'NIDS', 'PG-EXTRA'};
stoptol = 1e-6;
time_field = 'time_cpu_total';
time_kind = 'elapsed';

if numel(ranseeds)~=1
    error('lasso:Configuration','LIBSVM reports a fixed single seed; select one ranseeds value.');
end
graph_count=numel(graphs); alg_count=numel(alg_names); lam_count=numel(lam_moduli);
all_stats=cell(1,numel(dataset_names));
sizes=zeros(numel(dataset_names),2);
% Validate all inputs before emitting tables; never silently omit a dataset/run.
for ds=1:numel(dataset_names)
    dataset_name=dataset_names{ds};
    info=whos('-file',fullfile(script_dir,'data','LIBSVM',[dataset_name '.mat']),'A');
    m=info.size(1); d=info.size(2); sizes(ds,:)=[m,d];
    stats=struct([]);
    for l=1:lam_count
        for g=1:graph_count
            for a=1:alg_count
                file=build_filename(results_base_dir,alg_names{a},lam_moduli(l),dataset_name, ...
                    graphs{g},d,n_agents,m,ranseeds,rho,tau0);
                stat=collect_run_stats({file},ranseeds,time_field,stoptol);
                if isempty(stats), stats=repmat(stat,graph_count,alg_count,lam_count); end
                stats(g,a,l)=stat;
            end
        end
    end
    all_stats{ds}=stats;
end

%% Old horizontal layout, with elapsed time added inside each lambda group.
for ds=1:numel(dataset_names)
    stats=all_stats{ds};
    fprintf('\\begin{table}[t]\n\\centering\n\\footnotesize\n');
    fprintf('\\setlength{\\tabcolsep}{3pt}\n');
    fprintf('\\caption{LASSO on %s: $m=%d$, $d=%d$, %d agents, seed %d.}\n', ...
        strrep(dataset_names{ds},'_','\_'),sizes(ds,1),sizes(ds,2),n_agents,ranseeds);
    fprintf('\\label{tab:lasso_libsvm_%s}\n',dataset_names{ds});
    fprintf('\\resizebox{\\linewidth}{!}{%%\n');
    fprintf('%s\n',['\begin{tabular}{ll' repmat('ccc',1,lam_count) '}'],'\toprule');
    fprintf('\\multirow{2}{*}{Topology} & \\multirow{2}{*}{Method}');
    for l=1:lam_count
        lambda=format_lambda(lam_moduli(l));
        fprintf(' & \\multicolumn{3}{c}{$\\lambda_c=%s$}',lambda(2:end-1));
    end
    fprintf('%s\n',' \\');
    for l=1:lam_count, fprintf('\\cmidrule(lr){%d-%d}',3*l,3*l+2); end
    fprintf('\n & ');
    for l=1:lam_count, fprintf(' & \\texttt{iter.(\\#)} & \\texttt{time(s)} & \\texttt{KKTres}'); end
    fprintf('%s\n',' \\','\midrule');
    for g=1:graph_count
        for a=1:alg_count
            topology='';
            if a==1, topology=sprintf('\\multirow{%d}{*}{%s}',alg_count,graphs_latex{g}); end
            fprintf('%s & %s',topology,color_method_entry(alg_display{a},alg_names{a}));
            for l=1:lam_count
                s=stats(g,a,l);
                iter=sprintf('%.0f',s.iter_mean);
                if contains(alg_names{a},'DripALM'), iter=[iter sprintf(' (%.0f)',s.outer_mean)]; end
                cells={iter,sprintf('%.2f',s.time_mean),sprintf('%.1e',s.residual_mean)};
                for c=1:3, fprintf(' & %s',color_method_entry(cells{c},alg_names{a})); end
            end
            fprintf('%s\n',' \\');
        end
        if g<graph_count, fprintf('\\midrule\n'); end
    end
    fprintf('%s\n','\bottomrule','\end{tabular}}','\par\smallskip');
    fprintf('\\parbox{\\linewidth}{\\scriptsize ');
    fprintf('Single run per setting; parentheses: outer iterations for D-ripALM-re (default). ');
    if strcmp(time_kind,'cpu'), description='CPU time'; else, description='elapsed time'; end
    fprintf('Times are %s in seconds; KKTres is the final per-run residual. ',description);
    fprintf('Iter. counts cumulative inner updates or single-loop updates, excluding additional global stopping-check communication. ');
    fprintf('All termination costs, including unsuccessful runs, are retained.\\par\n');
fprintf('Not reaching tolerance (final KKTres $\\geq$ \\texttt{%.0e}), failed/$N$; methods in order: %s.\\par\n', ...
        stoptol,strjoin(alg_display,', '));
    for g=1:graph_count
        fprintf('%s: ',graphs_latex{g});
        for l=1:lam_count
            fprintf('$\\lambda_c=%s$: ',num2str(lam_moduli(l)));
            for a=1:alg_count
                s=stats(g,a,l); fprintf('%d/%d',s.failed_count,s.n_runs);
                if a<alg_count, fprintf(', '); end
            end
            if l<lam_count, fprintf('; '); end
        end
        fprintf('.\\par\n');
    end
    fprintf('}\n\\end{table}\n\n');
end

%% Local helpers
function str = color_method_entry(str,alg_name)
if contains(alg_name,'DripALM')
    str = ['\blue{' str '}'];
end
end

function file_name = build_filename(base_dir,alg_name,lam_mod,dataset_name,graph,d,n_agents,m,ranseed,rho,tau0)
folder = fullfile(dataset_name,sprintf('lambda_modu%s_graph_%s',num2str(lam_mod),graph));
name = sprintf('%s_lam_modu%s_data_%s_d%d_n%d_samp%d_graph_%s_seed%d', ...
    alg_name,num2str(lam_mod),dataset_name,d,n_agents,m,graph,ranseed);
if contains(alg_name,'DripALM')
    name = sprintf('%s_rho%s_tau%s',name,num2str(rho),num2str(tau0));
end
file_name = fullfile(base_dir,folder,[name '.mat']);
end

function str = format_lambda(value)
exponent = log10(value);
half = round(2*exponent)/2;
if abs(exponent-half)<=1e-12*max(1,abs(exponent))
    str = sprintf('$10^{%g}$',half);
else
    power = floor(exponent); coefficient = value/10^power;
    str = sprintf('$%.10g\\times 10^{%d}$',coefficient,power);
end
end

function stat = collect_run_stats(files,seeds,time_field,stoptol)
seeds = seeds(:).';
stat = struct('seeds',seeds,'files',{files},'iters',zeros(size(seeds)), ...
    'comms',zeros(size(seeds)),'times',zeros(size(seeds)),'residuals',zeros(size(seeds)));
for k=1:numel(files)
    file = files{k};
    if ~isfile(file), error('lasso:MissingFile','Missing input file: %s',file); end
    try
        data = load(file);
    catch err
        error('lasso:InvalidFile','Cannot load input file %s: %s',file,err.message);
    end
    stat.iters(k) = checked_value(data,'iters',file,true);
    stat.comms(k) = checked_value(data,'comms',file,true);
    stat.times(k) = checked_value(data,time_field,file,true);
    stat.residuals(k) = max(checked_value(data,'feasp',file,false),checked_value(data,'feasd',file,false));
end
stat.iter_mean = mean(stat.comms); stat.iter_std = std(stat.comms,0);
stat.outer_mean = mean(stat.iters);
stat.time_mean = mean(stat.times); stat.time_std = std(stat.times,0);
stat.residual_mean = mean(stat.residuals);
stat.failed_count = sum(stat.residuals>=stoptol); stat.n_runs = numel(seeds);
end

function value = checked_value(data,field,file,scalar_required)
if ~isfield(data,field)
    error('lasso:InvalidField','File %s: missing field %s.',file,field);
end
value = data.(field);
if ~isnumeric(value) || ~isreal(value) || isempty(value) || ~isvector(value) || ...
        (scalar_required && ~isscalar(value)) || any(~isfinite(value(:))) || any(value(:)<0)
    error('lasso:InvalidField','File %s: invalid field %s (expected finite nonnegative real scalar/vector).',file,field);
end
value = double(value(end));
end
