% Print six-column LASSO statistics without running optimization experiments.

clear; clc;

%% User configuration (edit as needed)
results_base_dir = fullfile(fileparts(mfilename('fullpath')), 'results', 'Benchmark_LASSO_random');
d_list = [1000];          % problem sizes (dimension)
Lmax_list = [100];       % L_max values to report
m = 200;                  % global sample count
ranseeds = 0:9;           % seeds to average (e.g., 0:4)
lam_moduli = [1e-1, 1e-2, 1e-3]; % lambda modulus values shown in rows
% lam_moduli = [10^(-1), 10^(-2)]; % lambda modulus values shown in rows
n_agents = 20;
rho = 0.99;
tau0 = 1e-3;

graphs = {'ring', 'erdos_renyi', 'geometric'};
graphs_latex = {'Ring', 'Erd\H{o}s--R\''enyi', 'Geometric'};
% graphs = {'ring'};
% graphs_latex = {'Ring'};

alg_names_all = {'DripALMre_FISTA', 'NIDS', 'PG_EXTRA'};
alg_display_all = {'D-ripALM-re', 'NIDS', 'PG-EXTRA'};
alg_selected = [1,2,3]; % indices of algorithms to print

stoptol = 1e-6;
time_field = 'time_cpu_total';
time_kind = 'elapsed';

alg_names = alg_names_all(alg_selected);
alg_display = alg_display_all(alg_selected);

graph_count = numel(graphs);
alg_count = numel(alg_names);
lam_count = numel(lam_moduli);

%% Validate every requested setting before printing any table.
all_stats = cell(numel(d_list),numel(Lmax_list));
for di=1:numel(d_list)
    d = d_list(di);
    for li=1:numel(Lmax_list)
        Lmax = Lmax_list(li);
        stats = struct([]);
        for l=1:lam_count
            for g=1:graph_count
                for a=1:alg_count
                    files = cell(1,numel(ranseeds));
                    for k=1:numel(ranseeds)
                        files{k} = build_filename(results_base_dir,alg_names{a},lam_moduli(l),Lmax, ...
                            graphs{g},d,n_agents,m,ranseeds(k),rho,tau0);
                    end
                    stat = collect_run_stats(files,ranseeds,time_field,stoptol);
                    if isempty(stats), stats = repmat(stat,graph_count,alg_count,lam_count); end
                    stats(g,a,l) = stat;
                end
            end
        end
        all_stats{di,li} = stats;
    end
end

%% Six-column LaTeX tables (booktabs and multirow).
for di=1:numel(d_list)
    d = d_list(di);
    for li=1:numel(Lmax_list)
        Lmax = Lmax_list(li); stats = all_stats{di,li};
        fprintf('\\begin{table}[t]\n\\centering\n\\footnotesize\n');
        fprintf('\\setlength{\\tabcolsep}{4pt}\n');
        fprintf('\\caption{LASSO with $L_{\\max}=%s$, $d=%d$, $m=%d$ global samples and %d agents.}\n', ...
            num2str(Lmax),d,m,n_agents);
        label_L = strrep(strrep(strrep(num2str(Lmax),'.','p'),'+','plus'),'-','minus');
        fprintf('\\label{tab:lasso_d%d_L%s}\n',d,label_L);
        fprintf('\\begin{tabular}{lllccc}\n\\toprule\n');
        fprintf('$\\lambda_c$ & Topology & Method & iter.(\\#) & time(s) & KKTres \\\\\n\\midrule\n');
        for l=1:lam_count
            for g=1:graph_count
                for a=1:alg_count
                    lambda_cell = ''; topology_cell = '';
                    if g==1 && a==1
                        lambda_cell = sprintf('\\multirow{%d}{*}[-5.4pt]{%s}',graph_count*alg_count,format_lambda(lam_moduli(l)));
                    end
                    if a==1
                        topology_cell = sprintf('\\multirow{%d}{*}{%s}',alg_count,graphs_latex{g});
                    end
                    stat = stats(g,a,l);
                    method_cells = {alg_display{a}, ...
                        format_iteration_entry(stat,contains(alg_names{a},'DripALM')), ...
                        format_time_entry(stat),format_sci(stat.residual_mean)};
                    fprintf('%s & %s & %s & %s & %s & %s \\\\\n',lambda_cell,topology_cell,method_cells{:});
                end
                if g<graph_count, fprintf('\\cmidrule(lr){2-6}\n'); end
            end
            if l<lam_count, fprintf('\\midrule\n'); end
        end
        fprintf('\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n');
        fprintf('\\parbox{\\linewidth}{\\scriptsize ');
        fprintf('$N=%d$ runs per setting; counts and times: mean $\\pm$ sample standard deviation (denominator $N-1$). ',numel(ranseeds));
        fprintf('Parentheses give mean $\\pm$ sample standard deviation of outer iterations for D-ripALM methods. KKTres is the mean final residual. ');
        if strcmp(time_kind,'cpu'), time_description='CPU time'; else, time_description='elapsed time'; end
        fprintf('Times are %s in seconds. Counts exclude additional global stopping-check communication. ',time_description);
        fprintf('Termination costs include all complete runs, including those not reaching tolerance. ');
        fprintf('KKTres is computed per run before averaging.\\par\n');
        fprintf('Not reaching tolerance (final residual $\\geq$ %s), failed/$N$:\\par\n',format_lambda(stoptol));
        for l=1:lam_count
            for g=1:graph_count
                fprintf('%s, %s: ',format_lambda(lam_moduli(l)),graphs_latex{g});
                for a=1:alg_count
                    stat=stats(g,a,l);
                    failure_entry = sprintf('%s %d/%d',alg_display{a},stat.failed_count,stat.n_runs);
                    fprintf('%s',color_method_entry(failure_entry,alg_names{a}));
                    if a<alg_count, fprintf('; '); end
                end
                fprintf('.\\par\n');
            end
        end
        fprintf('}\n\\end{table}\n\n');
    end
end

%% Local helpers
function str = color_method_entry(str,alg_name)
if contains(alg_name,'DripALM')
    str = ['\blue{' str '}'];
end
end

function file_name = build_filename(base_dir,alg_name,lam_mod,Lmax,graph,d,n_agents,m,ranseed,rho,tau0)
folder = sprintf('lambda_modu%s_Lmax%s_graph_%s',num2str(lam_mod),num2str(Lmax),graph);
name = sprintf('%s_lam_modu%s_Lmax%s_d%d_n%d_samp%d_graph_%s_seed%d', ...
    alg_name,num2str(lam_mod),num2str(Lmax),d,n_agents,m,graph,ranseed);
if contains(alg_name,'DripALM')
    name = sprintf('%s_rho%s_tau%s',name,num2str(rho),num2str(tau0));
end
file_name = fullfile(base_dir,folder,[name '.mat']);
end

function str = format_iteration_entry(stat,is_drip)
str = sprintf('%.0f $\\pm$ %.0f',stat.iter_mean,stat.iter_std);
if is_drip, str = [str sprintf(' (%.0f $\\pm$ %.0f)',stat.outer_mean,stat.outer_std)]; end
end

function str = format_time_entry(stat)
str = sprintf('%.2f $\\pm$ %.2f',stat.time_mean,stat.time_std);
end

function str = format_sci(value)
str = sprintf('%.1e',value);
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
stat.outer_mean = mean(stat.iters); stat.outer_std = std(stat.iters,0);
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
