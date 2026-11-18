% Run tonight's experiments in order. Stop if any batch fails.
run_batches();

function run_batches()
    code_dir = fileparts(mfilename('fullpath'));
    original_dir = pwd;
    restore_dir = onCleanup(@() cd(original_dir));
    cd(code_dir);

    scripts = {
        % 'RunAll_vs_IDEAL_LR_random.m', ...
        % 'RunAll_vs_NIDS_LASSO_random.m', ...
        % 'RunAll_vs_IDEAL_LR_MNIST.m', ...
        'RunAll_vs_NIDS_LASSO_LIBSVM.m'};

    for k = 1:numel(scripts)
        fprintf('\n[%d/%d] Starting %s\n', k, numel(scripts), scripts{k});
        run_one(fullfile(code_dir, scripts{k}));
        fprintf('\n[%d/%d] Finished %s\n', k, numel(scripts), scripts{k});
    end
    fprintf('\nAll four experiment batches finished.\n');
end

function run_one(script_path)
    % Keep each batch's clear statements out of the scheduler workspace.
    run(script_path);
end
