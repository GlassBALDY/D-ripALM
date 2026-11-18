function [A_list, b_list, data_ix] = agent_split(A, b, n_agents)
% Split data matrix and observations among agents

m = size(A, 1);
per = floor(m / n_agents);

data_ix = cell(n_agents,1);
A_list = cell(n_agents, 1);
b_list = cell(n_agents, 1);

for i = 1:n_agents
    start_idx = (i-1) * per + 1;
    if i < n_agents
        end_idx = i * per;
    else
        end_idx = m;
    end
    A_list{i} = A(start_idx:end_idx, :);
    b_list{i} = b(start_idx:end_idx);
    data_ix{i} = (start_idx:end_idx);
end
end

