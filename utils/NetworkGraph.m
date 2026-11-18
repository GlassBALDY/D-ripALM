classdef NetworkGraph < handle
    % NetworkGraph - Network graph and mixing matrix construction for decentralized optimization
    %
    % This class provides utilities for creating communication networks and
    % computing mixing matrices for consensus-based optimization algorithms.
    %
    % Supports various network topologies (ring, complete, grid, erdos_renyi, geometric)
    % and computes symmetric doubly stochastic mixing matrices using Metropolis weights.

    properties
        n_agents        % Number of agents
        graph_type      % Type of graph ('ring', 'complete', 'grid', 'erdos_renyi', 'geometric')
        seed            % Random seed for reproducibility
        G               % Graph adjacency matrix
        W               % Mixing matrix
        degrees
        lambda_max_W
        lambda_min_W
        gap_W
    end

    methods
        function obj = NetworkGraph(n_agents, graph_type, seed)
            % NetworkGraph Constructor
            %
            % Args:
            %   n_agents: Number of agents
            %   graph_type: Type of graph ('ring', 'complete', 'grid', 'erdos_renyi', 'geometric')
            %   seed: Random seed for reproducibility (default: 42)

            if nargin < 2
                graph_type = 'ring';
            end
            if nargin < 3
                seed = 42;
            end

            obj.n_agents = n_agents;
            obj.graph_type = graph_type;
            obj.seed = seed;

            % Set random seed
            rng(seed);

            % Create graph and mixing matrix
            [obj.G, obj.seed] = obj.create_connected_graph();
            obj.W = obj.create_mixing_matrix();
            [~] = obj.compute_spectral_gap();
        end

        function [G, seed] = create_connected_graph(obj)
            % Generate a random graph and ensure it is connected
            % Try generating a connected graph, and repeat if it is not connected
            isConnected = false;
            seed = obj.seed;
            
            % Loop to ensure the graph is connected
            while ~isConnected
                % Generate the graph
                obj.seed = seed;
                G = obj.create_graph();  % Create the graph based on the selected type
                
                % Check if the graph is connected
                isConnected = obj.check_connectivity(G);
                
                if ~isConnected
                    % If not connected, modify the seed and retry
                    seed = seed + 1;  % Change seed for random generation
                    rng(seed);  % Set the new random seed
                end
            end
        end
        
        function isConnected = check_connectivity(~, G)
            components = conncomp(graph(G));
            isConnected = numel(unique(components)) == 1;  % 如果只有一个连通分量，则图是连通的
        end

        function G = create_graph(obj)
            % Create communication graph based on specified topology
            %
            % Supported topologies:
            % - ring: Cycle graph (worst-case spectral gap)
            % - complete: Fully connected graph (best connectivity)
            % - grid: 2D grid graph (moderate connectivity)
            % - erdos_renyi: Random graph with edge probability p=0.2
            % - geometric: Geometric random graph in [0,1]^2
            %
            % Returns:
            %   G: Adjacency matrix (sparse)

            n = obj.n_agents;

            switch lower(obj.graph_type)
                case 'ring'
                    % Cycle graph
                    G = sparse(n, n);
                    for i = 1:n
                        j = mod(i, n) + 1;  % Next node (circular)
                        G(i, j) = 1;
                        G(j, i) = 1;
                    end

                case 'complete'
                    % Fully connected graph
                    G = ones(n, n) - eye(n);
                    G = sparse(G);

                case 'grid'
                    % 2D grid graph, adjusted to make it connected for non-square n
                    side = ceil(sqrt(n));
                    G = sparse(n, n);
                    for i = 1:side
                        for j = 1:side
                            idx = (i-1)*side + j;
                            if idx > n, break; end
                            % Right neighbor
                            if j < side && idx+1 <= n
                                G(idx, idx+1) = 1;
                                G(idx+1, idx) = 1;
                            end
                            % Bottom neighbor
                            if i < side && idx+side <= n
                                G(idx, idx+side) = 1;
                                G(idx+side, idx) = 1;
                            end
                        end
                    end
                    % Connect remaining nodes to the grid
                    for idx = side*side + 1 : n
                        % Connect to a random existing node
                        target = randi(n - (idx - 1));  % Random node from 1 to idx-1
                        G(idx, target) = 1;
                        G(target, idx) = 1;
                    end

                case 'erdos_renyi'
                    % Random graph with edge probability p=0.2
                    p = 0.2;
                    G = sparse(n, n);
                    for i = 1:n
                        for j = i+1:n
                            if rand() < p
                                G(i, j) = 1;
                                G(j, i) = 1;
                            end
                        end
                    end

                case 'geometric'
                    % Geometric random graph: nodes uniformly placed in [0,1]^2
                    % Choose radius to achieve average degree ≈ 6
                    radius = sqrt(6.0 / (pi * n));
                    positions = rand(n, 2);
                    G = sparse(n, n);
                    for i = 1:n
                        for j = i+1:n
                            dist = norm(positions(i,:) - positions(j,:));
                            if dist < radius
                                G(i, j) = 1;
                                G(j, i) = 1;
                            end
                        end
                    end

                otherwise
                    error('Unknown graph type: %s', obj.graph_type);
            end
        end

        function W = create_mixing_matrix(obj)
            % Create symmetric doubly stochastic mixing matrix using Metropolis weights
            %
            % The Metropolis weights are defined as:
            %   w_ij = 1/(max{d_i, d_j} + 1) for (i,j) ∈ E
            %   w_ii = 1 - Σ_{j∈N_i} w_ij
            %
            % This ensures the matrix W is:
            % - Symmetric: W = W^T
            % - Doubly stochastic: W1 = 1, 1^T W = 1^T
            % - Satisfies spectral bounds: -I ≺ W ⪯ I
            %
            % Returns:
            %   W: n x n mixing matrix satisfying Assumption 2.2 in manuscript

            n = obj.n_agents;
            W = zeros(n, n);

            % Get degrees
            obj.degrees = full(sum(obj.G, 2));

            % Metropolis weights: w_ij = 1/(max{d_i, d_j} + 1) for edges
            [rows, cols] = find(obj.G);
            for k = 1:length(rows)
                i = rows(k);
                j = cols(k);
                if i < j  % Process each edge once
                    w_ij = 1.0 / (max(obj.degrees(i), obj.degrees(j)) + 1);
                    W(i, j) = w_ij;
                    W(j, i) = w_ij;
                end
            end

            % Self-loops: w_ii = 1 - sum_j w_ij
            for i = 1:n
                W(i, i) = 1.0 - sum(W(i, :));
            end
        end

        function gap = compute_spectral_gap(obj)
            % Compute spectral gap: 1 - ς where ς = max{|λ_2(W)|, |λ_n(W)|}
            %
            % The spectral gap measures network connectivity. Larger gap → better connectivity.
            % Network condition number: κ_W = (1 + ς)/(1 - ς)
            %
            % Returns:
            %   gap: Spectral gap (scalar in [0, 1])

            eigenvalues = sort(eig(obj.W), 'descend');
            varsigma = max(abs(eigenvalues(2)), abs(eigenvalues(end)));
            gap = 1.0 - varsigma;
            obj.lambda_max_W = max(eigenvalues);
            obj.lambda_min_W = min(eigenvalues);
            obj.gap_W = gap;
        end

        function kappa = compute_network_condition_number(obj)
            % Compute network condition number κ_W = (1 + ς)/(1 - ς)
            %
            % Returns:
            %   kappa: Network condition number

            eigenvalues = sort(eig(obj.W), 'descend');
            varsigma = max(abs(eigenvalues(2)), abs(eigenvalues(end)));
            kappa = (1.0 + varsigma) / (1.0 - varsigma);
        end

        function neighbors = get_neighbors(obj, agent_i)
            % Get neighbors of agent i
            %
            % Args:
            %   agent_i: Agent index
            %
            % Returns:
            %   neighbors: List of neighbor indices

            neighbors = find(obj.G(agent_i, :))';
        end
    end
end
