function [xp] = prox_l1_GPU(x, lam)
    
    xp = sign(x) .* max(0, abs(x) - lam);

end