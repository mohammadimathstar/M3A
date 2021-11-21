function [w1, w2] = importanceOfdirection(ev)
    % it computes two different ways to specify the probability that the
    % intrinsic dimensionality of a manifold in a specific value
    D = length(ev);
    w1 = [ev(1:D-1) - ev(2:D); ev(D)] ;
    w2 = w1 ;
    for i=1:D
        w2(i) = i * w2(i);
    end
end