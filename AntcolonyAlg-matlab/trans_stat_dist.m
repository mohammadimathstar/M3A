function [Prob, Weight_mat, StationaryDist] = trans_stat_dist(Option) 
    %% ***************************************************
    %{
    It computes the transition probability and stationary distribution 
    :param Distance: distance to the underlying tangent space,
    :param NearestNeighbor: indices of nearest neighbor of samples,
    :param eVal: each column contains the eigenvalues of a sample,    
    :return: it returns the transition probability matrix and its stationary distribution 
    %}
    NearestNeighbor = Option.indices ;  
    pheromone = Option.pheromone ;
    
    d = Option.dim ;
    if length(d)==1
        Weight = Option.Weights_hard ;
    else
        Weight = Option.Weights ;
    end
    Gamma = Option.Gamma ;  
    
    disp('computing the transition probability and stationary distribution ...')
    
    %% ********* Randomly intialization of Ants
    Prob = zeros(length(pheromone));
    Weight_mat = zeros(length(pheromone));
    for i=1:length(pheromone)             
        Index = NearestNeighbor{i,1} ; 
        W = Weight{i,1} ;
        
        H = pheromone(Index) / sum(pheromone(Index)) ;
        P = (W.^Gamma) .* (H.^(1-Gamma)) ;%bsxfun(@times, W.^Gamma, H.^(1-Gamma)) ;  
        
        Weight_mat(i, Index) = W ;
        Prob(i,Index) = P'/sum(P) ;                                         
    end    
    [StationaryDist , ~] = eigs ( real(Prob)' ,1 ) ;    
    StationaryDist = real(StationaryDist) / sum(real(StationaryDist)) ;    
end
