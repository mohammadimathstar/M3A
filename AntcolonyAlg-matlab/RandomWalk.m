function FreqOfVisit = RandomWalk(Option)
    %% ******************************************************
    %{
    It perform a random walk:
    :param Distance: distance to the underlying tangent space,
    :param NearestNeighbor: indices of nearest neighbor of samples,
    :param eVal: each column contains the eigenvalues of a sample,    
    :return: the number of time the ant visits samples
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
    N_walk = Option.NumberOfSteps ;  
    
    NumOfVisit = zeros(length(pheromone),1) ;
    
    %% ********* Randomly intialization of Ants *************
    CurrentPos = randi([1 length(pheromone)] , 1 , 1) ;       
    NumOfVisit(CurrentPos) = NumOfVisit(CurrentPos)+1 ;
    r = rand(N_walk,1);
    
    for i=1:N_walk          
        Index = NearestNeighbor{CurrentPos,1} ;
        W = Weight{CurrentPos,1} ;        
        
        H = pheromone(Index) / sum(pheromone(Index)) ;
        P = (W.^Gamma) .* (H.^(1-Gamma)) ;
        P = cumsum(P/sum(P)) ;        
          
        CurrentPos = Index(find(P>=r(i),1)) ;          
        NumOfVisit(CurrentPos) = NumOfVisit(CurrentPos)+1 ; 
    end    
    FreqOfVisit = NumOfVisit / sum(NumOfVisit) ;
end
