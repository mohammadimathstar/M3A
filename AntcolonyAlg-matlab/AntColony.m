function pheromone = AntColony(Option)
    %% ***********************************************
    %{
    :param method: rw: random walk or st: stationary distribution
    %}        	
        
    %% ***************** Searching Algorithm ********************  
    N = length(Option.indices) ;
    
   	for Loop=1:Option.NumberOfIteration 
        
        freqOfvisit = zeros(N,1);        
        if Option.method == 'st'            
            [~, ~, freqOfvisit] = trans_stat_dist(Option) ;                            
        else                       
            parfor ant= 1:Option.NumOfAnts                
                propOfvisit = RandomWalk(Option);
                freqOfvisit = freqOfvisit + propOfvisit / Option.NumOfAnts ;
            end    
        end
        
    	Option.pheromone = Option.c * Option.EvapRate * freqOfvisit + ...
                           (1-Option.EvapRate)*Option.pheromone;   
    end
    pheromone = Option.pheromone ;
    
end
