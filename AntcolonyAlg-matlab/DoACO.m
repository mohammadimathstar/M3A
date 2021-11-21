function Option = DoACO(Data, r, k, numsteps, numants, d)   
    Option.k = k ;
    Option.radius = r ;
    
    if nargin==3
        Option.NumberOfSteps = length(Data) ;
        Option.NumOfAnts = 10 ;
        Option.dim = 1:size(Data,2);
    elseif nargin==4
        Option.NumberOfSteps = numsteps ;
        Option.NumOfAnts = 10 ;
        Option.dim = 1:size(Data,2);
    elseif nargin==5
	    Option.NumberOfSteps = numsteps ;
        Option.NumOfAnts = numants ;
        Option.dim = 1:size(Data,2);
    elseif nargin==6
        Option.NumberOfSteps = numsteps ;
        Option.NumOfAnts = numants ;
        Option.dim = d;
    else
        error('number of input is bigger or less than enough')
	end
    Option.Gamma = 0.9 ;
    Option.EvapRate = 0.1 ;           
    Option.NumberOfIteration = 10 ;
    Option.p = 50 ;
    Option.c = 2 ;
%     Option.dim = 1:size(Data,2);
    
    Option.method = 'rw';
    Option.data = Data; 
    N = length(Data) ;
    Option.pheromone = ones(N,1)/N ;

    disp('preprocessing')
%     Option = Preprocess_distToTang2(Option) ;
    Option = Preprocess_distToTang(Option) ;

    disp('Ant colony')
    pheromone = AntColony(Option);
    Option.pheromone = pheromone;
end
