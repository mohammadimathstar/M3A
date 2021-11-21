function pstruct = Preprocess_distToTang3(pstruct, sigma)
    %% **************************************************
    % ************ perplexity
    %{
    it compute the distance of a sample's neighbor to the tangent space
    :param Data: the data set
    :param Radius: the neighborhood size
    :param k: the minimum number of neighbors
    
    :return : it returns distance to tangent spaces (with d=1:D)
    %}
    
    %% *****to check whether hard or soft version *******   
    Data = pstruct.data ;
    k = pstruct.k;
    p = pstruct.p;
    dim = pstruct.dim ;    
    Radius = pstruct.radius;
    
    [N,D] = size(Data);
    
    
    %% Find nearest neighbors (at least k neighbors)
    idx = Neighbors(Data,Radius,k) ;        

    %% ********* Computing Eigen-values/vectors ********
    m = min(D, k);
    Distance = cell(N,1) ;      
    Weight = cell(N,1) ;
    Weight_hard = cell(N,1);
    EigenValue = zeros(D,N) ; 
    for i=1:N                
        X = Data(idx{i,1},:); %Mean = mean(X,1); X = X - Mean; 
        [Ev,u] = LocalPCA(X) ;
        EigenValue(1:length(Ev),i) = Ev ;
        
        Mu = KernelMean(Data(i,:),X, sigma(i)) ;
        
        % Orthogonal distance to the tangent space  
        m = min(D, size(X,1)) ;
        DistMat = zeros(size(X,1),m);
        for d=1:m            
        	DistMat(:,d) = vecnorm((eye(D) - u(:,1:d) * u(:,1:d)') * (X-Mu)',2,1)' ;
        end
        Distance{i,1} = DistMat ;
        
        [w1, w2] = importanceOfdirection(Ev) ; % w2 is a (D-1)*1 matrix   
        Weight{i,1} = ComputeWeight(DistMat, p, 1:m) * w2;
        if any(isnan(Weight{i,1}))               
            error('there is a nan value')
        end
            
        if length(dim)==1
            Weight_hard{i,1} = ComputeWeight(DistMat, p, dim) ;
        end
    end      

    %% ********************* Saving DataSet ************************** 
    pstruct.indices = idx; 
    pstruct.EgVal = EigenValue ;          
  	pstruct.DistanceToTangentSpace = Distance;   
    pstruct.Weights = Weight ;
    pstruct.Weights_hard = Weight_hard ;
%    	    save('check.mat','pstruct')
end

function idx = Neighbors(Data,Radius,k)
    [idx,~] = rangesearch(Data,Data,Radius);
    
    for i=1:length(Data)         
        if length(idx{i,1})<k+1
            [Index,~] = knnsearch(Data,Data(i,:),'K',k+1);    
            idx{i,1} = Index ;
        end
        idx{i,1}(1) = [] ;        
    end
end

function [EigVal,EigVec] = LocalPCA(X)
    [N, D] = size(X) ;
    Mean = mean(X,1);
    if N<D
        [EigVec, s, ~] = svd((X-Mean)','econ') ;
        EigVal = diag(s).^2 ./ sum(diag(s).^2,1) ;    
    else            
        [~, s, EigVec] = svd(X-Mean, 'econ');
        EigVal = diag(s).^2 ./ sum(diag(s).^2,1) ;
    end    
end

function Mu = KernelMean(x,Neighbors, sigma)    
    Kernel = exp(-vecnorm(x - Neighbors, 2, 2).^2 / (2*sigma^2));    
    Mu = sum(Kernel.* Neighbors, 1) / sum(Kernel);   
end


