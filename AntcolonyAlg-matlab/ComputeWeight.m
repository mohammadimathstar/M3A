function Weight = ComputeWeight(Distance, p, dim)
    %{
    It computes the weights of a data point to its neighbors
    :param Distance: distance to the underlying tangent space (estimated by PCA)
    :param dim: the dimensionality of manifold
    :param p: the percentile of neighbors with positive weights (other neighbors get zero weights)
    :return: it return a (n,1) array containing the weight of edges to the neighbors
    %}    
    D = size(Distance,2) ;    
    a = prctile(Distance(:,dim), p, 1) ;    
    
    Weight = heaviside(a - Distance(:, dim)) .* (1 - Distance(:,dim) ./ a) ;
    if length(dim)==D
        Weight(:,D) = 0 ; %since distance to the whole data space is zero
    end
    
%     Weight = bsxfun(@times,heaviside(bsxfun(@minus, a, Distance(:, dim))), (1 - bsxfun(@rdivide, Distance(:, dim), a))) ;
    
end