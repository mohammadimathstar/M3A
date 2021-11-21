import numpy as np
from functions import *
from scipy.io import savemat
import concurrent.futures

def MyPreprocess(Data, labels, Radius, k, FileName, p, d):
    """
    it performs the pre-processing step
    :param Data_c: cleaned data set
    :param Data: noisy data set
    :param labels: the label of data points
    :param Radius: neighborhood size
    :param k: the least number of neighbors
    :param FileName: it saves the result on a file
    :return: it compute the distance of data points to the underlying manifold
    """

    # **************** Setting parameters ******************    
    N = Data.shape[0]
    D = Data.shape[1]

    file_name_mat = FileName+'.mat'
    file_name_py = FileName
    
    # **************** Find nearest neighbors (at least k neighbors)
    print('Finding neighbors ...')
    idx = NN_Radius(Data, Radius, k)

    # ****************** Computing Eigen-values/vectors ******************
    print('Computing Distances to Tangent spaces ...')
    # Distance = np.empty((N,), dtype=object)
    weight_soft = np.empty((N,), dtype=object)
    weight_hard = np.empty((N,), dtype=object)
    Eval = np.zeros((N, D))
    for i in range(N):             
        X = Data[idx[i]]
        Evalue, U = LocalPCA(X)    
        Eval[i] = Evalue
        
        # Orthogonal distance to the tangent space
        DistMat = np.zeros((X.shape[0], D))
        for dim in range(D):    
            DistMat[:, dim] = ComputeDistance(X, U[0:dim+1])
        # Distance[i] = DistMat

        w1, w2 = soft_dim(Evalue)
        # w1[D-1] = 0  # the distance to D dimension space always is zero

        # ********* New version: exponentioal
        # sigma = 3
        # weight_soft[i] = np.sum(np.exp(-np.power(DistMat / sigma, 2) / 2) * w1, axis=1)
        # weight_hard[i] = np.exp(-np.power(DistMat[:, d-1] / sigma, 2) / 2)

        # ********* old version *********
        weight_soft[i] = np.sum(ComputeWeight(DistMat, np.arange(D), p) * w1, axis=1)
        weight_hard[i] = ComputeWeight(DistMat, [d-1], p)

        # *******************************

    print("Finished! \n")

    # ********************* Saving DataSet ************************** 
    pstruct = dict()
    pstruct['data'] = Data
    pstruct['idx'] = idx    
    pstruct['k'] = k 
    pstruct['radius'] = Radius
    pstruct['W_soft'] = weight_soft
    pstruct['W_hard'] = weight_hard
    pstruct['dim'] = d

    savemat(file_name_mat, {'Data': Data, 'idx': idx, 'W_soft': weight_soft, 'W_hard': weight_hard, 'radius': Radius, 'k': k, 'dim': d})
    np.savez(file_name_py, Data=Data, idx=idx, W_soft=weight_soft, W_hard=weight_hard, radius=Radius, k=k, dim=d)

    # savemat(file_name_mat, {'Data': Data, 'labels': labels, 'Distance': Distance, 'idx': idx, 'radius': Radius, 'k': k})
    # np.savez(file_name_py, Data=Data, labels=labels, idx=idx, Distance=Distance, radius=Radius, k=k, eVal=Eval)
        
    return pstruct



def ants(f,params,OutputFile,saving):
    """
    It perform the algorithm for a manifold with known dimensionality
    :param f: data set, nearest neighbors, distances
    :param params: parameters of the algorithm
    :param OutputFile: it saves the output on a file
    :param saving: 1: if you want to save the result
    :return: it returns the pheromone distribution
    """
    # *************** Loading Data set
    Data = f['Data']    
    NearestNeighbor = f['idx']
    Weight = f['W_hard']
    N = Data.shape[0]
    
    # ************** Performing Ant colony
    History = np.ones(N)/N
    for Loop in range(params['NumberOfIteration']): 
        print(f'{Loop}-th generation starts walking')

        # *********** Parallelization
        NumOfVisits = np.zeros(N)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(RandomWalk_hard, Weight, NearestNeighbor, History, params)
                       for _ in range(params['NumberOfAnts'])]

            for f in concurrent.futures.as_completed(results):
                NumOfVisits += f.result()

        # for Ants in range(params['NumberOfAnts']):
        #     N_visits = RandomWalk(Distance, NearestNeighbor, History, params['d'] - 1, params)
        #     NumOfVisits = NumOfVisits + N_visits

        StationaryDist = NumOfVisits / (params['NumberOfAnts'] * params['NumberOfSteps'])
        History = params['EvapRate'] * params['PherVsEvapPerVisit'] * StationaryDist + \
                (1 - params['EvapRate'])*History

    if saving == 1:
        savemat(OutputFile + '.mat', {'Parameters': params, 'Pheromone': History})
        np.savez(OutputFile, Parameters=params, Pheromone=History)
    return History

def ants_soft(f, params, OutputFile, saving):
    """
    It perform the algorithm when the dimensionality is unknown
    :param f: data set
    :param params: parameter values
    :param OutputFile: the name of output file
    :param saving: 1: if you want to save the output
    :return: pheromone distribution
    """
    
    # *************** Loading Data set
    Data = f['Data']    
    NearestNeighbor = f['idx']
    Distance = f['W_soft']
    N = Data.shape[0]
    D = Data.shape[1]

    # ************** Performing Ant colony  
    History = np.ones(N)/N

    for Loop in range(params['NumberOfIteration']): 
        print(f'{Loop}-th generation starts walking')
        NumOfVisits = np.zeros(N)

        # ************* Parallelization
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # results = [executor.submit(RandomWalk_soft, Distance, NearestNeighbor, eVal, History, params)
            results = [executor.submit(RandomWalk_soft, D, Distance, NearestNeighbor, History, params)
                       for _ in range(params['NumberOfAnts'])]

            for f in concurrent.futures.as_completed(results):
                NumOfVisits += f.result()

        # for Ants in range(params['NumberOfAnts']):
        #     # print('Ant: ', Ants, end=' ')
        #     N_visits = RandomWalk_soft(Distance, NearestNeighbor, eVal, History, params)
        #     NumOfVisits = NumOfVisits + N_visits

        StationaryDist = NumOfVisits / (params['NumberOfAnts'] * params['NumberOfSteps'])
        History = params['EvapRate'] * params['PherVsEvapPerVisit'] * StationaryDist + \
                (1 - params['EvapRate'])*History

        if saving == 1:
            savemat(OutputFile + str(Loop) + '.mat', {'Parameters': params, 'Pheromone': History})
            np.savez(OutputFile + str(Loop), Parameters=params, Pheromone=History)

    if saving == 1:
        savemat(OutputFile + '.mat', {'Parameters': params, 'Pheromone': History})
        np.savez(OutputFile, Parameters=params, Pheromone=History)

    return History


def Distribution(Data, Pheromone, Perc, r, k):
    """
    It computes the log likelihood, training set (PW centers): samples with high
    pheromone, test set: samples with low pheromone
    :param Data: the data set
    :param Pheromone: the pheromone distribution vector
    :param Perc: the percentile
    :param r: radius
    :param k: the minimum number of neighbors
    :return: returns the mean of log likelihood
    """
    PercValue = np.percentile(Pheromone, 100 - Perc)
    HighPherIndex = np.argwhere(Pheromone > PercValue).T[0]
    LowPherIndex = np.argwhere(Pheromone <= PercValue).T[0]

    TrainSet = Data[HighPherIndex]
    TestSet = Data[LowPherIndex]

    mean_vec = np.zeros(TrainSet.shape)
    cov_mat = np.zeros((TrainSet.shape[0], TrainSet.shape[1], TrainSet.shape[1]))
    prop_vec = np.zeros(TrainSet.shape[0])

    idx = NN_Radius(TrainSet, r, k)

    for i in range(TrainSet.shape[0]):

        mu, sigma = ManifoldParzenWindow(TrainSet[idx[i]])
        mean_vec[i] = mu
        cov_mat[i] = sigma
        prop_vec[i] = len(idx[i]) / TrainSet.shape[0]

    LLK = Loglikelihood(TestSet, mean_vec, cov_mat, prop_vec)
    return LLK
