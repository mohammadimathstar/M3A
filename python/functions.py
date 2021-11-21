import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.gridspec as gridspec
from numpy import linalg as LA
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import KFold
from scipy.io import loadmat
from scipy.spatial import distance
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def cross_validation(data, numofsplit):
    """
    It create a list of indices for cross validation
    :param data: data set
    :param numofsplit: number of folds
    :return: indices of training/valiation/test sets
    """
    N = data.shape[0]

    trainset = np.zeros((numofsplit, N), dtype=bool)
    validationset = np.zeros((numofsplit, N), dtype=bool)
    testset = np.zeros((numofsplit, N), dtype=bool)
    trainvalset = np.zeros((numofsplit, N), dtype=bool)

    kf = KFold(n_splits=numofsplit, shuffle=True)
    for k, (trainval, test) in enumerate(kf.split(data)):
        trainvalset[k, trainval] = True
        testset[k, test] = True

        kff = KFold(n_splits=numofsplit, shuffle=True)
        for train, val in kff.split(trainvalset[k, trainval]):
            trainset[k, trainval[train]] = True
            validationset[k, trainval[val]] = True
            break

    return trainset, validationset, trainvalset, testset


def UploadingMatFiles(filename):
    f = loadmat(filename)
    data = f['data']
    labels = f['labels']
    return data, labels


def KNN(Data, K):
    """
    computes the k-nearest neighbors for each data point
    :param Data: the data set
    :param K: number of neighbors
    :return: a list of indices of neighbors for each sample
    """
    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(Data)
    # 'brute': slow. 'ball_tree'and 'kd_tree': fast
    # kd_tree is fast for low dimensional data, but ball_tree fast for high-d
    indicesNN = nbrs.kneighbors(return_distance=False)
    return indicesNN


def NN_Radius(X, r, k):
    """
    computes the neighbors for each sample
    :param X: the data set
    :param r: the neighborhood size
    :param k: the minimum number of neighbors
    :return: the list of indices of the neighbors
    """
    neigh = NearestNeighbors(radius=r, algorithm='kd_tree').fit(X)
    indicesNN = neigh.radius_neighbors(return_distance=False)  # ,sort_results=True)
    # it return is a numpy array where each element is also numpy array, therefore if you want 
    # to access j-th element in i-th row, you need to write a[i][j] (not a[i,j])
    for i in range(X.shape[0]):
        if len(indicesNN[i]) < k:
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(X)
            ind = nbrs.kneighbors([X[i]], return_distance=False)
            indicesNN[i] = ind[0, 1:]
    return indicesNN


def LocalPCA(X):
    """
    compute PCA for a set of samples
    :param X: list of samples (n * D)
    :return: eigen values and eiven vectors
    """
    D = X.shape[1]
    pca = PCA(n_components=D)
    pca.fit(X)
    EigVal = pca.singular_values_
    EigVal = EigVal / np.sum(EigVal)
    EigVec = pca.components_
    return EigVal, EigVec


def soft_dim(eval):
    D = len(eval)
    S1 = np.append([eval[:D-1] - eval[1:]], [eval[D-1]])
    S2 = np.zeros(len(eval))
    for i in range(S1.shape[0]):
        S2[i] = (i + 1) * S1[i]
    S1 = S1 / np.sum(S1)
    return S1, S2


def Plots(X, color, History, Perc, i):
    """
    it has two columns: noisy data set (in the left column) vs. the denoised data set via removing samples with lowest pheromone (in the right column)
    :param X: the data set
    :param color: the color of each sample
    :param History: a vector containing the pheromone distribution
    :param Perc: the percentile of the pheromone distribution
    :param i: the iteration number
    :return: a plot for comparing the noisy and denoised data set
    """
    D = X.shape[1]
    Mini = X.min(axis=0).reshape((D, 1))
    Maxi = X.max(axis=0).reshape((D, 1))
    MinMax = np.concatenate((Mini, Maxi), axis=1)

    PercValue = np.percentile(History, Perc)
    HighPherIndex = np.argwhere(History > PercValue)
    New = np.reshape(HighPherIndex, (len(HighPherIndex),))

    fig = plt.figure(figsize=(15, 15))
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.01, hspace=0.01)  # set the spacing between axes.
    plt.suptitle("Manifold Learning in {}-th iteration".format(i), fontsize=14)

    ax = fig.add_subplot(gs1[0], projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, s=40, cmap=plt.get_cmap('hsv'), marker=".")
    plt.xlim(MinMax[0, 0], MinMax[0, 1]), plt.ylim(MinMax[1, 0], MinMax[1, 1]), ax.set_zlim(MinMax[2, 0], MinMax[2, 1])
    ax.set_title("Noisy data set")
    ax.view_init(25, -120)

    ax = fig.add_subplot(gs1[2], projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, s=40, cmap=plt.get_cmap('hsv'), marker=".")
    plt.xlim(MinMax[0, 0], MinMax[0, 1]), plt.ylim(MinMax[1, 0], MinMax[1, 1]), ax.set_zlim(MinMax[2, 0], MinMax[2, 1])
    ax.set_title("Noisy data set")
    ax.view_init(0, -90)

    ax = fig.add_subplot(gs1[1], projection='3d')
    ax.scatter(X[New, 0], X[New, 1], X[New, 2], c=History[New], s=40, cmap=plt.get_cmap('hsv'), marker=".")
    plt.xlim(MinMax[0, 0], MinMax[0, 1]), plt.ylim(MinMax[1, 0], MinMax[1, 1]), ax.set_zlim(MinMax[2, 0], MinMax[2, 1])
    ax.set_title("De-noised data set")
    ax.view_init(25, -120)

    ax = fig.add_subplot(gs1[3], projection='3d')
    ax.scatter(X[New, 0], X[New, 1], X[New, 2], c=History[New], s=40, cmap=plt.get_cmap('hsv'), marker=".")
    plt.xlim(MinMax[0, 0], MinMax[0, 1]), plt.ylim(MinMax[1, 0], MinMax[1, 1]), ax.set_zlim(MinMax[2, 0], MinMax[2, 1])
    ax.set_title("De-noised data set")
    ax.view_init(0, -90)
    plt.show()


def Plot_DisVsPher(dis,pher):
    plt.scatter(dis, pher)#, s=area, c=colors, alpha=0.5)
    plt.title('distance vs. pheromone')
    plt.xlabel('dis')
    plt.ylabel('pher')
    plt.show()


def ComputeDistance(neighborhood, U):
    """
    It computes the distance of neighbors to the underlying tangent space (estimated by PCA)
    :param neighborhood: the list of neighbors
    :param U: its columns contains the eigen vector deriven by PCA
    :return: it return the distance of neighbors to the underlying tangent space
    """

    D = neighborhood.shape[1]
    M = np.mean(neighborhood, axis=0)
    Distance = LA.norm(np.dot(neighborhood - M, np.identity(D) - np.dot(U.T, U)), axis=1)[:, None].T[0]
    return Distance


def ComputeWeight(Distance, dminus1, p):
    """
    It computes the weights of a data point to its neighbors
    :param Distance: distance to the underlying tangent space (estimated by PCA)
    :param dminus1: dimensionality of manifold - 1
    :param p: the percentile of neighbors with positive weights (other neighbors get zero weights)
    :return: it return a (n,) array containing the weight of edges to the neighbors
    """

    if len(dminus1) == 1:
        a = np.percentile(Distance[:, dminus1], p)
    else:
        a = np.percentile(Distance[:, dminus1], p, axis=0)

    Weight = np.heaviside(a - Distance[:, dminus1], 0) * (1 - Distance[:, dminus1] / a)

    return Weight

def DistanceToGroundTruth(X_c, X):
    """
    computes the euclidean distance to the ground truth
    :param X_c: cleaned manifold (without noise)
    :param X: noisy manifold
    :return: distance of each point on the noisy manifold to the ground truth
    """
    if X_c.shape==X.shape :
        print("two data sets should have the same shape")

    distances = distance.cdist(X, X_c, 'euclidean')
    return [min(diss) for diss in distances]


def RandomWalk_hard(Weight, NearestNeighbor, pheromone, Option):
    """
    a random walk on a manifold with known intrinsic dimensionality
    :param Weight: a matrix containing the weight to neighbors for each data point
    :param NearestNeighbor: the list of neighbors for each samples
    :param pheromone: a vector containing the pheromone distribution
    :param Option: the parameters values for the ant colony algorithm
    :return: a vector containing the number of times that each sample is visited by the ant
    """
    Gamma = Option['Gamma']
    N_walk = Option['NumberOfSteps']
    NumOfVisit = np.zeros(len(pheromone))
    # ********* Randomly intialization of Ants
    CurrentPos = np.random.randint(len(pheromone), size=1)
    NumOfVisit[CurrentPos] = NumOfVisit[CurrentPos] + 1
    r = np.random.random(N_walk)

    for i in range(N_walk):
        Index = NearestNeighbor[CurrentPos][0]
        W = Weight[CurrentPos][0]

        H = (pheromone[Index] / np.sum(pheromone[Index])).reshape((-1, 1))
        P = np.power(W, Gamma) * np.power(H, 1 - Gamma)

        CPP = P / np.sum(P)
        CPP = np.cumsum(CPP)
        NextPoint = np.argwhere(CPP >= r[i])[0]

        CurrentPos = Index[NextPoint]
        NumOfVisit[CurrentPos] = NumOfVisit[CurrentPos] + 1

    return NumOfVisit


def RandomWalk_soft(D, Weight, NearestNeighbor, pheromone, Option):
    """
    a random walk on a manifold with unknown intrinsic dimensionality
    :param D: the dimensionality of the data set
    :param Weight: a matrix containing the weights of neighbors for each data point
    :param NearestNeighbor: the list of neighbors for each samples
    :param pheromone: a vector containing the pheromone distribution
    :param Option: the parameters values for the ant colony algorithm
    :return: a vector containing the number of times that each sample is visited by the ant
    """
    Gamma = Option['Gamma']
    N_walk = Option['NumberOfSteps']
    NumOfVisit = np.zeros(len(pheromone))

    # ********* Initialization of Ants
    CurrentPos = np.random.randint(len(pheromone), size=1)
    NumOfVisit[CurrentPos] = NumOfVisit[CurrentPos] + 1
    r = np.random.random(N_walk)

    for i in range(N_walk):
        Index = NearestNeighbor[CurrentPos][0]
        W = Weight[CurrentPos][0]
        H = pheromone[Index] / np.sum(pheromone[Index])

        P = np.power(W, Gamma) * np.power(H, 1 - Gamma)
        # print(P.shape)
        CPP = P / np.sum(P)
        CPP = np.cumsum(CPP)
        NextPoint = np.argwhere(CPP >= r[i])[0]
        CurrentPos = Index[NextPoint]
        NumOfVisit[CurrentPos] = NumOfVisit[CurrentPos] + 1

    return NumOfVisit


def RandomWalk_hard_old(Distance, NearestNeighbor, pheromone, dim, Option):
    """Distance based
    a random walk on a manifold with known intrinsic dimensionality
    :param Distance: a matrix containing the distance of neighbors for each data point
    :param NearestNeighbor: the list of neighbors for each samples
    :param pheromone: a vector containing the pheromone distribution
    :param dim: the intrinsic dimensionality of the manifold
    :param Option: the parameters values for the ant colony algorithm
    :return: a vector containing the number of times that each sample is visited by the ant
    """
    Gamma = Option['Gamma']
    N_walk = Option['NumberOfSteps']
    p = Option['a']
    NumOfVisit = np.zeros(len(pheromone))
    # ********* Randomly intialization of Ants
    CurrentPos = np.random.randint(len(pheromone), size=1)
    NumOfVisit[CurrentPos] = NumOfVisit[CurrentPos] + 1
    r = np.random.random(N_walk)

    for i in range(N_walk):
        Index = NearestNeighbor[CurrentPos][0]

        Weight = ComputeWeight(Distance[CurrentPos][0], dim-1, p)
        H = pheromone[Index] / np.sum(pheromone[Index])
        P = np.power(Weight, Gamma) * np.power(H, 1 - Gamma)

        CPP = P / np.sum(P)
        CPP = np.cumsum(CPP)
        NextPoint = np.argwhere(CPP >= r[i])[0]
        CurrentPos = Index[NextPoint]
        NumOfVisit[CurrentPos] = NumOfVisit[CurrentPos] + 1

    return NumOfVisit


def RandomWalk_soft_old(Distance, NearestNeighbor, EigVal, pheromone, Option):
    """ Distance based
    a random walk on a manifold with unknown intrinsic dimensionality
    :param Distance: a matrix containing the distance of neighbors for each data point
    :param NearestNeighbor: the list of neighbors for each samples
    :param EigVal: eigen values (deriven by PCA) for each sample's neighborhood
    :param pheromone: a vector containing the pheromone distribution
    :param Option: the parameters values for the ant colony algorithm
    :return: a vector containing the number of times that each sample is visited by the ant
    """
    Gamma = Option['Gamma']
    N_walk = Option['NumberOfSteps']
    p = Option['a']
    NumOfVisit = np.zeros(len(pheromone))
    D = Distance[0].shape[1]
    # ********* Initialization of Ants
    CurrentPos = np.random.randint(len(pheromone), size=1)
    NumOfVisit[CurrentPos] = NumOfVisit[CurrentPos] + 1
    r = np.random.random(N_walk)

    for i in range(N_walk):
        Index = NearestNeighbor[CurrentPos][0]
        N = len(Index)
        eVal = EigVal[CurrentPos][0]
        Weight = np.zeros(N)
        w1, w2 = soft_dim(eVal)
        for dminus1 in range(D):
            Weight += w1[dminus1] * ComputeWeight(Distance[CurrentPos][0], dminus1, p)
        H = pheromone[Index] / np.sum(pheromone[Index])
        P = np.power(Weight, Gamma) * np.power(H, 1 - Gamma)

        CPP = P / np.sum(P)
        CPP = np.cumsum(CPP)
        NextPoint = np.argwhere(CPP >= r[i])[0]
        CurrentPos = Index[NextPoint]
        NumOfVisit[CurrentPos] = NumOfVisit[CurrentPos] + 1

    return NumOfVisit


def WeightMatrix_hard(Data, Pheromone, Neighbors, Distances, dim, params):
    """
    it computes the weight matrix of the neighborhood graph (known the intrinsic dimensionality of the manifold)
    :param Data: the data set
    :param Pheromone: pheromone distribution
    :param Neighbors: list of neighbors for each samples
    :param Distances: distance of any sample's neighbors to the underlying tangent space
    :param dim: the intrinsic dimensionality of the manifold
    :param params: the parameters of the ant colony
    :return: the weight matrix of the neighborhood graph
    """
    Gamma = params['Gamma']
    a = params['a']
    N = Data.shape[0]
    W_new = np.zeros((N, N))
    for i in range(N):
        Index = Neighbors[i]
        pher = Pheromone[Index] / np.sum(Pheromone[Index])
        Weight = ComputeWeight(Distances[i], dim-1, a)
        W_new[i, Index] = np.power(Weight, Gamma) * np.power(pher, 1 - Gamma)
    W_new = W_new + W_new.T
    return W_new


def WeightMatrix_soft(Data, Pheromone, Neighbors, Distances, EigVal, params):
    """
    the weight matrix of the neighborhood graph (the intrinsic dimensionality of the manifold is unknown)
    :param Data: the data set
    :param Pheromone: the pheromone distribution
    :param Neighbors: the list of neighbors
    :param Distances: containing the distance of neighbors to the underlying tangent spaces
    :param EigVal: the list of eigenvalues for each sample's neighborhood
    :param params: the list of parameters in the Ant colony
    :return: it returns the weight matrix of the neighborhood graph
    """
    Gamma = params['Gamma']
    a = params['a']
    N = Data.shape[0]
    D = Data.shape[1]
    W_new = np.zeros((N, N))
    for i in range(N):
        Index = Neighbors[i]
        pher = Pheromone[Index] / np.sum(Pheromone[Index])
        eVal = EigVal[i]
        w1, w2 = soft_dim(eVal)
        Weight = np.zeros(len(Index))
        for dminus1 in range(D):
            Weight += w2[dminus1] * ComputeWeight(Distances[i], dminus1, a)
        W_new[i, Index] = np.power(Weight, Gamma) * np.power(pher, 1 - Gamma)
    W_new = W_new + W_new.T

    return W_new


def TransitionProbability(W):
    """
    it computes the transition probability matrix
    :param W: the weight matrix
    :return: transition probability matrix and its stationary distribution
    """
    Pr = W / (np.sum(W, axis=1).reshape((len(W), 1)))
    eval, evec = eigs(Pr.T, k=1)
    # w, v = LA.eig(Pr.T)
    # evec = v[:, 0]
    return Pr, evec


def Loglikelihood(X, mu, sigma, p):
    """
    computes log likelihood function of a mixture model
    :param X: data set
    :param mu: each row contain a mu vector
    :param sigma:
    :param p: the proportion of each component
    :return: the mean log likelihood
    """
    Z = np.zeros(X.shape[0])
    for i in range(len(p)):
        Z = Z + p[i] * multivariate_normal.pdf(X, mu[i], sigma[i])
    return np.mean(np.log(Z))

def ManifoldParzenWindow(X):
    """
    it fits a Gaussian distribution to a neighborhood
    :param X: neighbors
    :return: mean and covariance matrix of the gaussian distribution
    """
    mu = np.mean(X, axis=0)
    sigma = np.dot((X - mu).T, X - mu) / X.shape[0]
    return mu, sigma


def CompareClustering(Data, W_new, num_clusters):
    clustering1 = SpectralClustering(n_clusters=num_clusters, assign_labels="kmeans",
                                     affinity='precomputed', random_state=0).fit(W_new)
    clustering2 = SpectralClustering(n_clusters=num_clusters, assign_labels="kmeans",
                                     n_neighbors=20, random_state=0).fit(Data)
    l_ant = clustering1.labels_
    l_gaus = clustering2.labels_
    return l_ant, l_gaus


def Error(label, pred_label):
    r1 = np.mean(label == pred_label)
    r2 = np.mean((1 - label) == pred_label)
    return 1 - np.max([r1, r2])


def CircleSphere(r, Nc, Ns, Nn, sigma):

    t = 2 * np.pi * np.random.rand(Nc, 1)
    X1 = r * np.concatenate((np.cos(t), np.sin(t)), axis=1) + sigma * np.random.randn(Nc, 2)
    y1 = np.zeros((Nc, 1))

    X2 = np.random.randn(Ns, 2)
    y2 = np.ones((Ns, 1))

    X3 = (2 * r + 1) * np.random.rand(Nn, 2) - r - 0.5
    y3 = 1 * np.ones((Nn, 1))

    X = np.concatenate((X1, X2, X3), axis=0)
    y = np.concatenate((y1, y2, y3), axis=0)
    # X = np.concatenate((X1, X2), axis=0)
    # y = np.concatenate((y1, y2), axis=0)

    customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139']
    plt.figure(figsize=(5, 5))
    for i in range(3):
        f = (y == i).T[0]

        plt.scatter(X[f, 0], X[f, 1], c=customPalette[i])  # , s=10,c=y, cmap=viridis)  # , s=area, c=colors, alpha=0.5)
    plt.show()
    print(X.shape)
    return X, y
