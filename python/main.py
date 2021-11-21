import numpy as np
from Algorithms import *
import time
from sklearn import manifold, datasets

params = dict()
params['NumberOfAnts'] = 10
params['NumberOfIteration'] = 5    # number of times that an ant randomly starts walking
params['NumberOfSteps'] = 10000     # number of steps that an ant walks in each iteration
params['Gamma'] = 0.9               # balance between distances and directions (PCA)
params['EvapRate'] = 0.1
params['PherVsEvapPerVisit'] = 2
params['a'] = 50                    # how far they can jump
params['radius'] = 0.7
params['k'] = 20
params['d'] = 2
params['Top'] = 20                  # % percent of points with the highest pheromone
saving = 1
dim = 2

filename = 'synth'
Method = 'soft'

# print('********************************')
# print('*       Loading data set       *')
# print('********************************\n')
Data_c, labels = datasets.make_s_curve(6000, 0, random_state=None)
Data = Data_c + 0.2 * np.random.randn(Data_c.shape[0], Data_c.shape[1])
# #
# #
# # print('***********************************')
# # print('*    Pre-processing is started    *')
# # print('***********************************')
MyPreprocess(Data, labels, params['radius'], params['k'], filename+'_pre', params['a'], dim)


print('*********************************************')
print('*    Performing the Ant Colony Algorithm    *')
print('*********************************************')
f = np.load(filename+'_pre.npz', allow_pickle=True)
start = time.perf_counter()

if Method == 'hard':
    ants(f, params, filename+'_'+Method, saving)
else:
    ants_soft(f, params, filename+'_'+Method, saving)

finish = time.perf_counter()
print(f'finished in {round(finish-start,2)} second(s)')


f1 = np.load(filename+'_pre.npz', allow_pickle=True)
f2 = np.load(filename + '_' + Method + '.npz', allow_pickle=True)

Data = f1['Data']
Neighbors = f1['idx']
# Distance = f1['Distance']
# eVal = f1['eVal']
# labels = f1['labels']
Pheromone = f2['Pheromone']

# disVec = DistanceToGroundTruth(Data, Data_c)

Plots(Data, 'g', Pheromone, 100-params['Top'], params['NumberOfIteration'])
# Plot_DisVsPher(disVec, Pheromone)
