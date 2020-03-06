#!/usr/bin/env python
# coding: utf-8

# # Set up the environment

# In[ ]:


import torchvision
import torch
import numpy as np
import scipy.stats
import sklearn.manifold
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn
import pandas as pd
import sklearn
import os
import pickle
import time
from scipy.spatial import distance
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


# # Define a function to create a filename

# In[ ]:


def getFileName(name, n_samples, model_name, layer_name):
    return name         + "_{}_".format(n_samples)         + "_{}_".format(model_name)         + "_{}".format(layer_name)          + ".npy"   


# # Define a function to create the upper triangular of Input RDMs

# In[ ]:


def get_upper_triangular(rdm):
    num_conditions = rdm.shape[0] #num samples
    return rdm[np.triu_indices(num_conditions,1)] #take all above the main diagonal (excluding it), returns flattened version


# # Create model RDMs by correlating between Input RDMs from different layers and models

# In[ ]:


model_name = 'resnet34' #change to model_names when running for multiple models
layer_names_1 = np.array(['layer1.0','layer1.1','layer1.2','layer2.0','layer2.1','layer2.2','layer2.3','layer3.0','layer3.1',
              'layer3.2','layer3.3','layer3.4','layer3.5','layer4.0','layer4.1','layer4.2'])
#layer_names_2 = ... - for multiple models, you may need multiple layer names arrays
NR_OF_SAMPLES = 10000;
ROOT_PATH = '/mnt/antares_raid/home/agnessa/RSA/' 
RSA_matrix = np.ones((layer_names_1.shape[0],layer_names_1.shape[0])) #num layers x num layers
RSA_matrix[:] = np.nan

#1. get upper triangulars
#2. correlate (spearman) between the upper triangulars
#3. repeat for all pairs of layers (and models)

#for multiple models: replace by model & layer loops
for layer_i in layer_names_1:
    for layer_j in layer_names_1:
    
        ## load RDMs ##
        RDM_PATH_i = os.path.join(ROOT_PATH, 'Input_RDM/' + getFileName('Input_RDM', NR_OF_SAMPLES, model_name, layer_i))
        RDM_PATH_j = os.path.join(ROOT_PATH, 'Input_RDM/' + getFileName('Input_RDM', NR_OF_SAMPLES, model_name, layer_j))        
        np_load_old = np.load # save np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k) # modify the default parameters of np.load: the function is acting weird otherwise
        Input_RDM_i = np.load(RDM_PATH_i)
        Input_RDM_j = np.load(RDM_PATH_j)
        np.load = np_load_old #revert back to old function
    
        ## get upper triangulars, without the 0 diagonal
        print('Getting the upper triangular of ->', layer_i)
        ut_rdm_i = get_upper_triangular(Input_RDM_i)
        print('Getting the upper triangular of ->', layer_j)
        ut_rdm_j = get_upper_triangular(Input_RDM_j)
            
        # Spearman correlation
        print('Calculating the correlation between ->', layer_i, 'and', layer_j)
        RSA_i_j = spearmanr(ut_rdm_i,ut_rdm_j)[0]
        print('Finished the correlation between ->', layer_i, 'and', layer_j)
        
        # Save into a matrix 
        print('Saving the correlation between ->', layer_i, 'and', layer_j)
        RSA_matrix[np.where(layer_names_1==layer_i)[0][0],np.where(layer_names_1==layer_j)[0][0]] = RSA_i_j
        
#save model RDM
path = os.path.join(ROOT_PATH + 'Model_RDM/', getFileName('Model_RDM', NR_OF_SAMPLES, model_name, 'all'))
np.save(path,RSA_matrix)

        

