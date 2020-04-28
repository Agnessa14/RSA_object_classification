#!/usr/bin/env python
# coding: utf-8

# # Set up the environment

# In[ ]:


import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn
import os
import pickle
import json
from scipy.stats import spearmanr


# # Define a function to create a filename

# In[ ]:


def getFileName(name, n_samples, model_name, layer_name):
    return name         + "_{}_".format(n_samples)         + "_{}_".format(model_name)         + "_{}".format(layer_name)          + ".npy"   


# # Define a function to create the upper triangular of Input RDMs

# In[ ]:


def get_upper_triangular(rdm):
    num_conditions = rdm.shape[0] #num samples
    return rdm[np.triu_indices(num_conditions,1)] #take all above the main diagonal (excluding it), returns flattened version


# # Load the model(s) and layers

# In[ ]:


multiple_models = 1 #comparing within a model or between models; 1 = between, 0 = within

#load the np file containing the shape of the activations
ROOT_PATH = '/mnt/raid/ni/agnessa/RSA/'
NR_OF_SAMPLES = 10000
json_file_layers=os.path.join(ROOT_PATH,'resnets_selected_layers.json')
with open(json_file_layers, "r") as fd:
    selected_layers = json.load(fd)

#get the name of the model(s) and of the layers
num_layers_1 = 16
model_begin_1 = 0 #index of the first layer of the desired model
model_name_1 = selected_layers[model_begin_1].get('model') 
layer_names_1 = []

if multiple_models == 1:
    num_layers_2 = 16
    model_begin_2 = 16
    model_name_2 = selected_layers[model_begin_2].get('model')
    layer_names_2 = []


#append the layers from the first model
for i in range(num_layers_1):
    layer_names_1.append(selected_layers[model_begin_1+i].get('layer'))   
    
if multiple_models == 1:
    #append the layers from the second model
    for j in range(num_layers_2):
        layer_names_2.append(selected_layers[model_begin_2+j].get('layer'))

if multiple_models == 1:
    print('Comparing', model_name_1, ', layers', layer_names_1, 'and', model_name_2, ', layers', layer_names_2)
else:
    print('Comparing models', model_name_1, 'and', model_name_1, ', layers', layer_names_1)


# # Create model RDMs by correlating between Input RDMs from different layers and models

# In[ ]:


size_rdm = np.array(layer_names_1).shape[0]   
RSA_matrix = np.ones((size_rdm,size_rdm)) #num layers x num layers
RSA_matrix[:] = np.nan

if multiple_models == 1:
    model_name = model_name_1 + '_' + model_name_2
else:
    model_name = model_name_2 = model_name_1  
    
#1. get upper triangulars
#2. calculate the correlation distance (1-Spearman's coefficient) between the upper triangulars
#3. repeat for all pairs of layers (and models)

for layer_i in layer_names_1:
#     if np.where(np.array(layer_names)==layer_i)[0][0] < num_layers_1+model_begin_1:
#         model_name_i = model_name_1
#     elif num_layers_1+model_begin_1 < np.where(np.array(layer_names)==layer_i)[0][0] < num_layers_2+model_begin+2:
#         model_name_i = model_name_2
    
    ## load RDMs ##
    RDM_PATH_i = os.path.join(ROOT_PATH, 'Input_RDM/' + getFileName('Input_RDM', NR_OF_SAMPLES, model_name_1, layer_i))
    print(RDM_PATH_i)
    Input_RDM_i = np.load(RDM_PATH_i)
    ## get upper triangulars, without the 0 diagonal
    print('Getting the upper triangular of ->', layer_i)
    ut_rdm_i = get_upper_triangular(Input_RDM_i)

    
    for layer_j in layer_names_2: #layer_names[np.where(np.array(layer_names)==layer_i)[0][0]:len(layer_names)]:
#         print(layer_names[np.where(layer_i)[0][0]:len(layer_names)])
#         if np.where(np.array(layer_names)==layer_j)[0][0] < num_layers_1+model_begin_1:
#             model_name_j = model_name_1
#         elif num_layers_1+model_begin_1 < np.where(np.array(layer_names)==layer_j)[0][0] < num_layers_2+model_begin+2:
#             model_name_j = model_name_2
        
        RDM_PATH_j = os.path.join(ROOT_PATH, 'Input_RDM/' + getFileName('Input_RDM', NR_OF_SAMPLES, model_name_2, layer_j))        
        print(RDM_PATH_j)
        Input_RDM_j = np.load(RDM_PATH_j)
        print('Getting the upper triangular of ->', layer_j)
        ut_rdm_j = get_upper_triangular(Input_RDM_j)
            
        # Spearman correlation
        print('Calculating the correlation distance between ->', layer_i, 'and', layer_j)
        RSA_i_j = 1-spearmanr(ut_rdm_i,ut_rdm_j)[0]
        print('Finished the correlation distance between ->', layer_i, 'and', layer_j)
        print(RSA_i_j)
        # Save into a matrix 
        print('Saving the correlation distance between ->', layer_i, 'and', layer_j)
        RSA_matrix[np.where(np.array(layer_names_1)==layer_i)[0][0],np.where(np.array(layer_names_2)==layer_j)[0][0]] = RSA_i_j
#         RSA_matrix[np.where(np.array(layer_names)==layer_j)[0][0],np.where(np.array(layer_names)==layer_i)[0][0]] = 

# save model RDM          
path = os.path.join(ROOT_PATH + 'Model_RDM/', getFileName('Model_RDM', NR_OF_SAMPLES, model_name, 'all'))
np.save(path,RSA_matrix)

        


# In[ ]:




