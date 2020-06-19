#!/usr/bin/env python
# coding: utf-8

# ### Setup Environment
# Use validation set images from ILSVRC 2012 Challenge in a google drive folder with their labels in a meta.json. (Using validation set since no meta data available for test set)
# 
# For more information see: http://image-net.org/challenges/LSVRC/2012/

# In[ ]:


from __future__ import print_function, division
import os
import json
import io
import datetime
import collections
from skimage import io
from functools import partial
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import seaborn
import tables

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import torchvision.transforms as transforms
import torchvision.models as tmodels
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader 
from torchvision.models import *

from scipy.stats.stats import pearsonr #maybe use this instead


# In[ ]:


#########################################################################################################

# replace with own directories
imagenet_validation_path = '/mnt/raid/data/ni/dnn/ILSVRC2012_img_val'
meta_file_path = '/mnt/raid/ni/agnessa/RSA/'
ROOT_PATH = '/mnt/raid/ni/agnessa/RSA/'

#########################################################################################################


# # Select Data and get Metadata
# Select 10 images of each of the 1000 classes of the validation data set together with their label. 

# In[ ]:


class ILSVRCSubDataset(Dataset):
    """ILSVRC 2012 subset of the original val dataset"""

    def __init__(self, json_file, root, transform=None):
        """
        Args:
            json_file (string): Path to the json file with meta.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        
        # Parse META File
        with open(json_file, "r") as fd:
            self.meta = json.load(fd)
        print(self.meta)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        path = os.path.join(self.root,
                            self.meta[idx]["0"]) #merge root and the filename of the sample
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
            
        wnid = self.meta[idx]["1"]
            
        return sample, wnid #sample, class


# In[ ]:


data_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset_val = ILSVRCSubDataset(json_file=os.path.join(meta_file_path,'meta.json'),
                               root=imagenet_validation_path,
                               transform=data_transforms)

dataloaders = torch.utils.data.DataLoader(dataset_val, #Combines a dataset and a sampler, and provides an iterable over the given dataset.
                                          batch_size=20, #how many samples per batch to load
                                          shuffle=False)


# # Get model and activations
# Use the subset with 10 images of 1000 classes on torchvisions pretrained models, get the activations of specific layers and calculate the Input RDM by correlating between the activations.

# In[ ]:


def getFileName(n_samples, name):
    return name         + "_{}_".format(n_samples)         + "_{}_".format(model_name)         + "_{}".format(layer_name)          + ".npy"
#         + datetime.datetime.now().replace(microsecond=0).isoformat() \
        


# In[ ]:


def getFileNameh5(n_samples, name):
    return name         + "_{}_".format(n_samples)         + "_{}_".format(model_name)         + "_{}".format(layer_name)          + ".h5"
#         + datetime.datetime.now().replace(microsecond=0).isoformat() \
        


# In[ ]:


model_names = np.array(['resnet50'
#                       , 'resnext101_32x8d', 'resnext50_32x4d', ,resnext101_32x8d', 
#                         'vgg13', 'vgg16', 'resnet101', 'googlenet', 'alexnet',
#                         'squeezenet1_0', 'squeezenet1_1', 'mobilenet', 'vgg13_bn', 
#                         'vgg11_bn', 'resnet18', 'vgg16_bn', 'vgg19_bn', 
#                         'resnext101_32x8d'
                           ])

models = np.array([resnet50(pretrained=True),
#                  resnext101_32x8d(pretrained=True),  
#                    resnext50_32x4d(pretrained=True),resnet34(pretrained=True) , 
#                    vgg13(pretrained=True), vgg16(pretrained=True), resnet101(pretrained=True), 
#                    googlenet(pretrained=True), alexnet(pretrained=True), 
#                    squeezenet1_0(pretrained=True), squeezenet1_1(pretrained=True),
#                   mobilenet_v2(pretrained=True), vgg13_bn(pretrained=True), 
#                    vgg11_bn(pretrained=True), resnet18(pretrained=True), 
#                    vgg16_bn(pretrained=True), vgg19_bn(pretrained=True), 
#                    resnext101_32x8d(pretrained=True)
                  ])


# # Get the activations from a layer for all samples, save them

# In[ ]:


# Iterator shouldn't be recreated every time, because it always returns the first element
# Which breaks everything if shuffling is disabled

#load json file with the layers of interest (resnets)
json_file_layers=os.path.join(meta_file_path,'resnets_selected_layers.json')
with open(json_file_layers, "r") as fd:
    selected_layers = json.load(fd)
model_name, layer_name = selected_layers[16].get('model'),  selected_layers[16].get('layer') #change the index at each iteration
model = eval( model_name+'(pretrained=True)')

# #if you need to look up the index of a specific model
# for idx, dictionary in enumerate(selected_layers):
#     if dictionary.get('model') == 'resnet50':
#         print(idx)
#         break
        
NR_OF_SAMPLES = 10000 #num classes*num samples per class;  len(dataset_val)   
batch_size = 20

#important: put model in evaluation mode for consistent results
model.eval()
print('Getting activations for model->',model_name,'and layer->', layer_name)

#create an iterator for each layer
data_iterator = iter(dataloaders) 
activations = list() 
m = model
handle = m.register_forward_hook(lambda m, i, o: activations.append(list(o.data.numpy().squeeze()))) 
#arguments: model, input, output. every time that an output is computed, this hook is called and the lambda is executed

#for each batch get the activations for each batch
for i in range(int(NR_OF_SAMPLES/batch_size)): 
    print(".", end='')
    cur = next(data_iterator)[0] #cur: images, labels             
    out = model(cur) #probabilities of each class

#prepare for flattening over features
print('size activations->',np.array(activations).shape)

#flatten into num samples x num features
flattened = np.array(activations).reshape(NR_OF_SAMPLES,-1)
print(flattened.shape)

#save activations  
path = os.path.join(ROOT_PATH + 'activations/', getFileName(NR_OF_SAMPLES,"activations"))
print("Save Activation -> {}".format(path))
np.save(path, flattened)
#clear variables
handle.remove() #remove hook
del(activations)
del(data_iterator)


# In[ ]:


# # calculate INPUT RDM optimized - original

# def correlation_matrix(batch_size, file_name):

#     print("Receiving activation length")
#     g = tables.open_file(file_name, mode='r')
#     total_size, activation_length = np.array(g.root.array_c).shape
#     print("Activation Length: ", activation_length)
#     print("Total Size: ", total_size)
#     g.close()
    
#     def pearsonr_optimized(xm, ss_xm, ym, ss_ym):
# #         x = np.asarray(x)
# #         y = np.asarray(y)
# #         n = len(x)
# #         mx = x.mean()
# #         my = y.mean()
# #         xm, ym = x - mx, y - my
#       r_num = np.add.reduce(xm * ym)
#       r_den = np.sqrt(ss_xm * ss_ym)
#       r = r_num / r_den

#       # Presumably, if abs(r) > 1, then it is only some small artifact of floating
#       # point arithmetic.
#       r = max(min(r, 1.0), -1.0)

#       return r

#   #copied directly from scipy sources without change
#     def ss(a, axis=0):
#         def _chk_asarray(a, axis):
#             if axis is None:
#                 a = np.ravel(a)
#                 outaxis = 0
#             else:
#                 a = np.asarray(a)
#                 outaxis = axis

#             if a.ndim == 0:
#                 a = np.atleast_1d(a)
#             return a, outaxis

#         a, axis = _chk_asarray(a, axis)
#         return np.sum(a*a, axis)

    
#     act = np.memmap(file_name, mode="r", shape=(total_size, activation_length)) 
#     correlationd = np.zeros((total_size,total_size))
#     total = sum(x for x in range(int(total_size / batch_size))) 
#     index = 0
    
#     for i in range(int(total_size / batch_size)):
#         centered_activations_1 = np.ones((batch_size,activation_length)) * -17
#         centered_squared_summed_activations_1 = np.ones((batch_size,)) * -17
        
#         start_1 = batch_size*i
#         end_1 = batch_size*(i+1)
        
#         list_of_activations_1 = act[start_1:end_1,:]
        
#         for j in range(int(total_size / batch_size)-i):
#             index += 1
#             print("New Iteration: i = {0}, j = {1}; {2}/{3}".format(i,j,index,total))
            
#             start_2 = batch_size*(i+j)
#             end_2 = batch_size*(i+j+1)
            
#             list_of_activations_2 = act[start_2:end_2,:]
            
#             centered_activations_2 = np.ones((batch_size,activation_length)) * -17
#             centered_squared_summed_activations_2 = np.ones((batch_size,)) * -17
            
            
#             for k in range(batch_size):
#                 if k % 200 == 0:
#                     print("Centering... done {0} of {1}".format(k, batch_size))
#                 centered_activations_1[k] = list_of_activations_1[k] - list_of_activations_1[k].mean()
#                 centered_squared_summed_activations_1[k] = ss(centered_activations_1[k])
                
#                 centered_activations_2[k] = list_of_activations_2[k] - list_of_activations_2[k].mean()
#                 centered_squared_summed_activations_2[k] = ss(centered_activations_2[k])
            
            
#             for l in range(batch_size):
#                 if l % 200 == 0:
#                     print("Correlation... done {0} of {1}".format(l, batch_size))
#                 for m in range(batch_size):
#                     correlationd[start_1+l,start_2+m] = correlationd[start_2+m, start_1+l] = 1 - pearsonr_optimized(centered_activations_1[l], centered_squared_summed_activations_1[l], centered_activations_2[m], centered_squared_summed_activations_2[m])

#     return(correlationd)


# # Calculating the correlations for each layer and creating input RDMs

# In[ ]:


# # calculate INPUT RDM optimized

# def correlationd_matrix(batch_size,array_activations): #(list_of_activations, n)
#     # turn into h5py
# #     flattened = np.array(activations).reshape(NR_OF_SAMPLES,-1) #if needed
# #     print("Save into .h5 file")
# #     file_name = os.path.join(ROOT_PATH+'activations/',getFileNameh5(NR_OF_SAMPLES,'activations'))
# #     h5f = h5py.File(file_name, 'w')
# #     h5f.create_dataset('flattened', data=flattened)
# #     h5f.close()
#     print("Receiving activation length")
#     g = tables.open_file(file_name, mode='r') #create tables
#     total_size, activation_length = np.array(g.root.flattened).shape
#     print("Activation Length: ", activation_length)
#     print("Total Size: ", total_size)
#     g.close()
    
#     def pearsonr_optimized(xm, ss_xm, ym, ss_ym):
# #         x = np.asarray(x)
# #         y = np.asarray(y)
# #         n = len(x)
# #         mx = x.mean()
# #         my = y.mean()
# #         xm, ym = x - mx, y - my
#         r_num = np.add.reduce(xm * ym)
#         r_den = np.sqrt(ss_xm * ss_ym)
#         r = r_num / r_den

#         # Presumably, if abs(r) > 1, then it is only some small artifact of floating
#         # point arithmetic.
#         r = max(min(r, 1.0), -1.0)

#         return r

#     #copied directly from scipy sources without change 
#     def ss(a, axis=0):
#         def _chk_asarray(a, axis):
#             if axis is None:
#                 a = np.ravel(a)
#                 outaxis = 0
#             else:
#                 a = np.asarray(a)
#                 outaxis = axis

#             if a.ndim == 0:
#                 a = np.atleast_1d(a)

#             return a, outaxis

#         a, axis = _chk_asarray(a, axis)
#         return np.sum(a*a, axis)

#     #memmap is used to access a part of the file. maybe: use memmap in the loop to access the first 1000 samples?    
#     act = np.memmap(file_name, mode="r", shape=(total_size, activation_length)) #numsamples x numfeatures
#     correlationd = np.zeros((total_size,total_size))
#     correlationd[:] = np.nan
#     total = sum(x for x in range(int(total_size / batch_size))) #num 1000-wise comparisons to do: 45
#     index = 0
    
# #     # pre-allocate necessary values for the correlation
# #     centered_activations = np.ones(list_of_activations.shape) #num samples x num features
# #     centered_activations[:] = np.nan

# #     centered_squared_summed_activations = np.ones((list_of_activations.shape[0],)) #num samples x 1
# #     centered_squared_summed_activations[:] = np.nan
    
# #     #obtain the squared sum of activations
# #     for i in range(n):      
# #         centered_activations[i] = list_of_activations[i] - list_of_activations[i].mean()    
# #         centered_squared_summed_activations[i] = ss(centered_activations[i]) 
    
# #     #pre-allocate the input rdm matrix
# #     correlationd = np.empty((n,n))
# #     correlationd[:] = np.nan
    
# #     #correlate between each pair of samples
# #     for i in range(n):
# #         for j in range(i + 1, n):
# #             correlationd[i,j] = correlationd[j, i] = 1 - pearsonr_optimized(centered_activations[i], centered_squared_summed_activations[i], centered_activations[j], centered_squared_summed_activations[j])
      
# #     return(correlationd)

   
#     for i in range(int(total_size / batch_size)):
       
#         start_1 = batch_size*i
#         end_1 = batch_size*(i+1)
        
#         list_of_activations_1 = act[start_1:end_1,:]
        
#         for j in range(int(total_size / batch_size)-i):
#             index += 1
#             print("New Iteration: i = {0}, j = {1}; {2}/{3}".format(i,j,index,total))
            
#             start_2 = batch_size*(i+j)
#             end_2 = batch_size*(i+j+1)
            
#             list_of_activations_2 = act[start_2:end_2,:]
#             correlationd[start_1:end_1,start_2:end_2] = 1-np.corrcoef([list_of_activations_1;list_of_activations_2])
            
#     return(correlationd)


# In[ ]:


# import numpy as np
# import os
# import json

# #load json file with the layers of interest (resnets)
# print('Loading the json file')
# ROOT_PATH = '/mnt/raid/ni/agnessa/RSA/' 
# json_file_layers=os.path.join(ROOT_PATH,'resnets_selected_layers.json')
# with open(json_file_layers, "r") as fd:
#     selected_layers = json.load(fd)
# model_name, layer_name = selected_layers[0].get('model'),  selected_layers[0].get('layer') #change the index at each iteration     
# NR_OF_SAMPLES = 10000 #num classes*num samples per class;


# #load activations
# path = os.path.join(ROOT_PATH + 'activations/', getFileName(NR_OF_SAMPLES, "activations"))

# # save np.load
# np_load_old = np.load

# # modify the default parameters of np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# print('Loading activations for model ->',model_name,',layer ->',layer_name)
# flattened = np.load(path)
# np.load = np_load_old

# #correlation matrix = input RDM
# corr_matrix = correlationd_matrix(1000,flattened) 
# path = os.path.join(ROOT_PATH + 'Input_RDM/', getFileName(NR_OF_SAMPLES, "Input_RDM"))
# print("Save Input RDM ->len {}".format(path))
# np.save(path, np.array(input_rdm)) 
# print('calculating correlations for model ->',model_name, 'and layer ->', layer_name)
# print('flattened shape ->',flattened.shape)
# # input_rdm = 1-np.corrcoef(flattened) #check if this is optimal - maybe instead you can do:
# #input_rdm=[]
# #for sample,i in flattened 
# #for sample+1,j in flattened
# #inputrdm[i,j] = inputrdm[j,i] = np.corrcoef(i,j)


# # In case you need to run only one layer

# In[ ]:


# #for layer 1

# # Iterator shouldn't be recreated every time, because it always returns the first element
# # Which breaks everything if shuffling is disabled
# # data_iterator = iter(dataloaders)
# model_name = 'resnet34'
# layer_name = 'layer1.0'
# NR_OF_SAMPLES = 10000 #num classes*num samples/class;  len(dataset_val) 
# batch_size = 20

# path = os.path.join(ROOT_PATH + 'activations/', getFileName(NR_OF_SAMPLES,"activations"))
# # save np.load
# np_load_old = np.load

# # modify the default parameters of np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# activations = np.load(path)
# np.load = np_load_old

# #correlation matrix = input RDM
# y = activations.shape[2]
# x = activations.shape[3]
# w = activations.shape[4]

# print('size activations->',np.array(activations).shape)
# print('y,x,w->',y,x,w)

# flattened = np.array(activations).reshape(NR_OF_SAMPLES,y*x*w)
# corr_matrix = correlationd_matrix(flattened,NR_OF_SAMPLES) 
# path = os.path.join(ROOT_PATH + 'Input_RDM/', getFileName(NR_OF_SAMPLES, "Input_RDM"))
# print("Save Input RDM ->len {}".format(path))
# np.save(path, np.array(corr_matrix))

