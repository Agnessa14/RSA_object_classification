#!/usr/bin/env python
# coding: utf-8

# # Setup Environment
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
meta_file_path = '/mnt/antares_raid/home/agnessa/RSA/'
ROOT_PATH = '/mnt/antares_raid/home/agnessa/RSA/' 

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

# In[1]:


def getFileName(n_samples, name):
    return name         + "_{}_".format(n_samples)         + "_{}_".format(model_name)         + "_{}".format(layer_name)          + ".npy"
#         + datetime.datetime.now().replace(microsecond=0).isoformat() \
        


# In[ ]:


model_names = np.array(['resnext101_32x8d'
#                       , 'resnet50', 'resnext50_32x4d','resnext101_32x8d', 
#                         'vgg13', 'vgg16', 'resnet101', 'googlenet', 'alexnet',
#                         'squeezenet1_0', 'squeezenet1_1', 'mobilenet', 'vgg13_bn', 
#                         'vgg11_bn', 'resnet18', 'vgg16_bn', 'vgg19_bn', 
#                         'resnext101_32x8d'
                           ])

models = np.array([resnext101_32x8d(pretrained=True)
#                    , resnet50(pretrained=True), 
#                    resnext50_32x4d(pretrained=True),resnet34(pretrained=True) , 
#                    vgg13(pretrained=True), vgg16(pretrained=True), resnet101(pretrained=True), 
#                    googlenet(pretrained=True), alexnet(pretrained=True), 
#                    squeezenet1_0(pretrained=True), squeezenet1_1(pretrained=True),
#                   mobilenet_v2(pretrained=True), vgg13_bn(pretrained=True), 
#                    vgg11_bn(pretrained=True), resnet18(pretrained=True), 
#                    vgg16_bn(pretrained=True), vgg19_bn(pretrained=True), 
#                    resnext101_32x8d(pretrained=True)
                  ])


# In[ ]:


# calculate INPUT RDM optimized

def correlationd_matrix(list_of_activations, n):
    def pearsonr_optimized(xm, ss_xm, ym, ss_ym):
#         x = np.asarray(x)
#         y = np.asarray(y)
#         n = len(x)
#         mx = x.mean()
#         my = y.mean()
#         xm, ym = x - mx, y - my
        r_num = np.add.reduce(xm * ym)
        r_den = np.sqrt(ss_xm * ss_ym)
        r = r_num / r_den

        # Presumably, if abs(r) > 1, then it is only some small artifact of floating
        # point arithmetic.
        r = max(min(r, 1.0), -1.0)

        return r

    #copied directly from scipy sources without change 
    def ss(a, axis=0):
        def _chk_asarray(a, axis):
            if axis is None:
                a = np.ravel(a)
                outaxis = 0
            else:
                a = np.asarray(a)
                outaxis = axis

            if a.ndim == 0:
                a = np.atleast_1d(a)

            return a, outaxis

        a, axis = _chk_asarray(a, axis)
        return np.sum(a*a, axis)

    # pre-calculate necessary values
    centered_activations = np.ones(list_of_activations.shape) 
    centered_activations[:] = np.nan

    centered_squared_summed_activations = np.ones((list_of_activations.shape[0],))#nr of samples
    centered_squared_summed_activations[:] = np.nan
    
    for i in range(n): #for each sample      
        centered_activations[i] = list_of_activations[i] - list_of_activations[i].mean()    
        centered_squared_summed_activations[i] = ss(centered_activations[i]) 
        correlationd = np.empty((n,n))
        correlationd[:] = np.nan
    
    for i in range(n):
        for j in range(i + 1, n):
            correlationd[i,j] = correlationd[j, i] = 1 - pearsonr_optimized(centered_activations[i], centered_squared_summed_activations[i], centered_activations[j], centered_squared_summed_activations[j])
      
    return(correlationd)



# In[ ]:


# calculate Activations of specific layer of models (and store, if wanted) and calculate Input_RDM and save

## for debugging ##:
#del(activations)
#handle.remove()
#del(data_iterator)
            
# Iterator shouldn't be recreated every time, because it always returns the first element
# Which breaks everything if shuffling is disabled. However, it should be restarted at every

NR_OF_SAMPLES = 10000 #num classes*num samples/class;  len(dataset_val)
SAVE_PATH = ROOT_PATH   
batch_size = 20

for model, model_name in zip(models, model_names):
    print(model_name)
    #important: put model in evaluation mode for consistent results
    model.eval()

    #for each layer
    for layer_name, m in model.named_modules(): #name of layer (name), what happens in the layer (m) 
        if (layer_name != 'layer1.0') & (layer_name != 'layer1.1') & (layer_name != 'layer1.2'): 
            data_iterator = iter(dataloaders)
            activations = list()
            dots = 0
            for char in layer_name: 
                if char == '.':
                    dots = dots+1

            if dots == 1: #different if statements for diff models/layers      
                print("register layer -> ", layer_name)
                handle = m.register_forward_hook(
                lambda m, i, o: activations.append(list(o.data.numpy().squeeze())) #arguments: model, input, output. every time that 
                )                                        #an output is computed, this hook is called and the lambda is executed

                for i in range(int(NR_OF_SAMPLES/batch_size)): #for each batch; 
                #add a sample loop?
                    print(".", end='')
                    cur = next(data_iterator)[0] #cur: images, labels             
                    out = model(cur) #The outputs are energies for the 10 classes. The higher the energy for a class, 
                                     #the more the network thinks that the image is of the particular class. 
                handle.remove() #remove hook
                print('size activations->',np.array(activations).shape)
                y = np.array(activations).shape[2]
                x = np.array(activations).shape[3]
                w = np.array(activations).shape[4]

                print('size activations->',np.array(activations).shape)
                print('y,x,w->',y,x,w)

                flattened = np.array(activations).reshape(NR_OF_SAMPLES,y*x*w)
                print(flattened.shape)

                # to save activations additionally 
                path = os.path.join(SAVE_PATH + 'activations/', getFileName(NR_OF_SAMPLES,"activations"))
                print("Save Activation -> {}".format(path))
                np.save(path, np.array(activations))

                #correlation matrix = input RDM
                corr_matrix = correlationd_matrix(flattened,NR_OF_SAMPLES) 
                path = os.path.join(SAVE_PATH + 'Input_RDM/', getFileName(NR_OF_SAMPLES, "Input_RDM"))
                print("Save Input RDM ->len {}".format(path))
                np.save(path, np.array(corr_matrix))
                
                del(activations)
                del(data_iterator)


# In[ ]:


# #for layer 1

# # Iterator shouldn't be recreated every time, because it always returns the first element
# # Which breaks everything if shuffling is disabled
# # data_iterator = iter(dataloaders)
# model_name = 'resnet34'
# layer_name = 'layer1.0'
# NR_OF_SAMPLES = 10000 #num classes*num samples/class;  len(dataset_val)
# SAVE_PATH = ROOT_PATH   
# batch_size = 20

# path = os.path.join(SAVE_PATH + 'activations/', getFileName(NR_OF_SAMPLES,"activations"))
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
# path = os.path.join(SAVE_PATH + 'Input_RDM/', getFileName(NR_OF_SAMPLES, "Input_RDM"))
# print("Save Input RDM ->len {}".format(path))
# np.save(path, np.array(corr_matrix))


# In[5]:


# # after creating the input rdm
# import os
# import numpy as np
# model_name = 'resnet34'
# layer_name = 'layer4.2'
# SAVE_PATH = '/mnt/antares_raid/home/agnessa/RSA/' 
# NR_OF_SAMPLES = 10000
# path = os.path.join(SAVE_PATH + 'Input_RDM/', getFileName(NR_OF_SAMPLES, "Input_RDM"))
# # save np.load
# np_load_old = np.load

# # modify the default parameters of np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# Input_RDM = np.load(path)
# np.load = np_load_old


# In[8]:


# import matplotlib.pyplot as plt
# import seaborn
# Input_RDM[np.isnan(Input_RDM)]=0.0
# fig = plt.figure(figsize=(15,15))
# ax = seaborn.heatmap(Input_RDM, cmap='rainbow', vmin=0.8, vmax=1.0)


# In[9]:


# # output_name = [getFileName(NR_OF_SAMPLES, "Input_RDM_plot"), '.png']
# path = os.path.join(SAVE_PATH + 'Input_RDM_plots', getFileName(NR_OF_SAMPLES, "Input_RDM") + '.png')
# fig.savefig(path)


# In[ ]:





# In[ ]:




