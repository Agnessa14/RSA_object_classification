# RSA: object classification
This code is for running a representational similarity analysis (RSA) on the activations of different deep neural networks (DNNs).
The goal is to investigate whether similarity in the internal representations of various DNNs is related to similarity in performance.
For this, we need to perform the analysis for each DNN in the following order: 
  
  0. Create a file containing the meta-data for the images we would like to feed to our pre-trained DNN.
  1. Select the layers we want to include in the analysis. 
  2. Collect activations from multiple layers of a DNN. 
  3. For each layer, create a matrix, known as "Input representational dissimilarity matrix (RDM)", containing the Pearson correlation distances of
  the activations for all pairwise combinations of samples 
  4. Create a matrix, known as "Model RDM", containing the Spearman correlation distances of all pairwise combinations of layers. 
  5. Run the multidimensional scaling (MDS) analysis to reduce dimensionality & visualize the similarity in representational structure 
  between different layers. 
  
Once this has been done on multiple DNNs, we can: 

  6. Create a Model RDM for all DNNs at once.
  6. Run MDS using all layers from all DNNs. 
  
Below are the detailed descriptions of the steps and the code that they refer to.

## 0. Create a file containing the meta-data for the images
*Script*: **tbd**

First, we need to create a dictionary file (e.g., in .json format) which will have the information of the images we want to feed to our pre-trained DNN. The information is twofold:
for each image, we have its 1) image filename 2) image class. The script to create the meta file will follow soon. The file for the current analysis is [**meta.json**](/meta.json).
It has the information for the 10 000 images (10 per class for 1000 classes) that we will feed to a pre-trained DNN.

## 1. Select the desired DNN layers
*Script*: [**models_and_layers.ipynb**](/models_and_layers.ipynb)

*File*: [**all_models_all_layers.json**](/all_models_all_layers.json) &  [**resnets_selected_layers**](/resnets_selected_layers.json)

This script takes the names of the layers and models we want to include in the analysis. It and creates two .json dictionary files: 1) a file that contains the names of all models and layers, in case we need to reference it 2) a file that contains the layers we want to analyze.
In this case, we created the file **resnets_selected_layers** for working with ResNets. 

## 2. Collect activations from multiple layers of a DNN 
*Script*: [**Input_RDM_Activations.ipynb**](/Input_RDM_Activations.ipynb)
 

Once we have the metadata file, we have to collect activations from the DNN in order to create Input RDMs. In this script, we first transform the images listed in the meta file into the appropriate format. Then, we feed them to our DNN and collect activations from every layer specified in the [**resnets_selected_layers**](/resnets_selected_layers.json) file.
We save them in an array of size "number of samples x number of features". The script runs for one layer at a time: for the purposes of not taking up too much memory, we are not looping over all layers but instead clear the notebook after each layer and manually change the indices to the next one in the line

model_name, layer_name = selected_layers[57].get('model'),  selected_layers[57].get('layer'). 

The indices should be the same, as they refer to one model-layer combo. 

To get the activations from the specified layer, we register a hook on it and append the activations for all images into one array. Once we have looped over all samples, we transform the array into that of a size "number of samples x number of features", and save it.

After running this script for one layer of one model, make sure to clear the output before running another job.

## 3. Create Input RDMs for each of the selected layers of the DNN 
*Script*: [**Input_RDM_Correlations.ipynb**](/Input_RDM_Correlations.ipynb)

After getting the activations from a model's layer, we load them by changing the index in the following line

model_name, layer_name = selected_layers[57].get('model'),  selected_layers[57].get('layer'),

similarly to [**Input_RDM_Activations.ipynb**](/Input_RDM_Activations.ipynb). We compute the correlation distances (1-Pearson correlation) between the activations of two samples, for all pairwise combinations of samples. We need to do this in batches of 1000 x 1000 samples, otherwise we will take up too much CPU. 

The correlations are stored in a 10 000 x 10 000 matrix called "Input RDM". The Input RDM is then plotted as a heatmap and saved. 

After creating the input RDM for one layer of one model, make sure to clean the output before continuing with the analysis.

## 4. Create the Model RDM for the DNN 
*Script*: [**Creating_Model_RDMs.ipynb**](/Creating_Model_RDMs.ipynb)

Once we obtained Input RDMs for all desired layers of a model, we can correlate (1-Spearman correlation) the Input RDMs of all desired layers, resulting in a Model RDM, a "number of layers x number of layers" matrix. For that, we need to get the upper triangular part of each input RDM (since the input RDM is symmetrical about its diagonal), and compute the correlations of all pairwise combinations of layers. 

## 5. Run MDS for the DNN.
*Script*: [**Plotting_Model_RDM.ipynb**](/Plotting_Model_RDM.ipynb)

Once we created a Model RDM, we can first plot it as a heatmap. We can also run the multidimensional scaling analysis in order to more clearly visualize the similarities between different layers.

## 6. Create a Model RDM for all DNNs at once.
*Script*: [**Creating_Model_RDMs.ipynb**](/Creating_Model_RDMs.ipynb) 

After running steps 0-5 for one model, we can correlate between the representations of different models by creating a model RDM for multiple models. To that end, we can create a "number of layers in model 1 x number of layers in model 2" matrix containing the correlations for the layers of one model with the layers of the other model.  To create the matrix of the size "number of all layers x number of all layers", we can simply put together the individual Model RDMs and the inter-model Model RDM together in a symmetric matrix (making sure to transpose inter-model RDMs wherever needed). We have to add the details of the desired network here:

if multiple_models == 1:
    num_layers_2 = 16 #change depending on the model
    model_begin_2 = 0
    model_name_2 = selected_layers[model_begin_2].get('model')
    layer_names_2 = []


## 7. Run MDS using all layers from all DNNs. 
*Script*: [**Plotting_Model_RDM.ipynb**](/Plotting_Model_RDM.ipynb)

Similarly to individual Model RDMs, we can plot the inter-model Model RDM as a heatmap. We can also run the MDS analysis on all analyzed layers from all models to create one plot illustrating the similarities between different layers of different models. 
