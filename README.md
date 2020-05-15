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

## 0. Create a file containing the meta-data for the images.
*Script*: **tbd**

First, we need to create a dictionary file (e.g., in .json format) which will have the information of the images we want to feed to our pre-trained DNN. The information is twofold:
for each image, we have its 1) image filename 2) image class. The script to create the meta file will follow soon. The file for the current analysis is **meta.json**.
It has the information for the 10 000 images (10 per class for 1000 classes) that we will feed to a pre-trained DNN.

## 1. Select the desired DNN layers.
*Script*: **models_and_layers.ipynb**
*File*: **all_models_all_layers.json** & **resnets_selected_layers**
This script takes the names of the layers and models we want to include in the analysis. It and creates two .json dictionary files: 1) a file that contains the names of all models and layers, in case we need to reference it 2) a file that contains the layers we want to analyze.
In this case, we created the file **resnets_selected_layers** for working with ResNets. 

## 2. Collect activations from multiple layers of a DNN. 
*Script*: **Input_RDM_Activations.ipynb**

Once we have the metadata file, we have to collect activations from the DNN in order to create Input RDMs. In this script, we first transform the images listed in
the meta file into the appropriate format. Then, we feed them to our DNN and collect activations from every layer specified in the **resnets_selected_layes.json** file.
We save them in an array of size "number of samples x number of features". The script runs for one layer at a time: for the purposes of not taking up too much memory,
we are not looping over all layers, but instead, clear the notebook after each layer, and manually change the indices to the next one in the line

model_name, layer_name = selected_layers[57].get('model'),  selected_layers[57].get('layer'). 


## 3. Create Input RDMs for each of the selected layers of the DNN.  
## 4. Create the Model RDM for the DNN. 
## 5. Run MDS for the DNN.
## 6. Create a Model RDM for all DNNs at once.
## 7. Run MDS using all layers from all DNNs. 
