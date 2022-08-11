##### In the following workflow we create a double bifurcating dataset from Dyngen, a single cell RNA sequencing simulation data software. (https://dyngen.dynverse.org/index.html)
# We then apply imputation and feature selection software to the data to compare their performance. The following code is a mixture of R script and Python script.


####################

# Create the Dyngen dataset with Dyngen. This section is to be run in the R software enviroment.

library(tidyverse)
library(dyngen)

#

backbone <- bblego(
  bblego_start("A", type = "doublerep2", num_modules = 4),
  bblego_linear("A", "B", type = "doublerep2", num_modules = 10),
  bblego_branching("B", c("C", "D"), type = "simple",num_modules = 15),
  bblego_end("C", type = "doublerep2", num_modules = 10),
  bblego_end("D", type = "doublerep2", num_modules = 10)
)

config <-
  initialise_model(
    backbone = backbone,
    num_tfs = 750,
    num_cells = 2000,
    num_targets = 0,
    num_hks = 0,
    verbose = FALSE,
    simulation_params = simulation_default(ssa_algorithm = ssa_etl(tau = 10 / 3600),census_interval = 2)
)

# Run the model and create the dataset. Un-comment any of the plotting fucntions to look at the underlying model that creates the data.
#plot_backbone_statenet(config)
#plot_backbone_modulenet(config)
model <- generate_tf_network(config)
model$num_cores <- 11
#model$num_cores <- 60
#plot_feature_network(model, show_targets = FALSE)
model <- generate_feature_network(model)
#plot_feature_network(model)
#plot_feature_network(model, show_hks = TRUE)
model <- generate_kinetics(model)
#plot_feature_network(model)
model <- generate_gold_standard(model)
#plot_gold_simulations(model) + scale_colour_brewer(palette = "Dark2")
model <- generate_cells(model)
#plot_simulations(model)
model <- generate_experiment(model)
dataset <- as_dyno(model)

M1 <- as.matrix(dataset$counts)

#

backbone <- bblego(
  bblego_start("A", type = "doublerep2", num_modules = 4),
  bblego_linear("A", "B", type = "doublerep2", num_modules = 10),
  bblego_branching("B", c("C", "D", "E"), type = "simple",num_modules = 20),
  bblego_end("C", type = "doublerep2", num_modules = 10),
  bblego_end("D", type = "doublerep2", num_modules = 10),
  bblego_end("E", type = "doublerep2", num_modules = 10)
)

config <-
  initialise_model(
    backbone = backbone,
    num_tfs = 750,
    num_cells = 2000,
    num_targets = 0,
    num_hks = 0,
    verbose = FALSE,
    simulation_params = simulation_default(ssa_algorithm = ssa_etl(tau = 10 / 3600),census_interval = 2)
)

# Run the model and create the dataset. Un-comment any of the plotting fucntions to look at the underlying model that creates the data.
#plot_backbone_statenet(config)
#plot_backbone_modulenet(config)
model <- generate_tf_network(config)
model$num_cores <- 11
#model$num_cores <- 60
#plot_feature_network(model, show_targets = FALSE)
model <- generate_feature_network(model)
#plot_feature_network(model)
#plot_feature_network(model, show_hks = TRUE)
model <- generate_kinetics(model)
#plot_feature_network(model)
model <- generate_gold_standard(model)
#plot_gold_simulations(model) + scale_colour_brewer(palette = "Dark2")
model <- generate_cells(model)
#plot_simulations(model)
model <- generate_experiment(model)
dataset <- as_dyno(model)

M2 <- as.matrix(dataset$counts)

M3 <- rbind(cbind(M1,matrix(0,nrow=nrow(M1),ncol=ncol(M2))),cbind(matrix(0,nrow=nrow(M2),ncol=ncol(M1)),M2))

#write.csv(M3,"Dyngen_Counts.csv")


# Generate housekeeping genes

config <-
  initialise_model(
    backbone = backbone,
    num_tfs = 0,
    num_cells = 4000,
    num_targets = 0,
    num_hks = 3000,
    verbose = FALSE,
    simulation_params = simulation_default(ssa_algorithm = ssa_etl(tau = 100 / 3600))#,census_interval = 2)
)

# Run the model and create the dataset. Un-comment any of the plotting fucntions to look at the underlying model that creates the data.
#plot_backbone_statenet(config)
#plot_backbone_modulenet(config)
model <- generate_tf_network(config)
model$num_cores <- 11
#model$num_cores <- 60
#plot_feature_network(model, show_targets = FALSE)
model <- generate_feature_network(model)
#plot_feature_network(model)
#plot_feature_network(model, show_hks = TRUE)
model <- generate_kinetics(model)
#plot_feature_network(model)
model <- generate_gold_standard(model)
#plot_gold_simulations(model) + scale_colour_brewer(palette = "Dark2")
model <- generate_cells(model)
#plot_simulations(model)
model <- generate_experiment(model)
dataset <- as_dyno(model)

write.csv(as.matrix(dataset$counts),"HK_Genes.csv")


####################

import numpy as np
import matplotlib.pyplot as plt
import umap
import pandas as pd

path = "/home/ahr35/Dyngen_Data_Paper/Dyngen_Data_2/"
Dyngen_Counts = pd.read_csv(path + "Dyngen_Counts.csv",header=0,index_col=0)

Dyngen_Embedding = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2).fit_transform(Dyngen_Counts)
plt.figure()
plt.scatter(Dyngen_Embedding[:,0],Dyngen_Embedding[:,1])

HK_Counts = pd.read_csv(path + "HK_Genes.csv",header=0,index_col=0)
Dyngen_Counts.index = HK_Counts.index
Dyngen_Counts = pd.concat((Dyngen_Counts,HK_Counts),axis=1)

Dyngen_Embedding = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2).fit_transform(np.asarray(Dyngen_Counts))
plt.figure()
plt.scatter(Dyngen_Embedding[:,0],Dyngen_Embedding[:,1])

plt.show()


# The following code with add a set of randomly expressed genes, false negative drop outs, and drop out batch effects.

import numpy as np
import matplotlib.pyplot as plt
import umap
import pandas as pd
import copy

path = "/home/ahr35/Dyngen_Data_Paper/Dyngen_Data_2/"

Dyngen_Counts = pd.read_csv(path + "Dyngen_Counts.csv",header=0,index_col=0)

Remove_Burner_Genes = np.empty(0)
for i in np.arange(Dyngen_Counts.columns.shape[0]):
    if "Burn" in Dyngen_Counts.columns[i]:
        Remove_Burner_Genes = np.append(Remove_Burner_Genes,i)

Remove_Burner_Genes = Dyngen_Counts.columns[Remove_Burner_Genes.astype("i")]
Dyngen_Counts = Dyngen_Counts.drop(columns=Remove_Burner_Genes)

HK_Counts = pd.read_csv(path + "HK_Genes.csv",header=0,index_col=0)
Dyngen_Counts.index = HK_Counts.index
Comb_Dyngen_Counts = pd.concat((Dyngen_Counts,HK_Counts),axis=1)

Remove_Zero_Expression_Genes = np.where(np.sum(np.asarray(Comb_Dyngen_Counts),axis=0)==0)[0]
Comb_Dyngen_Counts = Comb_Dyngen_Counts.drop(columns=Comb_Dyngen_Counts.columns[Remove_Zero_Expression_Genes])

Synthetic_Data = np.asarray(copy.copy(Comb_Dyngen_Counts))

# Add randomly expressed genes genes by selecting a random proportion of cells to have the gene active in, and then using
# the expression distribution of an existing structured or housekeeping gene to sample the expression values.
Num_Noise_Features = 2595
Gene_IDs = np.zeros(Num_Noise_Features).astype("str")
for i in np.arange(Num_Noise_Features):
    Pick_Gene = np.random.randint(0,3000)
    Gene = Synthetic_Data[:,Pick_Gene]
    Expression_Profile = np.unique(Gene,return_counts=True)
    Expression_Values = Expression_Profile[0][np.arange(1,Expression_Profile[0].shape[0])]
    Expression_Probabilities = Expression_Profile[1][np.arange(1,Expression_Profile[1].shape[0])]
    Expression_Probabilities = Expression_Probabilities/np.sum(Expression_Probabilities)
    Random_Int = 20
    while Random_Int >= 10 or Random_Int <= 0:
        Random_Int = int(np.random.normal(3,3))
    Random_Gene = np.random.choice([0, 1], size=(Synthetic_Data.shape[0]), p=[(10-Random_Int)/10, (Random_Int)/10])  
    Random_Gene[np.where(Random_Gene==1)[0]] = np.random.choice(Expression_Values, np.sum(Random_Gene), p=Expression_Probabilities)
    Synthetic_Data = np.column_stack((Synthetic_Data,Random_Gene))
    Gene_IDs[i] = "NG-"+ str(i+1)


Complete_Synthetic_Data = pd.DataFrame(Synthetic_Data,columns=np.append(Comb_Dyngen_Counts.columns,Gene_IDs))
# Un-comment to save gene expression matrix
#Complete_Synthetic_Data.to_csv(path + "Complete_Synthetic_Data.csv")

# Visualise the data
Complete_Synthetic_Data = pd.read_csv(path + "Complete_Synthetic_Data.csv",header=0,index_col=0)

Dyngen_Embedding = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2).fit_transform(np.asarray(Dyngen_Counts))
plt.figure()
plt.scatter(Dyngen_Embedding[:,0],Dyngen_Embedding[:,1])

Dyngen_Embedding = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2).fit_transform(np.asarray(Complete_Synthetic_Data))
plt.figure()
plt.scatter(Dyngen_Embedding[:,0],Dyngen_Embedding[:,1])
plt.show()

####

Synthetic_Data = copy.copy(np.asarray(Complete_Synthetic_Data))
Total_Cells = Synthetic_Data.shape[0]
Dropout_Probabiliy = 0.3
#### Add FNs ####
# Two pool batch effects where odd or evening numbered samples/cells have a higher faction of dropouts than
# the corresponding odd or even samples. The drop out bias is determined by sampling from a normal distribution
# with mean 1 and standard deviation 0.4, and using the distance from the mean to determine whether the
# bias will be in the odd or even samples, and the magnitude of the bias.
Batch_1 = np.arange(0,Total_Cells,2) # Batch one is the even numbered cells
Batch_2 = np.arange(1,Total_Cells,2) # Batch two is the odd numbered cells
for i in np.arange(Synthetic_Data.shape[1]):
    Gene = Synthetic_Data[:,i]
    Batch_Bias = 5
    while Batch_Bias >= 2 or Batch_Bias <= 0:
        Batch_Bias = np.random.normal(1,0.3) 
    if Batch_Bias < 1: # If lower than more dropouts in Batch_1
        Batch_Drop_Outs = Batch_1[np.where(Gene[Batch_1] != 0)[0]]
        Batch_Drop_Outs = Batch_Drop_Outs[np.random.choice(Batch_Drop_Outs.shape[0],(Batch_Drop_Outs.shape[0]-(int(Batch_Drop_Outs.shape[0]*Batch_Bias))),replace=False)]
        Synthetic_Data[Batch_Drop_Outs,i] = Synthetic_Data[Batch_Drop_Outs,i] * 0
    if Batch_Bias > 1: # If greater than more dropouts in Batch_2
        Batch_Bias = 2 - Batch_Bias
        Batch_Drop_Outs = Batch_2[np.where(Gene[Batch_2] != 0)[0]]
        Batch_Drop_Outs = Batch_Drop_Outs[np.random.choice(Batch_Drop_Outs.shape[0],(Batch_Drop_Outs.shape[0]-(int(Batch_Drop_Outs.shape[0]*Batch_Bias))),replace=False)]
        Synthetic_Data[Batch_Drop_Outs,i] = Synthetic_Data[Batch_Drop_Outs,i] * 0
## Random Capture Efficiency Batch Effects. These are the main source of false negative dropouts in the data. They are introduced
# by randomly selecting a franction of all the data, and switching them to majority state expression if they currently display
# minority state expression.
Drop_Out = np.random.choice([-1, 0], size=(Synthetic_Data.shape[0],Synthetic_Data.shape[1]), p=[Dropout_Probabiliy, 1-Dropout_Probabiliy])
Drop_Out = np.where(Drop_Out == -1)
Synthetic_Data[Drop_Out] = 0

Dropout_Synthetic_Data = pd.DataFrame(Synthetic_Data,columns=Complete_Synthetic_Data.columns)
# Un-comment to save gene expression matrix
#]Dropout_Synthetic_Data.to_csv(path + "Dropout_Synthetic_Data.csv")

# Visualise all the layers of data from just structured genes to all genes with drop outs and batch effects added.

Dyngen_Counts = pd.read_csv(path + "Dyngen_Counts.csv",header=0,index_col=0)
Remove_Burner_Genes = np.empty(0)
for i in np.arange(Dyngen_Counts.columns.shape[0]):
    if "Burn" in Dyngen_Counts.columns[i]:
        Remove_Burner_Genes = np.append(Remove_Burner_Genes,i)

Remove_Burner_Genes = Dyngen_Counts.columns[Remove_Burner_Genes.astype("i")]
Dyngen_Counts = Dyngen_Counts.drop(columns=Remove_Burner_Genes)
Remove_Zero_Expression_Genes = np.where(np.sum(np.asarray(Dyngen_Counts),axis=0)==0)[0]
Dyngen_Counts = Dyngen_Counts.drop(columns=Dyngen_Counts.columns[Remove_Zero_Expression_Genes])

Complete_Synthetic_Data = pd.read_csv(path + "Complete_Synthetic_Data.csv",header=0,index_col=0)
Dropout_Synthetic_Data = pd.read_csv(path + "Dropout_Synthetic_Data.csv",header=0,index_col=0)

Dyngen_Embedding = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2).fit_transform(np.asarray(Dyngen_Counts))
Complete_Synthetic_Data_Embedding = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2).fit_transform(np.asarray(Complete_Synthetic_Data))
Dropout_Synthetic_Data_Embedding = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2).fit_transform(np.asarray(Dropout_Synthetic_Data))

plt.figure()
plt.scatter(Dyngen_Embedding[:,0],Dyngen_Embedding[:,1])
plt.figure()
plt.scatter(Complete_Synthetic_Data_Embedding[:,0],Complete_Synthetic_Data_Embedding[:,1])
plt.figure()

Total_Cells = Dropout_Synthetic_Data.shape[0]
Batch_1 = np.arange(0,Total_Cells,2)
Batch_2 = np.arange(1,Total_Cells,2)
plt.scatter(Dropout_Synthetic_Data_Embedding[Batch_1,0],Dropout_Synthetic_Data_Embedding[Batch_1,1])
plt.scatter(Dropout_Synthetic_Data_Embedding[Batch_2,0],Dropout_Synthetic_Data_Embedding[Batch_2,1])
plt.show()

####################

# In Python, run FFAVES to identify spurious data points and perform feature selection

import numpy as np
import matplotlib.pyplot as plt
import umap
import pandas as pd
import copy
import ffaves

path = "/home/ahr35/Dyngen_Data_Paper/Dyngen_Data_2/"
Dropout_Synthetic_Data = pd.read_csv(path + "Dropout_Synthetic_Data.csv",header=0,index_col=0)
Discretised_Data = np.asarray(copy.copy(Dropout_Synthetic_Data))
Discretised_Data[Discretised_Data > 0] = 1

Track_Imputation_Steps, Track_Percentage_Imputation = ffaves.FFAVES(Discretised_Data, Auto_Save=False)
# Calculate correlation matricies
Chosen_Cycle = 0
Sort_Gains, Sort_Weights, Cycle_Suggested_Imputations, ES_Matrices_Features_Used_Inds = ffaves.Calculate_ES_Sort_Matricies(Discretised_Data, Track_Imputation_Steps, Chosen_Cycle=Chosen_Cycle, Auto_Save=False)
# ESFW
Feature_Divergences, Cycle_Suggested_Imputations, Feature_Divergences_Used_Inds_0 = ffaves.ESFW(0.1, Discretised_Data, Track_Imputation_Steps, Chosen_Cycle=Chosen_Cycle, Auto_Save=False)
# Feature weights
Mean_Feature_Divergences_0 = np.mean(Feature_Divergences,axis=0) # Feature weights prior to applying FFAVES

plt.figure()
plt.hist(Mean_Feature_Divergences_0,bins=30)
plt.figure()
plt.scatter(Feature_Divergences_Used_Inds_0,Mean_Feature_Divergences_0)  

# Calculate correlation matricies
Chosen_Cycle = -1
Sort_Gains, Sort_Weights, Cycle_Suggested_Imputations, ES_Matrices_Features_Used_Inds = ffaves.Calculate_ES_Sort_Matricies(Discretised_Data, Track_Imputation_Steps, Chosen_Cycle=Chosen_Cycle, Auto_Save=False)
# ESFW
Feature_Divergences, Cycle_Suggested_Imputations, Feature_Divergences_Used_Inds_1 = ffaves.ESFW(0.1, Discretised_Data, Track_Imputation_Steps, Chosen_Cycle=Chosen_Cycle, Auto_Save=False)
# Feature weights
Mean_Feature_Divergences_1 = np.mean(Feature_Divergences,axis=0) # Feature weights after to applying FFAVES

plt.figure()
plt.hist(Mean_Feature_Divergences_1,bins=30)
plt.figure()
plt.scatter(Feature_Divergences_Used_Inds_1,Mean_Feature_Divergences_1)

plt.show()

# Typically the function paramter Auto_Save defaults to True, meaning files are automatically saved to the current directory.
# However, to keep the directories tidy for this our synthetica data example, we will save to the data directory pulled from github.
np.save(path+"Track_Percentage_Imputation.npy",Track_Percentage_Imputation)
np.save(path+"Track_Imputation_Steps.npy",Track_Imputation_Steps)
np.save(path+"Cycle_Suggested_Imputations.npy",Cycle_Suggested_Imputations)
np.save(path+"Mean_Feature_Divergences_0.npy",Mean_Feature_Divergences_0)
np.save(path+"Mean_Feature_Divergences_1.npy",Mean_Feature_Divergences_1)
np.save(path+"Feature_Divergences_Used_Inds_0.npy",Feature_Divergences_Used_Inds_0)
np.save(path+"Feature_Divergences_Used_Inds_1.npy",Feature_Divergences_Used_Inds_1)

# Use a basic imputation software from sklearn to imput the suggested false negative data points

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge

path = "/home/ahr35/Dyngen_Data_Paper/Dyngen_Data_2/"
Dropout_Synthetic_Data = pd.read_csv(path + "Dropout_Synthetic_Data.csv",header=0,index_col=0)
Track_Imputation_Steps = np.load("Track_Imputation_Steps.npy",allow_pickle=True)
Chosen_Cycle = -1

imputer = IterativeImputer(BayesianRidge(),n_nearest_features=200,max_iter=10,sample_posterior=True,initial_strategy="median",min_value=0)

Impute_Data = np.asarray(copy.copy(Dropout_Synthetic_Data)).astype("f")
Impute_Data[Track_Imputation_Steps[Chosen_Cycle][2]] = np.nan
Imputed_Data = imputer.fit_transform(Impute_Data)

FFAVES_Imputed_Data = pd.DataFrame(Imputed_Data,columns=Complete_Synthetic_Data.columns)
# Un-comment to save gene expression matrix
FFAVES_Imputed_Data.to_csv(path + "Dyngen_Imputation_FFAVES.csv")

# Visualise all the layers of the synthetic data again

FFAVES_Imputed_Embedding = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2).fit_transform(np.asarray(FFAVES_Imputed_Data))
Structured_Genes = Dyngen_Counts.columns
Structured_Genes_FFAVES_Imputed_Embedding = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2).fit_transform(np.asarray(FFAVES_Imputed_Data[Structured_Genes]))

Batch_1 = np.arange(0,Dyngen_Counts.shape[0],2)
Batch_2 = np.arange(1,Dyngen_Counts.shape[0],2)

plt.figure()
plt.scatter(Dyngen_Embedding[Batch_1,0],Dyngen_Embedding[Batch_1,1])
plt.scatter(Dyngen_Embedding[Batch_2,0],Dyngen_Embedding[Batch_2,1])
plt.figure()
plt.scatter(Complete_Synthetic_Data_Embedding[Batch_1,0],Complete_Synthetic_Data_Embedding[Batch_1,1])
plt.scatter(Complete_Synthetic_Data_Embedding[Batch_2,0],Complete_Synthetic_Data_Embedding[Batch_2,1])
plt.figure()
plt.scatter(Dropout_Synthetic_Data_Embedding[Batch_1,0],Dropout_Synthetic_Data_Embedding[Batch_1,1])
plt.scatter(Dropout_Synthetic_Data_Embedding[Batch_2,0],Dropout_Synthetic_Data_Embedding[Batch_2,1])
plt.figure()
plt.scatter(FFAVES_Imputed_Embedding[Batch_1,0],FFAVES_Imputed_Embedding[Batch_1,1])
plt.scatter(FFAVES_Imputed_Embedding[Batch_2,0],FFAVES_Imputed_Embedding[Batch_2,1])
plt.figure()
plt.scatter(Structured_Genes_FFAVES_Imputed_Embedding[Batch_1,0],Structured_Genes_FFAVES_Imputed_Embedding[Batch_1,1])
plt.scatter(Structured_Genes_FFAVES_Imputed_Embedding[Batch_2,0],Structured_Genes_FFAVES_Imputed_Embedding[Batch_2,1])

plt.show()

####################

# Perform imputation of the synthetic data with other populat methods.
# The following code is all run in the R environment.
counts <- read.csv("Dropout_Synthetic_Data.csv")
counts <- counts[,-1]
# Columns are cells, rows are genes
counts <- t(counts)

### SAVER ###
library(SAVER)

counts.saver <- saver(counts, ncores = 11)
## Un-comment to save results ##
#write.csv(counts.saver$estimate,"Synthetic_Imputation_SAVER.csv")

### MAGIC ###
library(Rmagic)

MAGIC_data <- magic(t(counts))
## Un-comment to save results ##
#write.csv(MAGIC_data$result,"Synthetic_Imputation_MAGIC.csv")

#### ALRA ####
library(Seurat)
library(SeuratWrappers)

colnames(counts) <- 1:dim(counts)[2]
s_obj <- CreateSeuratObject(counts = counts, project = "Synthetic Data")

s_obj <- RunALRA(s_obj)
ALRA <- as.data.frame(s_obj@assays$alra@data)
## Un-comment to save results ##
#write.csv(ALRA,"Synthetic_Imputation_ALRA.csv")

####################

# Visualise all the results in python.

### Import packages ###
import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns
import pandas as pd
import matplotlib.colors as c
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score
import umap
###

path = "/home/ahr35/Dyngen_Data_Paper/Dyngen_Data_2/"

Dyngen_Counts = pd.read_csv(path + "Dyngen_Counts.csv",header=0,index_col=0)
Remove_Burner_Genes = np.empty(0)
for i in np.arange(Dyngen_Counts.columns.shape[0]):
    if "Burn" in Dyngen_Counts.columns[i]:
        Remove_Burner_Genes = np.append(Remove_Burner_Genes,i)

Remove_Burner_Genes = Dyngen_Counts.columns[Remove_Burner_Genes.astype("i")]
Dyngen_Counts = Dyngen_Counts.drop(columns=Remove_Burner_Genes)
Remove_Zero_Expression_Genes = np.where(np.sum(np.asarray(Dyngen_Counts),axis=0)==0)[0]
Dyngen_Counts = Dyngen_Counts.drop(columns=Dyngen_Counts.columns[Remove_Zero_Expression_Genes])
Structured_Genes = Dyngen_Counts.columns
Structured_Genes_Inds = np.where(np.isin(Dyngen_Counts.columns,Structured_Genes))[0]

### Plot UMAPs for imputed data with different methods ###
# Load starting data
Complete_Synthetic_Data = pd.read_csv(path+"Complete_Synthetic_Data.csv",header=0,index_col=0)
Drop_Out_Synthetic_Data = pd.read_csv(path+"Dropout_Synthetic_Data.csv",header=0,index_col=0)
# Load each imputed data. See the "Get_Imputed_Expression_Matricies.r" file for the creation of these objects.
Synthetic_Imputation_MAGIC = pd.read_csv(path+"Synthetic_Imputation_MAGIC.csv",header=0,index_col=0)
Synthetic_Imputation_SAVER = pd.read_csv(path+"Synthetic_Imputation_SAVER.csv",header=0,index_col=0).T
Synthetic_Imputation_FFAVES = pd.read_csv(path+"Dyngen_Imputation_FFAVES.csv",header=0,index_col=0)
Synthetic_Imputation_ALRA = pd.read_csv(path+"Synthetic_Imputation_ALRA.csv",header=0,index_col=0).T

Total_Cells = Complete_Synthetic_Data.shape[0]
Batch_1 = np.arange(0,Total_Cells,2)
Batch_2 = np.arange(1,Total_Cells,2)

Neighbours = 30
Dist = 0.3
Complete_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(Complete_Synthetic_Data)#[Structured_Genes])
Noisy_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(Drop_Out_Synthetic_Data)#[Structured_Genes])
MAGIC_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(Synthetic_Imputation_MAGIC)#[Structured_Genes])
SAVER_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(Synthetic_Imputation_SAVER)#[Structured_Genes])
FFAVES_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(Synthetic_Imputation_FFAVES)#[Structured_Genes])
ALRA_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(Synthetic_Imputation_ALRA)#[Structured_Genes])


kmeans = KMeans(n_clusters=10, random_state=0).fit(Dyngen_Counts)
Cell_Labels = kmeans.labels_
Unique_Cell_Labels = np.unique(Cell_Labels)

size = 0.5
# create 3x1 subplots
fig, axs = plt.subplots(nrows=6, ncols=1,figsize = (5,8))
# clear subplots
for ax in axs:
    ax.remove()

# add subfigure per subplot
gridspec = axs[0].get_subplotspec().get_gridspec()
subfigs = [fig.add_subfigure(gs) for gs in gridspec]
Row_Titles = np.array(["Ground Truth","30% Dropouts + Batch Effects","FFAVES Imputation","MAGIC Imputation","ALRA Imputation","SAVER Imputation"])
Row_Embeddings = [Complete_Embedding,Noisy_Embedding,FFAVES_Embedding,MAGIC_Embedding,ALRA_Embedding,SAVER_Embedding]

for row, subfig in enumerate(subfigs):
    subfig.suptitle(Row_Titles[row],fontsize=10)
    # create 1x3 subplots per subfig
    axs = subfig.subplots(nrows=1, ncols=2)
    for col, ax in enumerate(axs):
        if col == 0:
           for i in np.arange(Unique_Cell_Labels.shape[0]):
                IDs = np.where(Cell_Labels == Unique_Cell_Labels[i])[0]
                if Unique_Cell_Labels[i] == 5:
                    ax.scatter(Row_Embeddings[row][IDs,0],Row_Embeddings[row][IDs,1],s=size,zorder=-1)
                else:
                    ax.scatter(Row_Embeddings[row][IDs,0],Row_Embeddings[row][IDs,1],s=size) 
        if col == 1:
            ax.scatter(Row_Embeddings[row][Batch_1,0],Row_Embeddings[row][Batch_1,1],c="black",alpha=0.8,s=size)
            ax.scatter(Row_Embeddings[row][Batch_2,0],Row_Embeddings[row][Batch_2,1],c="gold",alpha=0.4,s=size)

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

plt.savefig(path + "Dyngen_2_All_genes.png",dpi=1000)
plt.close()
plt.show()


Neighbours = 30
Dist = 0.3
Structured_Complete_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(np.asarray(Complete_Synthetic_Data)[:,Structured_Genes_Inds])
Structured_Noisy_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(np.asarray(Drop_Out_Synthetic_Data)[:,Structured_Genes_Inds])
Structured_MAGIC_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(np.asarray(Synthetic_Imputation_MAGIC)[:,Structured_Genes_Inds])
Structured_SAVER_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(np.asarray(Synthetic_Imputation_SAVER)[:,Structured_Genes_Inds])
Structured_FFAVES_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(np.asarray(Synthetic_Imputation_FFAVES)[:,Structured_Genes_Inds])
Structured_ALRA_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(np.asarray(Synthetic_Imputation_ALRA)[:,Structured_Genes_Inds])


size = 0.5
# create 3x1 subplots
fig, axs = plt.subplots(nrows=6, ncols=1,figsize = (5,8))
# clear subplots
for ax in axs:
    ax.remove()

# add subfigure per subplot
gridspec = axs[0].get_subplotspec().get_gridspec()
subfigs = [fig.add_subfigure(gs) for gs in gridspec]
Row_Titles = np.array(["Ground Truth","30% Dropouts + Batch Effects","FFAVES Imputation","MAGIC Imputation","ALRA Imputation","SAVER Imputation"])
Row_Embeddings = [Structured_Complete_Embedding,Structured_Noisy_Embedding,Structured_FFAVES_Embedding,Structured_MAGIC_Embedding,Structured_ALRA_Embedding,Structured_SAVER_Embedding]

for row, subfig in enumerate(subfigs):
    subfig.suptitle(Row_Titles[row],fontsize=10)
    # create 1x3 subplots per subfig
    axs = subfig.subplots(nrows=1, ncols=2)
    for col, ax in enumerate(axs):
        if col == 0:
           for i in np.arange(Unique_Cell_Labels.shape[0]):
                IDs = np.where(Cell_Labels == Unique_Cell_Labels[i])[0]
                if Unique_Cell_Labels[i] == 5:
                    ax.scatter(Row_Embeddings[row][IDs,0],Row_Embeddings[row][IDs,1],s=size,zorder=-1)
                else:
                    ax.scatter(Row_Embeddings[row][IDs,0],Row_Embeddings[row][IDs,1],s=size) 
        if col == 1:
            ax.scatter(Row_Embeddings[row][Batch_1,0],Row_Embeddings[row][Batch_1,1],c="black",alpha=0.8,s=size)
            ax.scatter(Row_Embeddings[row][Batch_2,0],Row_Embeddings[row][Batch_2,1],c="gold",alpha=0.4,s=size)

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

plt.savefig(path + "Dyngen_2_structured_genes.png",dpi=1000)
plt.close()
plt.show()

##### Silhouette plots #####

Complete_silhouette_vals = silhouette_samples(Complete_Synthetic_Data,Cell_Labels)
Drop_Out_silhouette_vals = silhouette_samples(Drop_Out_Synthetic_Data,Cell_Labels)
MAGIC_silhouette_vals = silhouette_samples(Synthetic_Imputation_MAGIC,Cell_Labels)
SAVER_silhouette_vals = silhouette_samples(Synthetic_Imputation_SAVER,Cell_Labels)
FFAVES_silhouette_vals = silhouette_samples(Synthetic_Imputation_FFAVES,Cell_Labels)
ALRA_silhouette_vals = silhouette_samples(Synthetic_Imputation_ALRA,Cell_Labels)

Min_Silh = np.round(np.min(np.concatenate([Complete_silhouette_vals,Drop_Out_silhouette_vals,MAGIC_silhouette_vals,SAVER_silhouette_vals,FFAVES_silhouette_vals])),3)
Max_Silh = np.round(np.max(np.concatenate([Complete_silhouette_vals,Drop_Out_silhouette_vals,MAGIC_silhouette_vals,SAVER_silhouette_vals,FFAVES_silhouette_vals])),3)

plt.figure(figsize=(5.5,7))
plt.subplot(2,3,1)
plt.title("Ground Truth", fontsize=13)
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = Complete_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      

ax.set_yticks(np.arange(0,1200,200))
plt.xlim([Min_Silh/2, Max_Silh/2])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
ax.set_ylabel('Cell IDs',fontsize=15)
ax.set_title('Ground Truth',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.gca().invert_yaxis()
plt.subplots_adjust(0.02,0.02,0.98,0.9)

plt.subplot(2,3,2)
plt.title("Dropouts", fontsize=13)
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = Drop_Out_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      


plt.xlim([Min_Silh/2, Max_Silh/2])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
#ax.set_ylabel('Cell IDs',fontsize=15,c="white")
ax.set_title('30% Dropouts + Batch Effects',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.gca().invert_yaxis()
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)

plt.subplot(2,3,3)
plt.title("FFAVES", fontsize=13)
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = FFAVES_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      

plt.xlim([Min_Silh/2, Max_Silh/2])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
ax.set_ylabel('Cell IDs',fontsize=15,c="white")
ax.set_title('FFAVES',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.gca().invert_yaxis()
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)

plt.subplot(2,3,4)
plt.title("MAGIC", fontsize=13)
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = MAGIC_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      

plt.xlim([Min_Silh, Max_Silh])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
ax.set_ylabel('Cell IDs',fontsize=15)
ax.set_title('MAGIC',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.gca().invert_yaxis()
plt.subplots_adjust(0.02,0.02,0.98,0.9)

plt.subplot(2,3,5)
plt.title("ALRA", fontsize=13)
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = ALRA_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      

plt.xlim([Min_Silh/2, Max_Silh/2])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
ax.set_ylabel('Cell IDs',fontsize=15,c="white")
ax.set_title('ALRA',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),fontsize=15)
plt.gca().invert_yaxis()
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)

plt.subplot(2,3,6)
plt.title("SAVER", fontsize=13)
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = SAVER_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      

plt.xlim([Min_Silh/2, Max_Silh/2])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
ax.set_ylabel('Cell IDs',fontsize=15,c="white")
ax.set_title('SAVER',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.gca().invert_yaxis()
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)

plt.tight_layout()

plt.savefig(path + "Dyngen_2_all_genes_sil.png",dpi=1000)
plt.close()
plt.show()

########

Complete_silhouette_vals = silhouette_samples(np.asarray(Complete_Synthetic_Data)[:,Structured_Genes_Inds],Cell_Labels)
Drop_Out_silhouette_vals = silhouette_samples(np.asarray(Drop_Out_Synthetic_Data)[:,Structured_Genes_Inds],Cell_Labels)
MAGIC_silhouette_vals = silhouette_samples(np.asarray(Synthetic_Imputation_MAGIC)[:,Structured_Genes_Inds],Cell_Labels)
SAVER_silhouette_vals = silhouette_samples(np.asarray(Synthetic_Imputation_SAVER)[:,Structured_Genes_Inds],Cell_Labels)
FFAVES_silhouette_vals = silhouette_samples(np.asarray(Synthetic_Imputation_FFAVES)[:,Structured_Genes_Inds],Cell_Labels)
ALRA_silhouette_vals = silhouette_samples(np.asarray(Synthetic_Imputation_ALRA)[:,Structured_Genes_Inds],Cell_Labels)

Min_Silh = np.min(np.concatenate([Complete_silhouette_vals,Drop_Out_silhouette_vals,MAGIC_silhouette_vals,SAVER_silhouette_vals,FFAVES_silhouette_vals]))
Max_Silh = np.max(np.concatenate([Complete_silhouette_vals,Drop_Out_silhouette_vals,MAGIC_silhouette_vals,SAVER_silhouette_vals,FFAVES_silhouette_vals]))

plt.figure(figsize=(5.5,7))
plt.subplot(2,3,1)
plt.title("Ground Truth", fontsize=13)
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = Complete_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      

ax.set_yticks(np.arange(0,1200,200))
plt.xlim([Min_Silh, Max_Silh])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
ax.set_ylabel('Cell IDs',fontsize=15)
ax.set_title('Ground Truth',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.gca().invert_yaxis()
plt.subplots_adjust(0.02,0.02,0.98,0.9)

plt.subplot(2,3,2)
plt.title("Dropouts", fontsize=13)
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = Drop_Out_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      


plt.xlim([Min_Silh, Max_Silh])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
#ax.set_ylabel('Cell IDs',fontsize=15,c="white")
ax.set_title('30% Dropouts + Batch Effects',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.gca().invert_yaxis()
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)

plt.subplot(2,3,3)
plt.title("FFAVES", fontsize=13)
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = FFAVES_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      

plt.xlim([Min_Silh, Max_Silh])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
ax.set_ylabel('Cell IDs',fontsize=15,c="white")
ax.set_title('FFAVES',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.gca().invert_yaxis()
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)

plt.subplot(2,3,4)
plt.title("MAGIC", fontsize=13)
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = MAGIC_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      

plt.xlim([Min_Silh, Max_Silh])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
ax.set_ylabel('Cell IDs',fontsize=15)
ax.set_title('MAGIC',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.gca().invert_yaxis()
plt.subplots_adjust(0.02,0.02,0.98,0.9)

plt.subplot(2,3,5)
plt.title("ALRA", fontsize=13)
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = ALRA_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      

plt.xlim([Min_Silh, Max_Silh])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
ax.set_ylabel('Cell IDs',fontsize=15,c="white")
ax.set_title('ALRA',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),fontsize=15)
plt.gca().invert_yaxis()
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)

plt.subplot(2,3,6)
plt.title("SAVER", fontsize=13)
y_ticks = []
y_lower = y_upper = 0
ax.plot(np.arange(100,150),np.zeros(50),linestyle="--",c="k")
for i,cluster in enumerate(np.unique(Cell_Labels)):
    cluster_silhouette_vals = SAVER_silhouette_vals[Cell_Labels ==cluster]
    cluster_silhouette_vals.sort()
    GT_Cluster = Complete_silhouette_vals[Cell_Labels ==cluster]
    GT_Cluster.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1)
    plt.plot(GT_Cluster,range(y_lower,y_upper),linestyle="--", c="k")
    y_lower += len(cluster_silhouette_vals)      

plt.xlim([Min_Silh, Max_Silh])
ax.set_xlabel('Silhouette Coefficient Scores',fontsize=15)
ax.set_ylabel('Cell IDs',fontsize=15,c="white")
ax.set_title('SAVER',fontsize=15)
#plt.legend(labels=np.array(["Ground Truth Silhouettes"]),loc = "lower left")
plt.gca().invert_yaxis()
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)

plt.tight_layout()

plt.savefig(path + "Dyngen_2_structured_genes_sil.png",dpi=1000)
plt.close()
plt.show()


####################

### Feature selection comparisons

### R Code for SCRAN Highly Variable Genes ####
library(SingleCellExperiment)
library(scran)

counts <- read.csv("Dropout_Synthetic_Data.csv")
counts <- counts[,-1]
# Columns are cells, rows are genes
counts <- t(counts)
colnames(counts) <- as.character(paste0("Cell-",1:dim(counts)[2]))
rownames(counts) <- as.character(paste0("Gene-",1:dim(counts)[1]))
sce <- SingleCellExperiment(list(counts=counts),
    metadata=list(study="Synthetic Data"))

logcounts(sce) <- log(counts+1)
rownames(sce) <- as.character(paste0("Gene-",1:dim(counts)[1]))
clusters <- quickCluster(sce)
sce <- computeSumFactors(sce, clusters=clusters)
summary(sizeFactors(sce))
dec <- modelGeneVar(sce)
plot(dec$mean, dec$total, xlab="Mean log-expression", ylab="Variance")
curve(metadata(dec)$trend(x), col="blue", add=TRUE)
# Get the top 10% of genes.
top.hvgs <- getTopHVGs(dec, prop=1)
# Get the top 2000 genes.
top.hvgs2 <- getTopHVGs(dec, n=dim(counts)[1])
# Get all genes with positive biological components.
top.hvgs3 <- getTopHVGs(dec, var.threshold=0)
# Get all genes with FDR below 5%.
top.hvgs4 <- getTopHVGs(dec, fdr.threshold=0.05)

## Un-comment to save results ##
SCRAN_HVG_Order <- order(-dec$bio)
#write.csv(SCRAN_HVG_Order, "SCRAN_HVG_Order.csv")

#### R Code for Seurat Highly Variable Genes ####
library(Seurat)
library(SeuratWrappers)

s_obj <- CreateSeuratObject(counts = counts, project = "Synthetic Data")
s_obj <- NormalizeData(s_obj, normalization.method = "LogNormalize", scale.factor = 10000)
s_obj <- FindVariableFeatures(s_obj, selection.method = "vst", nfeatures = 5000)
# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(s_obj), 10)
# plot variable features with and without labels
plot1 <- VariableFeaturePlot(s_obj)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 + plot2
#Seurat_HVG_Order <- as.numeric(str_remove_all(VariableFeatures(s_obj), "X"))
Seurat_HVGs <- VariableFeatures(s_obj)
Orig_Order <- rownames(counts)
#Seurat_HVGs <- gsub("-", "_", Seurat_HVGs)
Seurat_HVG_Order <- match(Seurat_HVGs,Orig_Order)
Seurat_HVG_Order = append(Seurat_HVG_Order,which((Orig_Order %in% Seurat_HVGs)==0))

## Un-comment to save results ##
#write.csv(Seurat_HVG_Order, "Seurat_HVG_Order.csv")

#### R code for scry Highly Variable Genes ####
library(scry)

m <- data.frame((matrix(0, ncol = 2, nrow = 7500)))
rowData(sce) <- m
deviances <- rowData(devianceFeatureSelection(sce))$binomial_deviance
scry_HVG_Order <- as.numeric(order(-deviances))

## Un-comment to save results ##
#write.csv(scry_HVG_Order, "scry_HVG_Order.csv")

### Plot Highly Variable Gene Precision-Recall Curves ###
# The ESFW and ESFW + FFAVES gene rankings are simply the ranked feature weights obtained by the ESFW function.
# Higher scores are better.
####################

# Now that we have the ranked list of genes for each method we can create precision recall curves.
path = "/home/ahr35/Dyngen_Data_Paper/Dyngen_Data_2/"

Dropout_Synthetic_Data = pd.read_csv(path + "Dropout_Synthetic_Data.csv",header=0,index_col=0)
Dyngen_Counts = pd.read_csv(path + "Dyngen_Counts.csv",header=0,index_col=0)
Remove_Burner_Genes = np.empty(0)
for i in np.arange(Dyngen_Counts.columns.shape[0]):
    if "Burn" in Dyngen_Counts.columns[i]:
        Remove_Burner_Genes = np.append(Remove_Burner_Genes,i)

Remove_Burner_Genes = Dyngen_Counts.columns[Remove_Burner_Genes.astype("i")]
Dyngen_Counts = Dyngen_Counts.drop(columns=Remove_Burner_Genes)
Remove_Zero_Expression_Genes = np.where(np.sum(np.asarray(Dyngen_Counts),axis=0)==0)[0]
Dyngen_Counts = Dyngen_Counts.drop(columns=Dyngen_Counts.columns[Remove_Zero_Expression_Genes])
Structured_Genes = Dyngen_Counts.columns
Structured_Genes_Inds = np.where(np.isin(Dyngen_Counts.columns,Structured_Genes))[0]

Mean_Feature_Divergences_0 = np.load(path+"Mean_Feature_Divergences_0.npy")
Feature_Divergences_Used_Inds_0 = np.load(path+"Feature_Divergences_Used_Inds_0.npy")
Mean_Feature_Divergences_1 = np.load(path+"Mean_Feature_Divergences_1.npy")
Feature_Divergences_Used_Inds_1 = np.load(path+"Feature_Divergences_Used_Inds_1.npy")

FFAVES_HVG_Order_0 = np.zeros(Dropout_Synthetic_Data.shape[1])
FFAVES_HVG_Order_0[Feature_Divergences_Used_Inds_0] = Mean_Feature_Divergences_0
FFAVES_HVG_Order_0 = np.argsort(-FFAVES_HVG_Order_0)
FFAVES_HVG_Order_1 = np.zeros(Dropout_Synthetic_Data.shape[1])
FFAVES_HVG_Order_1[Feature_Divergences_Used_Inds_1] = Mean_Feature_Divergences_1
FFAVES_HVG_Order_1 = np.argsort(-FFAVES_HVG_Order_1)

# Load ranked gene list from SCRAN, Seurat and scry. See the "Get_Information_Rich_Gene_Rankings.r" 
# file for instructions on creating the following objects.
SCRAN_HVG_Order = pd.read_csv(path+"SCRAN_HVG_Order.csv",header=0,index_col=0)
SCRAN_HVG_Order = np.asarray(SCRAN_HVG_Order.T)[0]
Seurat_HVG_Order = pd.read_csv(path+"Seurat_HVG_Order.csv",header=0,index_col=0)
Seurat_HVG_Order = np.asarray(Seurat_HVG_Order.T)[0]
scry_HVG_Order = pd.read_csv(path+"scry_HVG_Order.csv",header=0,index_col=0)
scry_HVG_Order = np.asarray(scry_HVG_Order.T)[0]

Structured_Genes = Structured_Genes_Inds
Random_Genes = np.delete(np.arange(FFAVES_HVG_Order_0.shape[0]),Structured_Genes_Inds)
Top_Percentages = np.linspace(1,100,100)/100
FFAVES_Precisions_0 = np.zeros(Top_Percentages.shape[0])
FFAVES_Precisions_1 = np.zeros(Top_Percentages.shape[0])
SCRAN_Precisions = np.zeros(Top_Percentages.shape[0])
Seurat_Precisions = np.zeros(Top_Percentages.shape[0])
scry_Precisions = np.zeros(Top_Percentages.shape[0])
FFAVES_Recall_0 = np.zeros(Top_Percentages.shape[0])
FFAVES_Recall_1 = np.zeros(Top_Percentages.shape[0])
SCRAN_Recall = np.zeros(Top_Percentages.shape[0])
Seurat_Recall = np.zeros(Top_Percentages.shape[0])
scry_Recall = np.zeros(Top_Percentages.shape[0])

for i in np.arange(Top_Percentages.shape[0]):
    Top_Gene_Range = int(FFAVES_HVG_Order_0.shape[0] * Top_Percentages[i])
    FFAVES_Genes = FFAVES_HVG_Order_0[np.arange(Top_Gene_Range)]
    Num_Structured = np.sum(np.isin(Structured_Genes,FFAVES_Genes))
    Num_Random = np.sum(np.isin(Random_Genes,FFAVES_Genes))
    FFAVES_Precisions_0[i] = Num_Structured/Structured_Genes.shape[0]  
    FFAVES_Recall_0[i] = Num_Structured/Top_Gene_Range
    #
    Top_Gene_Range = int(FFAVES_HVG_Order_1.shape[0] * Top_Percentages[i])
    FFAVES_Genes = FFAVES_HVG_Order_1[np.arange(Top_Gene_Range)]
    Num_Structured = np.sum(np.isin(Structured_Genes,FFAVES_Genes))
    Num_Random = np.sum(np.isin(Random_Genes,FFAVES_Genes))
    FFAVES_Precisions_1[i] = Num_Structured/Structured_Genes.shape[0]  
    FFAVES_Recall_1[i] = Num_Structured/Top_Gene_Range
    #
    SCRAN_Genes = SCRAN_HVG_Order[np.arange(Top_Gene_Range)]
    Num_Structured = np.sum(np.isin(Structured_Genes,SCRAN_Genes))
    Num_Random = np.sum(np.isin(Random_Genes,SCRAN_Genes))
    SCRAN_Precisions[i] = Num_Structured/Structured_Genes.shape[0]  
    SCRAN_Recall[i] = Num_Structured/Top_Gene_Range
    #
    Seurat_Genes = Seurat_HVG_Order[np.arange(Top_Gene_Range)]
    Num_Structured = np.sum(np.isin(Structured_Genes,Seurat_Genes))
    Num_Random = np.sum(np.isin(Random_Genes,Seurat_Genes))
    Seurat_Precisions[i] = Num_Structured/Structured_Genes.shape[0]  
    Seurat_Recall[i] = Num_Structured/Top_Gene_Range
    #
    scry_Genes = scry_HVG_Order[np.arange(Top_Gene_Range)]
    Num_Structured = np.sum(np.isin(Structured_Genes,scry_Genes))
    Num_Random = np.sum(np.isin(Random_Genes,scry_Genes))
    scry_Precisions[i] = Num_Structured/Structured_Genes.shape[0]  
    scry_Recall[i] = Num_Structured/Top_Gene_Range

plt.figure(figsize=(7,5.5))
plt.plot(FFAVES_Precisions_0,FFAVES_Recall_0,label="ESFW without FFAVES")
plt.plot(FFAVES_Precisions_1,FFAVES_Recall_1,label="ESFW + FFAVES after 7 cycles",linestyle=(0, (5, 10)),linewidth=3)
plt.plot(SCRAN_Precisions,SCRAN_Recall,label="Scran")
plt.plot(Seurat_Precisions,Seurat_Recall,label="Seurat",c='#9467bd')
plt.plot(scry_Precisions,scry_Recall,label="scry",c="#e377c2",alpha=0.7)
plt.axhline(Structured_Genes.shape[0]/FFAVES_HVG_Order_0.shape[0],linestyle="--",c="r",zorder=-1,label="Equivalent to random sampling")
plt.title("Feature Selection\nPrecision-Recall",fontsize=15)
plt.xlabel("Recall",fontsize=13)
plt.ylabel("Precision",fontsize=13)
#plt.legend(facecolor='white', framealpha=1)

plt.savefig(path + "Dyngen_2_precision_recall.png",dpi=1000)
plt.close()
plt.show()



### Plot AUC Curves

Structured_Genes = Structured_Genes_Inds
Random_Genes = np.delete(np.arange(FFAVES_HVG_Order_0.shape[0]),Structured_Genes_Inds)
Top_Percentages = np.linspace(1,100,100)/100
FFAVES_Precisions_0 = np.zeros(Top_Percentages.shape[0])
FFAVES_Precisions_1 = np.zeros(Top_Percentages.shape[0])
SCRAN_Precisions = np.zeros(Top_Percentages.shape[0])
Seurat_Precisions = np.zeros(Top_Percentages.shape[0])
scry_Precisions = np.zeros(Top_Percentages.shape[0])
FFAVES_Recall_0 = np.zeros(Top_Percentages.shape[0])
FFAVES_Recall_1 = np.zeros(Top_Percentages.shape[0])
SCRAN_Recall = np.zeros(Top_Percentages.shape[0])
Seurat_Recall = np.zeros(Top_Percentages.shape[0])
scry_Recall = np.zeros(Top_Percentages.shape[0])

for i in np.arange(Top_Percentages.shape[0]):
    # specificity (= false positive fraction = FP/(FP+TN))
    Top_Gene_Range = int(FFAVES_HVG_Order_0.shape[0] * Top_Percentages[i])
    FFAVES_Genes = FFAVES_HVG_Order_0[np.arange(Top_Gene_Range)]
    Num_Structured = np.sum(np.isin(Structured_Genes,FFAVES_Genes))
    Num_Random = np.sum(np.isin(Random_Genes,FFAVES_Genes))
    Specificity = (Num_Random/(Random_Genes.shape[0]))
    Sensitivity = Num_Structured/(Structured_Genes.shape[0])
    FFAVES_Precisions_0[i] = Specificity  
    FFAVES_Recall_0[i] = Sensitivity
    #
    Top_Gene_Range = int(FFAVES_HVG_Order_1.shape[0] * Top_Percentages[i])
    FFAVES_Genes = FFAVES_HVG_Order_1[np.arange(Top_Gene_Range)]
    Num_Structured = np.sum(np.isin(Structured_Genes,FFAVES_Genes))
    Num_Random = np.sum(np.isin(Random_Genes,FFAVES_Genes))
    Specificity = (Num_Random/(Random_Genes.shape[0]))
    Sensitivity = Num_Structured/(Structured_Genes.shape[0])
    FFAVES_Precisions_1[i] = Specificity
    FFAVES_Recall_1[i] = Sensitivity
    #
    SCRAN_Genes = SCRAN_HVG_Order[np.arange(Top_Gene_Range)]
    Num_Structured = np.sum(np.isin(Structured_Genes,SCRAN_Genes))
    Num_Random = np.sum(np.isin(Random_Genes,SCRAN_Genes))
    Specificity = (Num_Random/(Random_Genes.shape[0]))
    Sensitivity = Num_Structured/(Structured_Genes.shape[0])
    SCRAN_Precisions[i] = Specificity
    SCRAN_Recall[i] = Sensitivity
    #
    Seurat_Genes = Seurat_HVG_Order[np.arange(Top_Gene_Range)]
    Num_Structured = np.sum(np.isin(Structured_Genes,Seurat_Genes))
    Num_Random = np.sum(np.isin(Random_Genes,Seurat_Genes))
    Specificity = (Num_Random/(Random_Genes.shape[0]))
    Sensitivity = Num_Structured/(Structured_Genes.shape[0])
    Seurat_Precisions[i] = Specificity
    Seurat_Recall[i] = Sensitivity
    #
    scry_Genes = scry_HVG_Order[np.arange(Top_Gene_Range)]
    Num_Structured = np.sum(np.isin(Structured_Genes,scry_Genes))
    Num_Random = np.sum(np.isin(Random_Genes,scry_Genes))
    Specificity = (Num_Random/(Random_Genes.shape[0]))
    Sensitivity = Num_Structured/(Structured_Genes.shape[0])
    scry_Precisions[i] = Specificity
    scry_Recall[i] = Sensitivity


FFAVES_0_AUC = np.round(metrics.auc(FFAVES_Precisions_0, FFAVES_Recall_0),3)
FFAVES_1_AUC = np.round(metrics.auc(FFAVES_Precisions_1, FFAVES_Recall_1),3)
SCRAN_AUC = np.round(metrics.auc(SCRAN_Precisions, SCRAN_Recall),3)
Seurat_AUC = np.round(metrics.auc(Seurat_Precisions, Seurat_Recall),3)
scry_AUC = np.round(metrics.auc(scry_Precisions, scry_Recall),3)

plt.figure(figsize=(7,5.5))
plt.plot(FFAVES_Precisions_0,FFAVES_Recall_0,label="ESFW without FFAVES" + " - AUC = "+ str(FFAVES_0_AUC))
plt.plot(FFAVES_Precisions_1,FFAVES_Recall_1,label="ESFW + FFAVES" + " - AUC = "+ str(FFAVES_1_AUC),linestyle=(0, (5, 10)),linewidth=3)
plt.plot(SCRAN_Precisions,SCRAN_Recall,label="Scran" + " - AUC = "+ str(SCRAN_AUC))
plt.plot(Seurat_Precisions,Seurat_Recall,label="Seurat" + " - AUC = "+ str(Seurat_AUC),c='#9467bd')
plt.plot(scry_Precisions,scry_Recall,label="scry" + " - AUC = "+ str(scry_AUC),c="#e377c2",alpha=0.7)
plt.xlabel("False positive rate", fontsize=12)
plt.ylabel("True positive rate", fontsize=12)
plt.title("Feature Selection\nArea Under Curve (AUC)",fontsize=15)
plt.legend()

plt.savefig(path + "Dyngen_2_AUC_Curves.png",dpi=1000)
plt.close()
plt.show()








