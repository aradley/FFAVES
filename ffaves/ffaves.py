# Arthur Branch

### Dependencies ###

import numpy as np
from functools import partial 
import multiprocessing
import copy
from scipy.stats import halfnorm, zscore
import matplotlib.pyplot as plt

### Dependencies ###

### Find the partition basis for each reference feature.
def Find_Partitions(Input_Binarised_Data):
    # Create variables to fill.
    Permutables = np.zeros(Input_Binarised_Data.shape[1])
    Switch_State_Inidicies = np.zeros(Input_Binarised_Data.shape[1])
    i = 0
    # For each feature, identify which state is less common, and track the sample IDs for that state.
    while i < Input_Binarised_Data.shape[1]:
        Feature_Temp = Input_Binarised_Data[:,i]
        Feature_Temp_1 = np.where(Feature_Temp == 1)[0]
        Feature_Temp_0 = np.where(Feature_Temp == 0)[0]
        if Feature_Temp_1.shape[0] >= Feature_Temp_0.shape[0]:
            Permutables[i] = Feature_Temp_0.shape[0]
            Switch_State_Inidicies[i] = 1
        if Feature_Temp_0.shape[0] > Feature_Temp_1.shape[0]:
            Permutables[i] = Feature_Temp_1.shape[0]
        i = i + 1
    # Return a vector of the cardinality of each less common group, and whether the less common group is actually the 0 (inactive) state.
    return Permutables, Switch_State_Inidicies

### Find minority group overlapping inds for each feature. Calculating this now streamlines future calculations
def Parallel_Find_Minority_Group_Overlaps(Use_Cores):
    Inds = np.arange(Gene_Cardinality)
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes = Use_Cores)
        Result = pool.map(Find_Minority_Group_Overlaps, Inds)
        pool.close()
        pool.join()
    Reference_Gene_Minority_Group_Overlaps = np.asarray(Result)
    return Reference_Gene_Minority_Group_Overlaps

def Find_Minority_Group_Overlaps(Ind):
    # For each feature, identify how often its minority state samples overlap with the minority state samples of every other feature.
    Reference_Gene = Minority_Group_Matrix[:,Ind]
    Temp_Input_Binarised_Data = Minority_Group_Matrix + Reference_Gene[:,np.newaxis]
    Overlaps = np.sum(Temp_Input_Binarised_Data == 2,axis=0)
    return Overlaps

### Parallel function calculated numerious Entropy Sorting properties while fixing a reference gene (RG) as the central point of calculation.
## By fixing the RG, you are essentially asking which points of the RG that are designated as being a member of the more common state of the RG
## but we find that this consistently contradicts the state structure of the query genes (QG) and hence we should reconsider whether the RG
## point should be changed from the more common state to the less common state. 
def Parallel_RG_Calc_ES_Info(Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores):
    # Append the feature Ind to the Reference_Gene_Minority_Group_Overlaps matrix so that each core knows which RG it is looking at.
    Feature_Inds = np.arange(Gene_Cardinality)
    Pass_Info_To_Cores = np.concatenate((Feature_Inds.reshape(1,Feature_Inds.shape[0]),Reference_Gene_Minority_Group_Overlaps))
    Pass_Info_To_Cores = np.transpose(Pass_Info_To_Cores)
    # Parrallel calculate information gains matrix
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes = Use_Cores)
        Result = pool.map(partial(RG_Calc_ES_Info, Cell_Cardinality=Cell_Cardinality,Permutables=Permutables), Pass_Info_To_Cores)
        pool.close()
        pool.join()
    Result = np.asarray(Result,dtype=object)
    # Retreive Information_Gain_Matrix
    # Retreive Cell_Divergence_Matrix and put the features back in the original feature ordering.
    Sort_Into_Cell_Divergences = np.stack(Result[:,0],axis=1)
    # Retreive Information_Gains_Matrix and put the features back in the original feature ordering.
    Information_Gains_Matrix = np.stack(Result[:,1],axis=1)
    Information_Gains_Matrix[np.isnan(Information_Gains_Matrix)] = 0
    # Retreive Weights_Matrix and put the features back in the original feature ordering.
    Weights_Matrix = np.stack(Result[:,2],axis=1)
    Weights_Matrix[np.isnan(Weights_Matrix)] = 0
    # Retreive Informative_Genes and put the features back in the original feature ordering.
    Informative_Genes = np.asarray(Result[:,3],dtype=object)
    return Sort_Into_Cell_Divergences, Information_Gains_Matrix, Weights_Matrix, Informative_Genes

### Calculate information gain matrix with fixed RG
def RG_Calc_ES_Info(Pass_Info_To_Cores,Cell_Cardinality,Permutables):
    # Extract which gene is being used as the Reference Gene
    Feature_Inds = int(Pass_Info_To_Cores[0])
    # Remove the Reference Gene ind from the data vector
    Reference_Gene_Minority_Group_Overlaps = np.delete(Pass_Info_To_Cores,0)
    if np.isnan(Permutables[Feature_Inds]) == 0:        
        ## Calculate Sorting Information for this fixed Reference Gene
        with np.errstate(invalid='ignore'):
            Information_Gains, Split_Weights, Split_Directions, Split_Permute_Entropies, Max_Permuation_Entropies, Max_Entropy_Permutation, Min_Entropy_ID_2, Minimum_Entropies = Calculate_RG_Sort_Values(Feature_Inds,Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps)
        ## Calculate Divergence Information
        ## Sort Into Minority Group Divergences
        # Identify features whose minority group sorts into the minority group of the RG
        Sort_Into_Divergence_Inds = np.where(np.logical_and(np.isnan(Permutables) == 0,Split_Directions == 1))[0]
        # Subset to these features
        Sort_Into_Query_Genes = Minority_Group_Matrix[:,Sort_Into_Divergence_Inds]
        # Overlay RG feature with the QG features
        Sort_Into_Query_Genes = Sort_Into_Query_Genes + np.tile((Minority_Group_Matrix[:,Feature_Inds]*-1),(Sort_Into_Query_Genes.shape[1],1)).T
        # Wherever the minority states of the QG and RG pairs do not overlap the value will equal 1, so ignore all other points.
        Sort_Into_Query_Genes[Sort_Into_Query_Genes != 1] = 0
        # Get the caluclated observed entropies for each RG/QG pair.
        Divergences = Split_Permute_Entropies[Sort_Into_Divergence_Inds] - Minimum_Entropies[Sort_Into_Divergence_Inds]
        # Identify how many cells don't overlap for each RG/QG pair.
        Divergent_Cell_Cardinalities = np.sum(Sort_Into_Query_Genes,axis=0)
        # Find the average divergence for each cell that is diverging from the optimal sort.
        with np.errstate(divide='ignore',invalid='ignore'):
            Cell_Divergences = Divergences / Divergent_Cell_Cardinalities
        # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
        Max_Num_Cell_Divergences = Min_Entropy_ID_2[Sort_Into_Divergence_Inds] - Max_Entropy_Permutation[Sort_Into_Divergence_Inds]
        Minimum_Background_Noise = Max_Permuation_Entropies[Sort_Into_Divergence_Inds]/Max_Num_Cell_Divergences
        # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
        RG_QG_Divergences = Cell_Divergences - Minimum_Background_Noise
        # Null/Ignore points that aren't usable.
        RG_QG_Divergences[np.isinf(RG_QG_Divergences)] = 0
        RG_QG_Divergences[np.isnan(RG_QG_Divergences)] = 0
        # Featues whose RG_QG_Divergences are less than 0 would add more entropy to the system per data point imputed.
        # Null these data points by setting all overlaps to 0.
        Uninformative_Genes = np.where(RG_QG_Divergences <= 0)[0]
        Informative_Genes = np.where(RG_QG_Divergences > 0)[0]
        Sort_Into_Query_Genes[:,Uninformative_Genes] = 0 
        # Mutiply the diverging cells by their average divergence and sum all the divergences for each RG/QG pair of each cell, to get a total
        # divergence for each cell.
        Ref_Sort_Into_Cell_Divergences = np.sum((Sort_Into_Query_Genes*RG_QG_Divergences),axis=1)
        # Count how many times a cell was observed to be divergent when this feature is the RG.
        Ref_Sort_Into_Divergence_Counts = np.sum((Sort_Into_Query_Genes),axis=1)
        # Normalise divergences
        #if Informative_Genes.shape[0] > 0:
        #    Ref_Sort_Into_Cell_Divergences = Ref_Sort_Into_Cell_Divergences / Informative_Genes.shape[0]
        #Normalise_Inds = np.where(Ref_Sort_Into_Divergence_Counts != 0)[0]
        #Ref_Sort_Into_Cell_Divergences[Normalise_Inds] = Ref_Sort_Into_Cell_Divergences[Normalise_Inds]/Ref_Sort_Into_Divergence_Counts[Normalise_Inds]
        # Track Informative Genes
        Informative_Genes = Sort_Into_Divergence_Inds[Informative_Genes]
    else:
        # When a feature cannot be used just give all points a value of 0.
        Ref_Sort_Into_Cell_Divergences = np.zeros(Cell_Cardinality)
        Information_Gains = np.zeros(Reference_Gene_Minority_Group_Overlaps.shape[0])
        Split_Directions = np.zeros(Reference_Gene_Minority_Group_Overlaps.shape[0])
        Split_Weights = np.zeros(Reference_Gene_Minority_Group_Overlaps.shape[0])
        Informative_Genes = []
    # Collate Results
    Results = []
    Results.append(Ref_Sort_Into_Cell_Divergences)
    Results.append(Information_Gains*Split_Directions)
    Results.append(Split_Weights)  
    Results.append(Informative_Genes) 
    # Output Results
    return Results 

### Perform the Entropy Sorting Calculations with fixed Reference Gene (RG) and all other features as the QGs.
def Calculate_RG_Sort_Values(Feature_Inds,Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps):
    # Note which QG features cannot be used (probably because their minority group cardinality does not meet the Min_Clust_Size threshold)
    Do_Not_Use = np.where(np.isnan(Permutables))[0]
    # Extract the group 1 and group 2 cardinalities. Group 1 is always the minority group in this set up.
    Group1_Cardinality = Permutables[Feature_Inds]
    Group2_Cardinality = Cell_Cardinality - Group1_Cardinality
    # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESQ)
    Max_Entropy_Permutation = (Group1_Cardinality * Permutables)/(Group1_Cardinality + Group2_Cardinality)
    # The maximum and minimum points of the ESQ are identified from the boundaries of the ESQ curve.
    Min_Entropy_ID_1 = np.zeros(Permutables.shape[0])
    Min_Entropy_ID_2 = copy.copy(Permutables)
    Check_Fits_Group_1 = Group1_Cardinality - Min_Entropy_ID_2
    # If the minority group of the QG is larger than the minority group of the RG then the boundary point is the cardinality of the RG minority group.
    Min_Entropy_ID_2[np.where(Check_Fits_Group_1 < 0)[0]] = Group1_Cardinality
    # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
    Split_Permute_Value = Reference_Gene_Minority_Group_Overlaps
    ## Calculate all of the critial points on the curve.
    # The maximum entropy of the RG/QG system.
    Max_Permuation_Entropies = Calc_RG_Entropies(Max_Entropy_Permutation,Group1_Cardinality,Group2_Cardinality,Permutables)
    Max_Permuation_Entropies[Do_Not_Use] = np.nan
    # The minimum entropy if none of the QG minority states are in the RG minority group.
    Min_Entropy_IDs_1 = Calc_RG_Entropies(Min_Entropy_ID_1,Group1_Cardinality,Group2_Cardinality,Permutables)
    Min_Entropy_IDs_1[Do_Not_Use] = np.nan
    # The minimum entropy if the RG minority group has as many of the QG minority state samples in it as possible.
    Min_Entropy_IDs_2 = Calc_RG_Entropies(Min_Entropy_ID_2,Group1_Cardinality,Group2_Cardinality,Permutables)
    Min_Entropy_IDs_2[Do_Not_Use] = np.nan
    # The entropy of the arrangment observed in the data set.
    Split_Permute_Entropies = Calc_RG_Entropies(Split_Permute_Value,Group1_Cardinality,Group2_Cardinality,Permutables)
    Split_Permute_Entropies[Do_Not_Use] = np.nan
    # Identify Split Direction (whether the observed arrangment is sorting towards the global minimum entropy or not. I.e. is the QG sorting into the
    # minority or majority group of the RG.)
    Sort_Into_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) >= 0)[0]
    Sort_Out_Of_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) < 0)[0]
    # Assign Split Directions for each RG/QG pair to a vector.
    Split_Directions = np.repeat(1,Permutables.shape[0])
    Split_Directions[Sort_Out_Of_Inds] = -1
    # Assign Minimum Entropies for each RG/QG pair to a vector.
    Minimum_Entropies = np.zeros(Permutables.shape[0])
    Minimum_Entropies[Sort_Into_Inds] = Min_Entropy_IDs_2[Sort_Into_Inds]
    Minimum_Entropies[Sort_Out_Of_Inds] = Min_Entropy_IDs_1[Sort_Out_Of_Inds]
    # Calculate ES parabola properties
    Max_Entropy_Differences = Max_Permuation_Entropies - Minimum_Entropies
    Entropy_Losses = Max_Permuation_Entropies - Split_Permute_Entropies
    # Vector of Information Gain values for each RG/QG pair.
    Information_Gains = Entropy_Losses/Max_Entropy_Differences
    # Vector of Split Weights values for each RG/QG pair.
    Split_Weights = (Max_Permuation_Entropies - Minimum_Entropies) / Max_Permuation_Entropies
    return Information_Gains, Split_Weights, Split_Directions, Split_Permute_Entropies, Max_Permuation_Entropies, Max_Entropy_Permutation, Min_Entropy_ID_2, Minimum_Entropies


### Caclcuate entropies based on group cardinalities with fixed RG
def Calc_RG_Entropies(x,Group1_Cardinality,Group2_Cardinality,Permutables):
    ## Entropy Sort Equation (ESQ) is split into for parts for convenience
    # Equation 1
    Eq_1 = np.zeros(x.shape[0])
    Calculate_Inds = np.where((x/Group1_Cardinality) > 0)[0]
    Eq_1[Calculate_Inds] = - (x[Calculate_Inds]/Group1_Cardinality)*np.log2(x[Calculate_Inds]/Group1_Cardinality)  
    # Equation 2  
    Eq_2 = np.zeros(x.shape[0])
    Calculate_Inds = np.where(((Group1_Cardinality - x)/Group1_Cardinality) > 0)[0]
    Eq_2[Calculate_Inds] = - ((Group1_Cardinality - x[Calculate_Inds])/Group1_Cardinality)*np.log2(((Group1_Cardinality - x[Calculate_Inds])/Group1_Cardinality))  
    # Equation 3  
    Eq_3 = np.zeros(x.shape[0])
    Calculate_Inds = np.where(((Permutables - x) / Group2_Cardinality) > 0)[0]
    Eq_3[Calculate_Inds] = - ((Permutables[Calculate_Inds] - x[Calculate_Inds]) / Group2_Cardinality)*np.log2(((Permutables[Calculate_Inds] - x[Calculate_Inds]) / Group2_Cardinality))
    # Equation 4  
    Eq_4 = np.zeros(x.shape[0])
    Calculate_Inds = np.where(((Group2_Cardinality-Permutables+x)/Group2_Cardinality) > 0)[0]
    Eq_4[Calculate_Inds] = - ((Group2_Cardinality-Permutables[Calculate_Inds]+x[Calculate_Inds])/Group2_Cardinality)*np.log2(((Group2_Cardinality-Permutables[Calculate_Inds]+x[Calculate_Inds])/Group2_Cardinality))
    # Calculate overall entropy for each RG/QG pair
    Entropy = (Group1_Cardinality/(Group1_Cardinality+Group2_Cardinality))*(Eq_1 + Eq_2) + (Group2_Cardinality/(Group1_Cardinality+Group2_Cardinality))*(Eq_3 + Eq_4)
    return Entropy


### Parallel function calculated numerious Entropy Sorting properties while fixing a reference gene (QG) as the central point of calculation.
## By fixing the QG, you are essentially asking are there points of the QG that are designated as being a member of the less common state of the QG
## that consistently overlap with expression states of the RG arrangments that are inconsistent with the sort direction of the ES curve. If a QG points
## consistently diverges from the structure in this way, the point should be changed from the less common state to the more common state of the QG. 
def Parallel_QG_Calc_ES_Info(Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores):
    Feature_Inds = np.arange(Gene_Cardinality)
    Pass_Info_To_Cores = np.concatenate((Feature_Inds.reshape(1,Feature_Inds.shape[0]),Reference_Gene_Minority_Group_Overlaps))
    Pass_Info_To_Cores = np.transpose(Pass_Info_To_Cores)
    # Parrallel calculate information gains matrix
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes = Use_Cores)
        Result = pool.map(partial(QG_Calc_ES_Info, Cell_Cardinality=Cell_Cardinality,Permutables=Permutables), Pass_Info_To_Cores)
        pool.close()
        pool.join()
    Result = np.asarray(Result,dtype=object)
    # Retreive Information_Gain_Matrix
    # Retreive Sort_Out_Of_Cell_Divergences and put the features back in the original feature ordering.
    Sort_Out_Of_Cell_Divergences = np.stack(Result[:,0],axis=1)
    # Retreive Information_Gains_Matrix and put the features back in the original feature ordering.
    Information_Gains_Matrix = np.stack(Result[:,1],axis=1)
    Information_Gains_Matrix[np.isnan(Information_Gains_Matrix)] = 0
    # Retreive Weights_Matrix and put the features back in the original feature ordering.
    Weights_Matrix = np.stack(Result[:,2],axis=1)
    Weights_Matrix[np.isnan(Weights_Matrix)] = 0
    return Sort_Out_Of_Cell_Divergences, Information_Gains_Matrix, Weights_Matrix


### Calculate information gain matrix with fixed QG
def QG_Calc_ES_Info(Pass_Info_To_Cores,Cell_Cardinality,Permutables):
    # Extract which gene is being used as the Reference Gene
    Feature_Inds = int(Pass_Info_To_Cores[0])
    # Remove the Reference Gene ind from the data vector
    Reference_Gene_Minority_Group_Overlaps = np.delete(Pass_Info_To_Cores,0)
    if np.isnan(Permutables[Feature_Inds]) == 0:        
        ## Calculate Sorting Information for this fixed Query Gene
        with np.errstate(invalid='ignore'):
            Information_Gains, Split_Weights, Split_Directions, Split_Permute_Entropies, Max_Permuation_Entropies, Max_Entropy_Permutation, Min_Entropy_ID_2, Minimum_Entropies = Calculate_QG_Sort_Values(Feature_Inds,Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps)
        ## Calculate Divergence Information
        ## Sort Out Of Minority Group Divergences
        # Identify features where the minority state of the QG sorts into the majority state of the RG
        Sort_Out_Of_Divergence_Inds = np.where(np.logical_and(np.isnan(Permutables) == 0,Split_Directions == -1))[0] 
        # Subset to these features
        Sort_Out_Of_Query_Genes = Minority_Group_Matrix[:,Sort_Out_Of_Divergence_Inds]
        # Overlay QG feature with the RG features
        Sort_Out_Of_Query_Genes = Sort_Out_Of_Query_Genes + np.tile((Minority_Group_Matrix[:,Feature_Inds]),(Sort_Out_Of_Query_Genes.shape[1],1)).T
        # Wherever the minority states of the QG overlaps with the RG majority states we have a Sort Out Of group divergence so ignore all other points.
        Sort_Out_Of_Query_Genes[Sort_Out_Of_Query_Genes != 2] = 0
        Sort_Out_Of_Query_Genes[Sort_Out_Of_Query_Genes == 2] = 1
        # Get the caluclated observed entropies for each QG/RG pair.
        Divergences = Split_Permute_Entropies[Sort_Out_Of_Divergence_Inds]
        # Identify how many cells overlap for each QG/RG pair.
        Divergent_Cell_Cardinalities = np.sum(Sort_Out_Of_Query_Genes,axis=0)
        # Find the average divergence for each cell that is diverging from the optimal sort.
        with np.errstate(divide='ignore',invalid='ignore'):
            Cell_Divergences = Divergences / Divergent_Cell_Cardinalities
        # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
        Max_Num_Cell_Divergences = Max_Entropy_Permutation[Sort_Out_Of_Divergence_Inds]
        Minimum_Background_Noise = Max_Permuation_Entropies[Sort_Out_Of_Divergence_Inds]/Max_Num_Cell_Divergences
        # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
        RG_QG_Divergences = Cell_Divergences - Minimum_Background_Noise
        # Null/Ignore points that aren't usable.
        RG_QG_Divergences[np.isinf(RG_QG_Divergences)] = 0
        RG_QG_Divergences[np.isnan(RG_QG_Divergences)] = 0
        # Featues whose RG_QG_Divergences are less than 0 would add more entropy to the system per data point imputed.
        # Null these data points by setting all overlaps to 0.
        Uninformative_Genes = np.where(RG_QG_Divergences <= 0)[0]
        Informative_Genes = np.where(RG_QG_Divergences > 0)[0]
        Sort_Out_Of_Query_Genes[:,Uninformative_Genes] = 0 
        # Mutiply the diverging cells by their average divergence and sum all the divergences for each QG/RG pair of each cell, to get a total
        # divergence for each cell.
        Ref_Sort_Out_Of_Cell_Divergences = np.sum((Sort_Out_Of_Query_Genes*RG_QG_Divergences),axis=1)
        # Count how many times a cell was observed to be divergent when this feature is the RG.
        Ref_Sort_Out_Of_Divergence_Counts = np.sum((Sort_Out_Of_Query_Genes),axis=1)
        # Normalise divergences
        #if Informative_Genes.shape[0] > 0:
        #    Ref_Sort_Out_Of_Divergence_Counts = Ref_Sort_Out_Of_Divergence_Counts / Informative_Genes.shape[0]
        #Normalise_Inds = np.where(Ref_Sort_Out_Of_Divergence_Counts != 0)[0]
        #Ref_Sort_Out_Of_Cell_Divergences[Normalise_Inds] = Ref_Sort_Out_Of_Cell_Divergences[Normalise_Inds]/Ref_Sort_Out_Of_Divergence_Counts[Normalise_Inds]
    else:
        # When a feature cannot be used just give all points a value of 0.
        Ref_Sort_Out_Of_Cell_Divergences = np.zeros(Cell_Cardinality)
        Ref_Sort_Out_Of_Divergence_Counts = np.zeros(Cell_Cardinality)
        Information_Gains = np.zeros(Reference_Gene_Minority_Group_Overlaps.shape[0])
        Split_Directions = np.zeros(Reference_Gene_Minority_Group_Overlaps.shape[0])
        Split_Weights = np.zeros(Reference_Gene_Minority_Group_Overlaps.shape[0])
    # Collate Results
    Results = []
    Results.append(Ref_Sort_Out_Of_Cell_Divergences)
    Results.append(Information_Gains*Split_Directions)
    Results.append(Split_Weights) 
    # Output Results
    return Results


def Calculate_QG_Sort_Values(Feature_Inds,Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps):
    # Note which RG features cannot be used (probably because their minority group cardinality does not meet the Min_Clust_Size threshold)
    Do_Not_Use = np.where(np.isnan(Permutables))[0]
    # Extract the group 1 and group 2 cardinalities. Group 1 is always the minority group in this set up.
    Group1_Cardinality = Permutables
    Group2_Cardinality = Cell_Cardinality - Permutables
    Permutable = Permutables[Feature_Inds]
    # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESQ)
    Max_Entropy_Permutation = (Group1_Cardinality * Permutable)/(Group1_Cardinality + Group2_Cardinality)
    # The maximum and minimum points of the ESQ are identified from the boundaries of the ESQ curve.
    Min_Entropy_ID_1 = np.zeros(Permutables.shape[0])
    Min_Entropy_ID_2 = np.repeat(Permutable,Permutables.shape[0])
    Check_Fits_Group_1 = Group1_Cardinality - Min_Entropy_ID_2
    # If the minority group of the QG is larger than the minority group of the RG then the boundary point is the cardinality of the RG minority group.
    Min_Entropy_ID_2[np.where(Check_Fits_Group_1 < 0)[0]] = Group1_Cardinality[np.where(Check_Fits_Group_1 < 0)[0]]
    # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
    Split_Permute_Value = Reference_Gene_Minority_Group_Overlaps
    ## Calculate all of the critial points on the curve.
    # The maximum entropy of the RG/QG system.
    Max_Permuation_Entropies = Calc_QG_Entropies(Max_Entropy_Permutation,Group1_Cardinality,Group2_Cardinality,Permutable)
    Max_Permuation_Entropies[Do_Not_Use] = np.nan
    # The minimum entropy if none of the QG minority states are in the RG minority group.
    Min_Entropy_IDs_1 = Calc_QG_Entropies(Min_Entropy_ID_1,Group1_Cardinality,Group2_Cardinality,Permutable)
    Min_Entropy_IDs_1[Do_Not_Use] = np.nan
    # The minimum entropy if the RG minority group has as many of the QG minority state samples in it as possible.
    Min_Entropy_IDs_2 = Calc_QG_Entropies(Min_Entropy_ID_2,Group1_Cardinality,Group2_Cardinality,Permutable)
    Min_Entropy_IDs_2[Do_Not_Use] = np.nan
    # The entropy of the arrangment observed in the data set.
    Split_Permute_Entropies = Calc_QG_Entropies(Split_Permute_Value,Group1_Cardinality,Group2_Cardinality,Permutable)
    Split_Permute_Entropies[Do_Not_Use] = np.nan
    # Identify Split Direction (whether the observed arrangment is sorting towards the global minimum entropy or not. I.e. is the QG sorting into the
    # minority or majority group of the RG.)
    Sort_Into_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) >= 0)[0]
    Sort_Out_Of_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) < 0)[0]
    # Assign Split Directions for each QG/RG pair to a vector.
    Split_Directions = np.repeat(1,Permutables.shape[0])
    Split_Directions[Sort_Out_Of_Inds] = -1
    # Assign Minimum Entropies for each QG/RG pair to a vector.
    Minimum_Entropies = np.zeros(Permutables.shape[0])
    Minimum_Entropies[Sort_Into_Inds] = Min_Entropy_IDs_2[Sort_Into_Inds]
    Minimum_Entropies[Sort_Out_Of_Inds] = Min_Entropy_IDs_1[Sort_Out_Of_Inds]
    # Calculate ES parabola properties
    Max_Entropy_Differences = Max_Permuation_Entropies - Minimum_Entropies
    Entropy_Losses = Max_Permuation_Entropies - Split_Permute_Entropies
    # Vector of Information Gain values for each QG/RG pair.
    Information_Gains = Entropy_Losses/Max_Entropy_Differences
    # Vector of Split Weights values for each QG/RG pair.
    Split_Weights = (Max_Permuation_Entropies - Minimum_Entropies) / Max_Permuation_Entropies
    return Information_Gains, Split_Weights, Split_Directions, Split_Permute_Entropies, Max_Permuation_Entropies, Max_Entropy_Permutation, Min_Entropy_ID_2, Minimum_Entropies


### Caclcuate entropies based on group cardinalities with fixed QG
def Calc_QG_Entropies(x,Group1_Cardinality,Group2_Cardinality,Permutable):
    ## Entropy Sort Equation (ESQ) is split into for parts for convenience
    # Equation 1
    Eq_1 = np.zeros(x.shape[0])
    Calculate_Inds = np.where((x/Group1_Cardinality) > 0)[0]
    Eq_1[Calculate_Inds] = - (x[Calculate_Inds]/Group1_Cardinality[Calculate_Inds])*np.log2(x[Calculate_Inds]/Group1_Cardinality[Calculate_Inds])  
    # Equation 2  
    Eq_2 = np.zeros(x.shape[0])
    Calculate_Inds = np.where(((Group1_Cardinality - x)/Group1_Cardinality) > 0)[0]
    Eq_2[Calculate_Inds] = - ((Group1_Cardinality[Calculate_Inds] - x[Calculate_Inds])/Group1_Cardinality[Calculate_Inds])*np.log2(((Group1_Cardinality[Calculate_Inds] - x[Calculate_Inds])/Group1_Cardinality[Calculate_Inds]))  
    # Equation 3  
    Eq_3 = np.zeros(x.shape[0])
    Calculate_Inds = np.where(((Permutable - x) / Group2_Cardinality) > 0)[0]
    Eq_3[Calculate_Inds] = - ((Permutable - x[Calculate_Inds]) / Group2_Cardinality[Calculate_Inds])*np.log2(((Permutable - x[Calculate_Inds]) / Group2_Cardinality[Calculate_Inds]))
    # Equation 4  
    Eq_4 = np.zeros(x.shape[0])
    Calculate_Inds = np.where(((Group2_Cardinality-Permutable+x)/Group2_Cardinality) > 0)[0]
    Eq_4[Calculate_Inds] = - ((Group2_Cardinality[Calculate_Inds]-Permutable+x[Calculate_Inds])/Group2_Cardinality[Calculate_Inds])*np.log2(((Group2_Cardinality[Calculate_Inds]-Permutable+x[Calculate_Inds])/Group2_Cardinality[Calculate_Inds]))
    # Calculate overall entropy for each QG/RG pair
    Entropy = (Group1_Cardinality/(Group1_Cardinality+Group2_Cardinality))*(Eq_1 + Eq_2) + (Group2_Cardinality/(Group1_Cardinality+Group2_Cardinality))*(Eq_3 + Eq_4)
    return Entropy



#Binarised_Input_Matrix, Min_Clust_Size, Divergences_Significance_Cut_Off, Use_Cores, Num_Cycles, Save_Step_Plots = np.asarray(Fake_Data), 5, 0.99, (multiprocessing.cpu_count()-2), 1, 0

def FFAVES(Binarised_Input_Matrix, Min_Clust_Size = 5, Divergences_Significance_Cut_Off = 0.99, Use_Cores=(multiprocessing.cpu_count()-2), Num_Cycles = 1, Save_Step_Plots = 0):
    # Define data dimensions
    global Cell_Cardinality
    Cell_Cardinality = Binarised_Input_Matrix.shape[0]
    global Gene_Cardinality
    Gene_Cardinality = Binarised_Input_Matrix.shape[1]
    # Track what cycle FFAVES is on.
    Imputation_Cycle = 1
    print("Number of cells: " + str(Cell_Cardinality))
    print("Number of genes: " + str(Gene_Cardinality))
    Track_Percentage_Imputation = np.zeros((3,Num_Cycles+1))
    Track_Imputations = [[]] * (Num_Cycles + 1)
    # Combined Informative Genes
    Combined_Informative_Genes = [[]] * Gene_Cardinality
    while Imputation_Cycle <= Num_Cycles:
        if Imputation_Cycle > 1:
            print("Percentage of data suggested for imputation: " + str(np.round((Track_Imputations[Imputation_Cycle-1][0].shape[0]/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100,2)) + "%")   
            print("Percentage of data suggested as false negatives: " + str(np.round((np.sum(Binarised_Input_Matrix[Track_Imputations[Imputation_Cycle-1]] == 0)/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100,2)) + "%")
            print("Percentage of data suggested as false positives: " + str(np.round((np.sum(Binarised_Input_Matrix[Track_Imputations[Imputation_Cycle-1]] == 1)/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100,2)) + "%")
        print("Cycle Number " + str(Imputation_Cycle))
        print("Step 1: Identifying unreliable data points.")
        print('Identifying Sort Info')        
        # Set up Minority_Group_Matrix
        global Minority_Group_Matrix
        Minority_Group_Matrix = copy.copy(Binarised_Input_Matrix)
        # Convert suggested imputation points to correct state.
        Suggested_Impute_Inds = Track_Imputations[Imputation_Cycle-1]
        Minority_Group_Matrix[Suggested_Impute_Inds] = (Minority_Group_Matrix[Suggested_Impute_Inds] - 1) * -1         
        # Create Minority_Group_Matrix objects, Permutables and Switch_State_Inidicies objects.
        Permutables, Switch_State_Inidicies = Find_Partitions(Minority_Group_Matrix)
        # Switch Minority/Majority states to 0/1 where necessary.
        Switch_State_Indicie_Inds = np.where(Switch_State_Inidicies == 1)[0]
        Minority_Group_Matrix[:,Switch_State_Indicie_Inds] = (Minority_Group_Matrix[:,Switch_State_Indicie_Inds] * -1) + 1  
        # Calculate minority group overlap matrix
        if __name__ == "__main__":     
            Reference_Gene_Minority_Group_Overlaps = Parallel_Find_Minority_Group_Overlaps(Use_Cores)
        Permutables[Permutables < Min_Clust_Size] = np.nan
        print("Calculating Divergence Matricies")
        if __name__ == "__main__":
            Sort_Out_Of_Cell_Divergences, Information_Gains_Matrix, Weights_Matrix = Parallel_QG_Calc_ES_Info(Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores)            
        #plt.figure()
        #plt.imshow(Sort_Out_Of_Cell_Divergences.T)
        if Save_Step_Plots == 2:
            plt.figure()
            plt.imshow(Sort_Out_Of_Cell_Divergences.T, cmap='seismic', interpolation='nearest')
            Title = "Imputation Cycle " + str(Imputation_Cycle) + "\n Minority State Point Divergences From Prevailing Structure"
            plt.title(Title)
            plt.colorbar(fraction=0.046, pad=0.04).set_label(label='Gene Expression Value',size=10)
            plt.xlabel('Cells')
            plt.ylabel('Genes')
            plt.savefig((Title + ".svg"), format='svg', dpi=1200)
            plt.close() 
        print("Identifying unreliable data points via half normal distribution")
        # Use half normal distribution of normalised divergent points to suggest which points should be re-evaluated
        Use_Inds = np.where(Minority_Group_Matrix != 0)
        Divergences = Sort_Out_Of_Cell_Divergences[Use_Inds]    
        # Get zscores for observed divergences    
        zscores = zscore(Divergences)
        zscores = zscores + np.absolute(np.min(zscores))    
        if Save_Step_Plots == 2:
            param = halfnorm.fit(Divergences)
            plt.figure()
            Title = "Imputation Cycle " + str(Imputation_Cycle) + "\n Fitted Half-Normal Distribution of Minority State Divergences"          
            x = np.linspace(np.min(zscores),np.max(zscores), 100)
            pdf_fitted = halfnorm.pdf(x, *param[:-2], loc=param[-2]) 
            plt.hist(zscores,density=True,bins=30)
            plt.plot(x,pdf_fitted, label="halfnorm")
            plt.title(Title)
            plt.vlines(x=halfnorm.ppf(Divergences_Significance_Cut_Off),ymin=0,ymax=np.max(pdf_fitted),linestyle="--")
            plt.xlabel('Divergences')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig((Title + ".svg"), format='svg', dpi=1200)
            plt.close() 
        if Save_Step_Plots == 2:
            plt.figure()
            Title = "Imputation Cycle " + str(Imputation_Cycle) + "\n Initial Discretised State Matrix"
            plt.imshow((Minority_Group_Matrix).T, cmap='hot', interpolation='nearest')
            plt.colorbar(fraction=0.046, pad=0.04).set_label(label='Gene Expression Value',size=10)
            plt.xlabel('Cells')
            plt.ylabel('Genes')
            plt.savefig((Title + ".svg"), format='svg', dpi=1200)
            plt.close() 
        # Identify points that diverge in a statistically significant way
        Pass_Threshold = np.where(halfnorm.cdf(zscores) >= Divergences_Significance_Cut_Off)[0]
        Use_Inds = (Use_Inds[0][Pass_Threshold],Use_Inds[1][Pass_Threshold])
        # Temporarily switch their state. This switch is only temporary because this version of FFAVES works on the assumption that 
        # false postives in scRNA-seq data are incredibly unlikely, and hence leaky gene expression may be genuine biological heterogineity.
        # However, we remove it at this stage to try and keep the imputation strategy cleaner and more conservative in suggesting points to impute.
        Minority_Group_Matrix[Use_Inds] = (Minority_Group_Matrix[Use_Inds] - 1) * -1
        if Save_Step_Plots == 2:
            plt.figure()
            Title = "Imputation Cycle " + str(Imputation_Cycle) + "\n Cleaned Discretised State Matrix"
            plt.imshow((Minority_Group_Matrix).T, cmap='hot', interpolation='nearest')
            plt.colorbar(fraction=0.046, pad=0.04).set_label(label='Gene Expression Value',size=10)
            plt.xlabel('Cells')
            plt.ylabel('Genes')
            plt.savefig((Title + ".svg"), format='svg', dpi=1200)
            plt.close() 
        print("Step 2: Identifying data points for imputation.")
        print('Identifying Sort Info')
        # Create Minority_Group_Matrix objects, Permutables and Switch_State_Inidicies objects.
        Permutables, Switch_State_Inidicies = Find_Partitions(Minority_Group_Matrix)
        # Switch Minority/Majority states to 0/1 where necessary. 
        Switch_State_Indicie_Inds = np.where(Switch_State_Inidicies == 1)[0]
        Minority_Group_Matrix[:,Switch_State_Indicie_Inds] = (Minority_Group_Matrix[:,Switch_State_Indicie_Inds] * -1) + 1
        # Calculate minority group overlap matrix
        if __name__ == "__main__":
            Reference_Gene_Minority_Group_Overlaps = Parallel_Find_Minority_Group_Overlaps(Use_Cores)
        Permutables[Permutables < Min_Clust_Size] = np.nan
        print("Calculating Divergence Matricies")
        if __name__ == "__main__":
            Sort_Into_Cell_Divergences, Information_Gains_Matrix, Weights_Matrix, Informative_Genes = Parallel_RG_Calc_ES_Info(Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores)                    
        for i in np.arange(Gene_Cardinality):
            if np.asarray(Informative_Genes[i]).shape[0] > 0:
                Combined_Informative_Genes[i] = np.unique(np.append(Combined_Informative_Genes[i],Informative_Genes[i]))
        #plt.figure()
        #plt.imshow(Sort_Into_Cell_Divergences.T)
        if Save_Step_Plots == 2:
            plt.figure()
            plt.imshow(Sort_Into_Cell_Divergences.T, cmap='seismic', interpolation='nearest')
            Title = "Imputation Cycle " + str(Imputation_Cycle) + "\n Majority State Point Divergences From Prevailing Structure"
            plt.title(Title)
            plt.colorbar(fraction=0.046, pad=0.04).set_label(label='Gene Expression Value',size=10)
            plt.xlabel('Cells')
            plt.ylabel('Genes')
            plt.savefig((Title + ".svg"), format='svg', dpi=1200)
            plt.close()         
        print("Identifying data points for imputation via half normal distribution")
        # Use half normal distribution of normalised divergent points to suggest which points should be re-evaluated
        Use_Inds = np.where(Minority_Group_Matrix == 0)
        Divergences = Sort_Into_Cell_Divergences[Use_Inds]
        zscores = zscore(Divergences)
        zscores = zscores + np.absolute(np.min(zscores))              
        if Save_Step_Plots == 2:
            param = halfnorm.fit(Divergences)
            plt.figure()
            Title = "Imputation Cycle " + str(Imputation_Cycle) + "\n Fitted Half-Normal Distribution of Majority State Divergences"          
            x = np.linspace(np.min(zscores),np.max(zscores), 100)
            pdf_fitted = halfnorm.pdf(x, *param[:-2], loc=param[-2]) 
            plt.hist(zscores,density=True,bins=30)
            plt.plot(x,pdf_fitted, label="halfnorm")
            plt.title(Title)
            plt.vlines(x=halfnorm.ppf(Divergences_Significance_Cut_Off),ymin=0,ymax=np.max(pdf_fitted),linestyle="--")
            plt.xlabel('Divergences')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig((Title + ".svg"), format='svg', dpi=1200)
            plt.close() 
        # Identify points that diverge in a statistically significant way
        Pass_Threshold = np.where(halfnorm.cdf(zscores) >= Divergences_Significance_Cut_Off)[0]
        Use_Inds = (Use_Inds[0][Pass_Threshold],Use_Inds[1][Pass_Threshold])
        Step_2_Flat_Use_Inds = np.ravel_multi_index(Use_Inds, (Binarised_Input_Matrix.shape[0],Binarised_Input_Matrix.shape[1]))
        Minority_Group_Matrix[Use_Inds] = (Minority_Group_Matrix[Use_Inds] - 1) * -1
        if Save_Step_Plots == 2:
            plt.figure()
            Plot_Minority_Group_Matrix = copy.copy(Minority_Group_Matrix)
            Plot_Minority_Group_Matrix = Plot_Minority_Group_Matrix * -1
            Plot_Minority_Group_Matrix[Use_Inds] = (Plot_Minority_Group_Matrix[Use_Inds] * -1) + 1
            plt.imshow(Plot_Minority_Group_Matrix.T, cmap='seismic', interpolation='nearest')
            Title = "Imputation Cycle " + str(Imputation_Cycle) + "\n Track Initial Suggested Imputations"
            plt.title(Title)
            plt.colorbar(fraction=0.046, pad=0.04).set_label(label='Imputation Cycle',size=10)
            plt.xlabel('Cells')
            plt.ylabel('Genes')
            plt.savefig((Title + ".svg"), format='svg', dpi=1200)
            plt.close()   
        print("Step 3: Cleaning up untrustworthy imputed values.")
        print('Identifying Sort Info')
        # Create Minority_Group_Matrix objects, Permutables and Switch_State_Inidicies objects.
        Permutables, Switch_State_Inidicies = Find_Partitions(Minority_Group_Matrix)
        # Switch Minority/Majority states to 0/1 where necessary.
        Switch_State_Indicie_Inds = np.where(Switch_State_Inidicies == 1)[0]
        Minority_Group_Matrix[:,Switch_State_Indicie_Inds] = (Minority_Group_Matrix[:,Switch_State_Indicie_Inds] * -1) + 1
        # Calculate minority group overlap matrix
        if __name__ == "__main__":
            Reference_Gene_Minority_Group_Overlaps = Parallel_Find_Minority_Group_Overlaps(Use_Cores)
        Permutables[Permutables < Min_Clust_Size] = np.nan
        print("Calculating Divergence Matricies")
        if __name__ == "__main__":
            Sort_Out_Of_Cell_Divergences, Information_Gains_Matrix, Weights_Matrix = Parallel_QG_Calc_ES_Info(Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores)            
        print("Identifying unreliable imputed data points via half normal distribution")
        Use_Inds = np.where(Minority_Group_Matrix != 0)
        Divergences = Sort_Out_Of_Cell_Divergences[Use_Inds]
        zscores = zscore(Divergences)
        zscores = zscores + np.absolute(np.min(zscores))    
        # Identify points that diverge in a statistically significant way
        Pass_Threshold = np.where(halfnorm.cdf(zscores) >= Divergences_Significance_Cut_Off)[0]
        Use_Inds = (Use_Inds[0][Pass_Threshold],Use_Inds[1][Pass_Threshold])
        if Imputation_Cycle > 1:
            All_Impute_Inds = np.unique(np.append(np.ravel_multi_index(Track_Imputations[Imputation_Cycle-1], (Binarised_Input_Matrix.shape[0],Binarised_Input_Matrix.shape[1])), Step_2_Flat_Use_Inds))
        else:
            All_Impute_Inds = Step_2_Flat_Use_Inds
        Step_3_Flat_Use_Inds = np.ravel_multi_index(Use_Inds, (Binarised_Input_Matrix.shape[0],Binarised_Input_Matrix.shape[1]))
        Ignore_Imputations = np.where(np.isin(All_Impute_Inds,Step_3_Flat_Use_Inds))[0]
        All_Impute_Inds = np.delete(All_Impute_Inds,Ignore_Imputations)
        All_Impute_Inds = np.unravel_index(All_Impute_Inds,(Binarised_Input_Matrix.shape[0],Binarised_Input_Matrix.shape[1]))
        Track_Imputations[Imputation_Cycle] = All_Impute_Inds
        if Save_Step_Plots == 2:
            plt.figure()
            Plot_Minority_Group_Matrix = copy.copy(Minority_Group_Matrix)
            Plot_Minority_Group_Matrix = Plot_Minority_Group_Matrix * -1
            Plot_Minority_Group_Matrix[All_Impute_Inds] = (Plot_Minority_Group_Matrix[All_Impute_Inds] * -1) + 1
            plt.imshow(Plot_Minority_Group_Matrix.T, cmap='seismic', interpolation='nearest')
            Title = "Imputation Cycle " + str(Imputation_Cycle) + "\n Track Final Suggested Imputations"
            plt.title(Title)
            plt.colorbar(fraction=0.046, pad=0.04).set_label(label='Imputation Cycle',size=10)
            plt.xlabel('Cells')
            plt.ylabel('Genes')
            plt.savefig((Title + ".svg"), format='svg', dpi=1200)
            plt.close()         
        print("Finished")
        Track_Percentage_Imputation[0,Imputation_Cycle] = (Track_Imputations[Imputation_Cycle][0].shape[0]/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100
        Track_Percentage_Imputation[1,Imputation_Cycle] = (np.sum(Binarised_Input_Matrix[Track_Imputations[Imputation_Cycle]] == 0)/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100
        Track_Percentage_Imputation[2,Imputation_Cycle] = (np.sum(Binarised_Input_Matrix[Track_Imputations[Imputation_Cycle]] == 1)/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100
        print("Saving Track_Imputations")
        np.save("Track_Imputations.npy",np.asarray(Track_Imputations,dtype=object))
        np.save("Information_Gains_Matrix.npy",Information_Gains_Matrix)
        np.save("Weights_Matrix.npy",Weights_Matrix)
        np.save("Combined_Informative_Genes.npy",np.asarray(Combined_Informative_Genes,dtype=object))
        Imputation_Cycle = Imputation_Cycle + 1
        np.save("Track_Percentage_Imputation.npy",Track_Percentage_Imputation)
    print("Percentage of data suggested for imputation: " + str(np.round((Track_Imputations[Imputation_Cycle-1][0].shape[0]/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100,2)) + "%")      
    return Track_Imputations, Track_Percentage_Imputation, Information_Gains_Matrix, Weights_Matrix





#####################################

def Optimise_Discretisation_Thresholds(Gene):
    Original_Gene = Original_Data[:,Gene]
    Imputed_Gene = Imputed_Data[:,Gene]
    Imputed_Cells = np.where(np.isnan(Imputed_Gene))[0]
    Results = []
    if Imputed_Cells.shape[0] > 0:
        False_Negatives = Imputed_Cells[np.where(Original_Gene[Imputed_Cells] == 0)[0]]
        False_Positives = Imputed_Cells[np.where(Original_Gene[Imputed_Cells] != 0)[0]]
        Target_Expression_States = copy.copy(Original_Gene)
        Target_Expression_States[Target_Expression_States > 0] = 1
        if np.sum(Target_Expression_States) < Target_Expression_States.shape[0]:
            Target_Expression_States = (Target_Expression_States * -1) + 1
        Target_Expression_States[np.where(np.isnan(Imputed_Gene))[0]] = (Target_Expression_States[np.where(np.isnan(Imputed_Gene))[0]] * -1) + 1
        Unique_Exspression = np.unique(Original_Gene)
        Errors = np.zeros((3,Unique_Exspression.shape[0]))
        Min_Error = np.inf
        for Thresh in np.arange(Unique_Exspression.shape[0]):
            Threshold = Unique_Exspression[Thresh]
            Threshold_Expression_States = copy.copy(Original_Gene)
            Threshold_Expression_States[Threshold_Expression_States < Threshold] = 0
            Threshold_Expression_States[Threshold_Expression_States != 0] = 1
            False_Negatives = Imputed_Cells[np.where(Original_Gene[Imputed_Cells] < Threshold)[0]]
            False_Positives = Imputed_Cells[np.where(Original_Gene[Imputed_Cells] >= Threshold)[0]]
            if np.sum(Threshold_Expression_States) < Threshold_Expression_States.shape[0]:
                Threshold_Expression_States = (Threshold_Expression_States * -1) + 1
            Differences = np.absolute(Target_Expression_States-Threshold_Expression_States)
            Error = (np.sum(Differences)/Imputed_Cells.shape[0])
            Error_Inds = np.where(Differences != 0)[0]
            if Error < Min_Error:
                Min_Error = Error
                Min_Error_Ind = Unique_Exspression[Thresh]
                Impute_Cells = Imputed_Cells[np.where(np.isin(Imputed_Cells,Error_Inds) == 1)[0]]
                Impute_Vector = np.zeros(Original_Gene.shape[0])
                Impute_Vector[Impute_Cells] = 1
            Errors[0,Thresh] = Error
            Errors[1,Thresh] = False_Negatives.shape[0]
            Errors[2,Thresh] = False_Positives.shape[0]
        #plt.figure()
        #plt.plot(Unique_Exspression,Errors[0,:])
        #plt.vlines(Min_Error_Ind,min(Errors[0,:]),max(Errors[0,:]),color="r")
        Results.append(Min_Error_Ind)
        Results.append(Impute_Vector)
        #plt.figure()
        #plt.scatter(np.arange(Original_Gene.shape[0]),Original_Gene)
        #plt.scatter(np.where(np.isnan(Imputed_Data[:,Ind]))[0],Original_Gene[np.where(np.isnan(Imputed_Data[:,Ind]))[0]])
        #plt.scatter(Results[1],Original_Data[Results[1],Ind])
    else:
        Impute_Vector = np.zeros(Original_Gene.shape[0])
        Results.append(0)
        Results.append(Impute_Vector)
    return Results

def Parallel_Optimise_Discretisation_Thresholds(Gene_Cardinality,Use_Cores):
    Inds = np.arange(Gene_Cardinality)
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes = Use_Cores)
        Result = pool.map(Optimise_Discretisation_Thresholds, Inds)
        pool.close()
        pool.join()
    Result = np.asarray(Result,dtype=object)
    Thresholds = Result[:,0].astype("f")
    Imputations = np.stack(Result[:,1],axis=1)
    return Thresholds, Imputations











