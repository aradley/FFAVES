
### Dependencies ###

import numpy as np
from functools import partial 
import multiprocessing
import copy
from scipy.stats import halfnorm, zscore

### Dependencies ###

### Here we have the FFAVES wrapper function that executes all the steps of FFAVES. ###

def FFAVES(Binarised_Input_Matrix, Min_Clust_Size = 5, Divergences_Significance_Cut_Off = 0.99, Use_Cores= -1, Num_Cycles = 1, Auto_Save = 1):
    # Set number of cores to use
    Cores_Available = multiprocessing.cpu_count()
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Avaiblable: " + str(Cores_Available))
    print("Cores Used: " + str(Use_Cores))
    # Define data dimensions
    global Cell_Cardinality
    Cell_Cardinality = Binarised_Input_Matrix.shape[0]
    global Gene_Cardinality
    Gene_Cardinality = Binarised_Input_Matrix.shape[1]
    # Set up Minority_Group_Matrix
    global Minority_Group_Matrix
    # Track what cycle FFAVES is on.
    Imputation_Cycle = 1
    print("Number of cells: " + str(Cell_Cardinality))
    print("Number of genes: " + str(Gene_Cardinality))
    Track_Percentage_Imputation = np.zeros((3,Num_Cycles+1))
    Track_Imputations = [[]] * (Num_Cycles + 1)
    # Combined Informative Genes
    Combined_Informative_Genes = [[]] * Gene_Cardinality
    # Cell Uncertainties
    Track_Cell_Uncertainties = np.zeros((Num_Cycles,Cell_Cardinality))
    while Imputation_Cycle <= Num_Cycles:
        if Imputation_Cycle > 1:
            print("Percentage of data suggested for imputation: " + str(np.round((Track_Imputations[Imputation_Cycle-1][0].shape[0]/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100,2)) + "%")   
            print("Percentage of data suggested as false negatives: " + str(np.round((np.sum(Binarised_Input_Matrix[Track_Imputations[Imputation_Cycle-1]] == 0)/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100,2)) + "%")
            print("Percentage of data suggested as false positives: " + str(np.round((np.sum(Binarised_Input_Matrix[Track_Imputations[Imputation_Cycle-1]] == 1)/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100,2)) + "%")
        print("Cycle Number " + str(Imputation_Cycle))         
        Minority_Group_Matrix = copy.copy(Binarised_Input_Matrix)
        # Convert suggested imputation points to correct state.
        Suggested_Impute_Inds = Track_Imputations[Imputation_Cycle-1]
        Minority_Group_Matrix[Suggested_Impute_Inds] = (Minority_Group_Matrix[Suggested_Impute_Inds] - 1) * -1 
        ### Step 1 of FFAVES is to identify and temporarily remove spurious Minority Group expression states
        Use_Inds, Cell_Uncertainties = FFAVES_Step_1(Min_Clust_Size,Divergences_Significance_Cut_Off,Use_Cores)
        ###
        Track_Cell_Uncertainties[(Imputation_Cycle-1),:] = Cell_Uncertainties   
        # Temporarily switch their state. This switch is only temporary because this version of FFAVES works on the assumption that 
        # false postives in scRNA-seq data are incredibly unlikely, and hence leaky gene expression may be genuine biological heterogineity.
        # However, we remove it at this stage to try and keep the imputation strategy cleaner and more conservative in suggesting points to impute.
        Minority_Group_Matrix[Use_Inds] = (Minority_Group_Matrix[Use_Inds] - 1) * -1
        ### Step 2 of FFAVES is to identify which majority states points are spurious
        Use_Inds, Informative_Genes = FFAVES_Step_2(Min_Clust_Size,Divergences_Significance_Cut_Off,Use_Cores)
        ###
        for i in np.arange(Gene_Cardinality):
            if np.asarray(Informative_Genes[i]).shape[0] > 0:
                Combined_Informative_Genes[i] = np.unique(np.append(Combined_Informative_Genes[i],Informative_Genes[i]))       
        Step_2_Flat_Use_Inds = np.ravel_multi_index(Use_Inds, (Binarised_Input_Matrix.shape[0],Binarised_Input_Matrix.shape[1]))
        Minority_Group_Matrix[Use_Inds] = (Minority_Group_Matrix[Use_Inds] - 1) * -1
        ### Step 3 of FFAVES is to identify and remove spurious suggested imputations
        Use_Inds = FFAVES_Step_3(Min_Clust_Size,Divergences_Significance_Cut_Off,Use_Cores)
        ###
        if Imputation_Cycle > 1:
            All_Impute_Inds = np.unique(np.append(np.ravel_multi_index(Track_Imputations[Imputation_Cycle-1], (Binarised_Input_Matrix.shape[0],Binarised_Input_Matrix.shape[1])), Step_2_Flat_Use_Inds))
        else:
            All_Impute_Inds = Step_2_Flat_Use_Inds          
        Step_3_Flat_Use_Inds = np.ravel_multi_index(Use_Inds, (Binarised_Input_Matrix.shape[0],Binarised_Input_Matrix.shape[1]))
        Ignore_Imputations = np.where(np.isin(All_Impute_Inds,Step_3_Flat_Use_Inds))[0]
        All_Impute_Inds = np.delete(All_Impute_Inds,Ignore_Imputations)
        All_Impute_Inds = np.unravel_index(All_Impute_Inds,(Binarised_Input_Matrix.shape[0],Binarised_Input_Matrix.shape[1]))
        Track_Imputations[Imputation_Cycle] = All_Impute_Inds
        print("Finished")
        Track_Percentage_Imputation[0,Imputation_Cycle] = (Track_Imputations[Imputation_Cycle][0].shape[0]/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100
        Track_Percentage_Imputation[1,Imputation_Cycle] = (np.sum(Binarised_Input_Matrix[Track_Imputations[Imputation_Cycle]] == 0)/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100
        Track_Percentage_Imputation[2,Imputation_Cycle] = (np.sum(Binarised_Input_Matrix[Track_Imputations[Imputation_Cycle]] == 1)/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100
        if Auto_Save == 1:
            print("Saving Track_Imputations")
            np.save("Track_Imputations.npy",np.asarray(Track_Imputations,dtype=object))
            np.save("Combined_Informative_Genes.npy",np.asarray(Combined_Informative_Genes,dtype=object))
            np.save("Track_Cell_Uncertainties.npy",Track_Cell_Uncertainties)
            Imputation_Cycle = Imputation_Cycle + 1
            np.save("Track_Percentage_Imputation.npy",Track_Percentage_Imputation)
    print("Percentage of data suggested for imputation: " + str(np.round((Track_Imputations[Imputation_Cycle-1][0].shape[0]/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100,2)) + "%")      
    return np.asarray(Track_Imputations,dtype=object), Track_Percentage_Imputation, Track_Cell_Uncertainties


def FFAVES_Step_1(Min_Clust_Size,Divergences_Significance_Cut_Off,Use_Cores): 
    # Create Minority_Group_Matrix objects, Permutables and Switch_State_Inidicies objects.
    Permutables, Switch_State_Inidicies = Find_Permutations(Minority_Group_Matrix)
    # Switch Minority/Majority states to 0/1 where necessary.
    Minority_Group_Matrix[:,Switch_State_Inidicies] = (Minority_Group_Matrix[:,Switch_State_Inidicies] * -1) + 1  
    # Calculate minority group overlap matrix 
    Reference_Gene_Minority_Group_Overlaps = Parallel_Find_Minority_Group_Overlaps(Use_Cores)
    Permutables[Permutables < Min_Clust_Size] = np.nan
    print("Step 1: Identifying unreliable data points.")
    print("Calculating Divergence Matricies")
    Fixed_QG_Neg_SD_Divergences = Parallel_Fixed_QG_Neg_SD(Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores)    
    print("Identifying unreliable data points via half normal distribution")
    # Use half normal distribution of normalised divergent points to suggest which points should be re-evaluated
    Use_Inds = np.where(Minority_Group_Matrix != 0)
    Divergences = Fixed_QG_Neg_SD_Divergences[Use_Inds]    
    # Get zscores for observed divergences    
    zscores = zscore(Divergences)
    zscores = zscores + np.absolute(np.min(zscores))
    # Identify points that diverge in a statistically significant way
    Pass_Threshold = np.where(halfnorm.cdf(zscores) >= Divergences_Significance_Cut_Off)[0]
    Use_Inds = (Use_Inds[0][Pass_Threshold],Use_Inds[1][Pass_Threshold])
    # Measure Cell Uncertainties
    Fixed_QG_Neg_SD_Divergences[Minority_Group_Matrix == 0] = np.nan
    Cell_Uncertainties = np.nanmean(Fixed_QG_Neg_SD_Divergences,axis=1) 
    return Use_Inds, Cell_Uncertainties


def FFAVES_Step_2(Min_Clust_Size,Divergences_Significance_Cut_Off,Use_Cores): 
    # Create Minority_Group_Matrix objects, Permutables and Switch_State_Inidicies objects.
    Permutables, Switch_State_Inidicies = Find_Permutations(Minority_Group_Matrix)
    # Switch Minority/Majority states to 0/1 where necessary. 
    Minority_Group_Matrix[:,Switch_State_Inidicies] = (Minority_Group_Matrix[:,Switch_State_Inidicies] * -1) + 1
    # Calculate minority group overlap matrix
    Reference_Gene_Minority_Group_Overlaps = Parallel_Find_Minority_Group_Overlaps(Use_Cores)
    Permutables[Permutables < Min_Clust_Size] = np.nan
    print("Step 2: Identifying data points for imputation.")
    print("Calculating Divergence Matricies")   
    Fixed_QG_Pos_SD_Divergences, Informative_Genes = Parallel_Fixed_QG_Pos_SD(Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores)     
    print("Identifying data points for imputation via half normal distribution")
    # Use half normal distribution of normalised divergent points to suggest which points should be re-evaluated
    Use_Inds = np.where(Minority_Group_Matrix == 0)
    Divergences = (Fixed_QG_Pos_SD_Divergences)[Use_Inds]
    zscores = zscore(Divergences)
    zscores = zscores + np.absolute(np.min(zscores))              
    # Identify points that diverge in a statistically significant way
    Pass_Threshold = np.where(halfnorm.cdf(zscores) >= Divergences_Significance_Cut_Off)[0]
    Use_Inds = (Use_Inds[0][Pass_Threshold],Use_Inds[1][Pass_Threshold])
    return Use_Inds, Informative_Genes


def FFAVES_Step_3(Min_Clust_Size,Divergences_Significance_Cut_Off,Use_Cores):
    # Create Minority_Group_Matrix objects, Permutables and Switch_State_Inidicies objects.
    Permutables, Switch_State_Inidicies = Find_Permutations(Minority_Group_Matrix)
    # Switch Minority/Majority states to 0/1 where necessary.
    Minority_Group_Matrix[:,Switch_State_Inidicies] = (Minority_Group_Matrix[:,Switch_State_Inidicies] * -1) + 1
    # Calculate minority group overlap matrix
    Reference_Gene_Minority_Group_Overlaps = Parallel_Find_Minority_Group_Overlaps(Use_Cores)
    Permutables[Permutables < Min_Clust_Size] = np.nan
    print("Step 3: Cleaning up untrustworthy imputed values.")
    print("Calculating Divergence Matricies")
    Fixed_QG_Neg_SD_Divergences = Parallel_Fixed_QG_Neg_SD(Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores)            
    #Fixed_RG_Neg_SD_Divergences, Information_Gains_Matrix, Weights_Matrix = Parallel_Fixed_RG_Neg_SD(Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores)        
    print("Identifying unreliable imputed data points via half normal distribution")
    Use_Inds = np.where(Minority_Group_Matrix != 0)
    Divergences = Fixed_QG_Neg_SD_Divergences[Use_Inds]
    zscores = zscore(Divergences)
    zscores = zscores + np.absolute(np.min(zscores))    
    # Identify points that diverge in a statistically significant way
    Pass_Threshold = np.where(halfnorm.cdf(zscores) >= Divergences_Significance_Cut_Off)[0]
    Use_Inds = (Use_Inds[0][Pass_Threshold],Use_Inds[1][Pass_Threshold])
    return Use_Inds


### Here we have all of FFAVES subfunctions that are needed to calculate ES scores. ###

### Find the partition basis for each reference feature.
def Find_Permutations(Minority_Group_Matrix):
    Permutables = np.sum(Minority_Group_Matrix,axis=0)
    Switch_State_Inidicies = np.where(Permutables >= (Cell_Cardinality/2))[0]
    Permutables[Switch_State_Inidicies] = Cell_Cardinality - Permutables[Switch_State_Inidicies]  
    return Permutables, Switch_State_Inidicies


### Find minority group overlapping inds for each feature. Calculating this now streamlines future calculations
def Parallel_Find_Minority_Group_Overlaps(Use_Cores):
    Inds = np.arange(Gene_Cardinality)
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

### Parallel function calculated numerious Entropy Sorting properties while fixing a query gene (QG) as the central point of calculation.
## By fixing the QG, you are essentially asking are there points of the QG that are designated as being a member of the less common state of the QG
## that consistently overlap with expression states of the RG arrangments that are inconsistent with the sort direction of the ES curve. If a QG points
## consistently diverges from the structure in this way, the point should be changed from the less common state to the more common state of the QG. 
def Parallel_Fixed_QG_Neg_SD(Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores):
    Feature_Inds = np.arange(Gene_Cardinality)
    Pass_Info_To_Cores = np.concatenate((Feature_Inds.reshape(1,Feature_Inds.shape[0]),Reference_Gene_Minority_Group_Overlaps))
    Pass_Info_To_Cores = np.transpose(Pass_Info_To_Cores)
    # Parrallel calculate information gains matrix
    pool = multiprocessing.Pool(processes = Use_Cores)
    Result = pool.map(partial(Fixed_QG_Neg_SD, Cell_Cardinality=Cell_Cardinality,Permutables=Permutables), Pass_Info_To_Cores)
    pool.close()
    pool.join()
    # Retreive Information_Gain_Matrix
    # Retreive Fixed_QG_Neg_SD_Divergences and put the features back in the original feature ordering.
    Fixed_QG_Neg_SD_Divergences = np.concatenate(Result).T
    return Fixed_QG_Neg_SD_Divergences


### Calculate information gain matrix with fixed QG
def Fixed_QG_Neg_SD(Pass_Info_To_Cores,Cell_Cardinality,Permutables):
    # Extract which gene is being used as the Query Gene
    Feature_Inds = int(Pass_Info_To_Cores[0])
    # Remove the Query Gene ind from the data vector
    Reference_Gene_Minority_Group_Overlaps = np.delete(Pass_Info_To_Cores,0)
    if np.isnan(Permutables[Feature_Inds]) == 0:        
        ## Calculate Sorting Information for this fixed Query Gene
        with np.errstate(invalid='ignore'):
            Split_Directions, Split_Permute_Entropies, Max_Permuation_Entropies, Max_Entropy_Permutation = Calculate_QG_Sort_Values(Feature_Inds,Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Outputs=1)
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
        #Informative_Genes = np.where(RG_QG_Divergences > 0)[0]
        Sort_Out_Of_Query_Genes[:,Uninformative_Genes] = 0 
        # Mutiply the diverging cells by their average divergence and sum all the divergences for each QG/RG pair of each cell, to get a total
        # divergence for each cell.
        Ref_Fixed_QG_Neg_SD_Divergences = np.sum((Sort_Out_Of_Query_Genes*RG_QG_Divergences),axis=1)
    else:
        # When a feature cannot be used just give all points a value of 0.
        Ref_Fixed_QG_Neg_SD_Divergences = np.zeros(Cell_Cardinality)
    # Collate Results
    Results = []
    Results.append(Ref_Fixed_QG_Neg_SD_Divergences)
    # Output Results
    return Results


def Parallel_Fixed_QG_Pos_SD(Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores):
    Feature_Inds = np.arange(Gene_Cardinality)
    Pass_Info_To_Cores = np.concatenate((Feature_Inds.reshape(1,Feature_Inds.shape[0]),Reference_Gene_Minority_Group_Overlaps))
    Pass_Info_To_Cores = np.transpose(Pass_Info_To_Cores)
    # Parrallel calculate information gains matrix
    pool = multiprocessing.Pool(processes = Use_Cores)
    Result = pool.map(partial(Fixed_QG_Pos_SD, Cell_Cardinality=Cell_Cardinality,Permutables=Permutables), Pass_Info_To_Cores)
    pool.close()
    pool.join()
    Result = np.asarray(Result,dtype=object)
    # Retreive Information_Gain_Matrix
    # Retreive Fixed_QG_Pos_SD_Divergences and put the features back in the original feature ordering.
    Fixed_QG_Pos_SD_Divergences = np.stack(Result[:,0],axis=1)
    # Retreive Informative_Genes and put the features back in the original feature ordering.
    Informative_Genes = np.asarray(Result[:,1],dtype=object)
    return Fixed_QG_Pos_SD_Divergences, Informative_Genes


### Calculate information gain matrix with fixed QG
def Fixed_QG_Pos_SD(Pass_Info_To_Cores,Cell_Cardinality,Permutables):
    # Extract which gene is being used as the Reference Gene
    Feature_Inds = int(Pass_Info_To_Cores[0])
    # Remove the Reference Gene ind from the data vector
    Reference_Gene_Minority_Group_Overlaps = np.delete(Pass_Info_To_Cores,0)
    if np.isnan(Permutables[Feature_Inds]) == 0:        
        ## Calculate Sorting Information for this fixed Query Gene
        with np.errstate(invalid='ignore'):
            Split_Directions, Split_Permute_Entropies, Max_Permuation_Entropies, Max_Entropy_Permutation, Min_Entropy_ID_2, Minimum_Entropies = Calculate_QG_Sort_Values(Feature_Inds,Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Outputs=2)
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
        Divergences = np.zeros(Sort_Into_Divergence_Inds.shape[0])
        Permutable_Cardinality_Differences = Permutables[Sort_Into_Divergence_Inds] - Permutables[Feature_Inds]
        Use_Gap = np.where(Permutable_Cardinality_Differences >= 0)[0]
        Divergences[Use_Gap] = Split_Permute_Entropies[Sort_Into_Divergence_Inds][Use_Gap]
        Ignore_Gap = np.where(Permutable_Cardinality_Differences < 0)[0]
        Divergences[Ignore_Gap] = Split_Permute_Entropies[Sort_Into_Divergence_Inds][Ignore_Gap] - Minimum_Entropies[Sort_Into_Divergence_Inds][Ignore_Gap]
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
        Ref_Fixed_QG_Pos_SD_Divergences = np.sum((Sort_Into_Query_Genes*RG_QG_Divergences),axis=1)
        # Track Informative Genes
        Informative_Genes = Sort_Into_Divergence_Inds[Informative_Genes]
    else:
        # When a feature cannot be used just give all points a value of 0.
        Ref_Fixed_QG_Pos_SD_Divergences = np.zeros(Cell_Cardinality)
        Informative_Genes = []
    # Collate Results
    Results = []
    Results.append(Ref_Fixed_QG_Pos_SD_Divergences) 
    Results.append(Informative_Genes) 
    # Output Results
    return Results 


def Calculate_QG_Sort_Values(Feature_Inds,Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Outputs):
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
    if Outputs == 1:
        return Split_Directions, Split_Permute_Entropies, Max_Permuation_Entropies, Max_Entropy_Permutation
    if Outputs == 2:
        return Split_Directions, Split_Permute_Entropies, Max_Permuation_Entropies, Max_Entropy_Permutation, Min_Entropy_ID_2, Minimum_Entropies
    if Outputs == 3:
        return Information_Gains, Split_Weights, Split_Directions


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


#### Here we have additional functions to help with analysis of the results ####


def Calculate_ES_Sort_Matricies(Binarised_Input_Matrix, Track_Imputations, Min_Clust_Size = 5, Chosen_Cycle = -1, Use_Cores = -1, Auto_Save = 1):
    # Set number of cores to use
    Cores_Available = multiprocessing.cpu_count()
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Avaiblable: " + str(Cores_Available))
    print("Cores Used: " + str(Use_Cores))
    # Set up Minority_Group_Matrix
    global Minority_Group_Matrix
    Minority_Group_Matrix = copy.copy(Binarised_Input_Matrix)
    # Convert suggested imputation points to correct state.
    Suggested_Impute_Inds = Track_Imputations[Chosen_Cycle]
    Minority_Group_Matrix[Suggested_Impute_Inds] = (Minority_Group_Matrix[Suggested_Impute_Inds] - 1) * -1 
    # Create Minority_Group_Matrix objects, Permutables and Switch_State_Inidicies objects.
    Permutables, Switch_State_Inidicies = Find_Permutations(Minority_Group_Matrix)
    # Switch Minority/Majority states to 0/1 where necessary.
    Minority_Group_Matrix[:,Switch_State_Inidicies] = (Minority_Group_Matrix[:,Switch_State_Inidicies] * -1) + 1  
    # Calculate minority group overlap matrix 
    Reference_Gene_Minority_Group_Overlaps = Parallel_Find_Minority_Group_Overlaps(Use_Cores)
    Permutables[Permutables < Min_Clust_Size] = np.nan
    print("Step 1: Identifying unreliable data points.")
    print("Calculating Divergence Matricies")
    Information_Gains, Split_Weights = Parallel_Fixed_QG_Pos_SD_ES_Info(Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores)
    if Auto_Save == 1:
        np.save("Information_Gains.npy",Information_Gains)
        np.save("Split_Weights.npy",Split_Weights)
    return Information_Gains, Split_Weights


def Parallel_Fixed_QG_Pos_SD_ES_Info(Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores):
    Feature_Inds = np.arange(Gene_Cardinality)
    Pass_Info_To_Cores = np.concatenate((Feature_Inds.reshape(1,Feature_Inds.shape[0]),Reference_Gene_Minority_Group_Overlaps))
    Pass_Info_To_Cores = np.transpose(Pass_Info_To_Cores)
    # Parrallel calculate information gains matrix
    pool = multiprocessing.Pool(processes = Use_Cores)
    Result = pool.map(partial(Fixed_QG_Pos_SD_ES_Info, Cell_Cardinality=Cell_Cardinality,Permutables=Permutables), Pass_Info_To_Cores)
    pool.close()
    pool.join()
    Result = np.asarray(Result)
    # Retreive Information_Gain_Matrix
    # Retreive Information_Gains and put the features back in the original feature ordering.
    Information_Gains = Result[:,0]
    Information_Gains[np.isnan(Information_Gains)] = 0
    # Retreive Informative_Genes and put the features back in the original feature ordering.
    Split_Weights = Result[:,1]
    Split_Weights[np.isnan(Split_Weights)] = 0
    return Information_Gains, Split_Weights


### Calculate information gain matrix with fixed QG
def Fixed_QG_Pos_SD_ES_Info(Pass_Info_To_Cores,Cell_Cardinality,Permutables):
    # Extract which gene is being used as the Reference Gene
    Feature_Inds = int(Pass_Info_To_Cores[0])
    # Remove the Reference Gene ind from the data vector
    Reference_Gene_Minority_Group_Overlaps = np.delete(Pass_Info_To_Cores,0)
    if np.isnan(Permutables[Feature_Inds]) == 0:        
        ## Calculate Sorting Information for this fixed Query Gene
        with np.errstate(invalid='ignore'):
            Information_Gains, Split_Weights, Split_Directions = Calculate_QG_Sort_Values(Feature_Inds,Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Outputs=3)
    else:
        # When a feature cannot be used just give all points a value of 0.
        Information_Gains = np.zeros(Gene_Cardinality)
        Split_Weights = np.zeros(Gene_Cardinality)
    # Collate Results
    Results = []
    Results.append(Information_Gains*Split_Directions) 
    Results.append(Split_Weights) 
    # Output Results
    return Results 

#####


def Parallel_Optimise_Discretisation_Thresholds(Binarised_Input_Matrix,Track_Imputations,Chosen_Cycle = -1,Use_Cores=-1):
     # Set number of cores to use
    Cores_Available = multiprocessing.cpu_count()
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Avaiblable: " + str(Cores_Available))
    print("Cores Used: " + str(Use_Cores))
    # Set up Minority_Group_Matrix
    Imputed_Matrix = copy.copy(Binarised_Input_Matrix)
    # Convert suggested imputation points to correct state.
    Suggested_Impute_Inds = Track_Imputations[Chosen_Cycle]
    Imputed_Matrix[Suggested_Impute_Inds] = np.nan
    Paired = [[]] * Binarised_Input_Matrix.shape[1]
    for i in np.arange(Binarised_Input_Matrix.shape[1]):
        Paired[i] = np.stack((Binarised_Input_Matrix[:,i],Imputed_Matrix[:,i]))       
    pool = multiprocessing.Pool(processes = Use_Cores)
    Result = pool.map(Optimise_Discretisation_Thresholds, Paired)
    pool.close()
    pool.join()
    Result = np.asarray(Result,dtype=object)
    Thresholds = Result[:,0].astype("f")
    Imputations = np.stack(Result[:,1],axis=1)
    return Thresholds, Imputations


def Optimise_Discretisation_Thresholds(Paired):
    Original_Gene = Paired[0,:]
    Imputed_Gene = Paired[1,:]
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














