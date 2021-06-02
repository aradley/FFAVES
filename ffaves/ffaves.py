### Dependencies ###

import numpy as np
from functools import partial 
import multiprocessing
import copy
from scipy.stats import halfnorm, zscore

### Dependencies ###

### Here we have the FFAVES wrapper function that executes all the steps of FFAVES. ###

def FFAVES(Binarised_Input_Matrix, Min_Clust_Size = 5, Divergences_Significance_Cut_Off = 0.999, Use_Cores= -1, Num_Cycles = 1, Tolerance = 0.1, Auto_Save = 1):
    # Set number of cores to use
    Cores_Available = multiprocessing.cpu_count()
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Avaiblable: " + str(Cores_Available))
    print("Cores Used: " + str(Use_Cores))
    # Remove genes below Min_Clust_Size
    Keep_Features = np.where(np.sum(Binarised_Input_Matrix,axis=0) >= Min_Clust_Size)[0]
    Ignore = np.where(np.sum(Binarised_Input_Matrix,axis=0) >= (Binarised_Input_Matrix.shape[1]-Min_Clust_Size))[0] 
    Keep_Features = np.delete(Keep_Features,np.where(np.isin(Keep_Features,Ignore))[0])
    if Keep_Features.shape[0] < Binarised_Input_Matrix.shape[1]:
        print("Ignoring " + str(Binarised_Input_Matrix.shape[1]-Keep_Features.shape[0]) + " features which are below the Min_Clust_Threshold")
        Binarised_Input_Matrix = Binarised_Input_Matrix[:,Keep_Features]
    # Define data dimensions
    #global Cell_Cardinality
    Cell_Cardinality = Binarised_Input_Matrix.shape[0]
    #global Gene_Cardinality
    Gene_Cardinality = Binarised_Input_Matrix.shape[1]
    # Convert Binarised_Input_Matrix into Minority_Group_Matrix
    Permutables, Switch_State_Inidicies = Find_Permutations(Binarised_Input_Matrix,Cell_Cardinality)
    Binarised_Input_Matrix[:,Switch_State_Inidicies] = (Binarised_Input_Matrix[:,Switch_State_Inidicies] * -1) + 1  
    # Set up Minority_Group_Matrix
    global Minority_Group_Matrix
    # Track what cycle FFAVES is on.
    Imputation_Cycle = 1
    print("Number of cells: " + str(Cell_Cardinality))
    print("Number of genes: " + str(Gene_Cardinality))
    Track_Percentage_Imputation = np.zeros((3,Num_Cycles+1))
    Track_Imputation_Steps = np.empty(((Num_Cycles + 1),),dtype="object")
    Track_Imputation_Steps[...]=[([[]]*3) for _ in range((Num_Cycles + 1))]  
    # Cell Uncertainties
    Track_Cell_Uncertainties = np.zeros((Num_Cycles,Cell_Cardinality))
    # Set up tolerance tracker
    Covergence_Counter = 0
    All_Impute_Inds = (np.array([]).astype("i"),np.array([]).astype("i"))
    while Imputation_Cycle <= Num_Cycles and Covergence_Counter < 3:
        if Imputation_Cycle > 1:
            print("Percentage of original data suggested as Type 1 Error: " + str(np.round((Track_Imputation_Steps[Imputation_Cycle-1][0][0].shape[0]/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100,2)) + "%")
            print("Percentage of original data suggested as Type 2 Error: " + str(np.round((Track_Imputation_Steps[Imputation_Cycle-1][2][0].shape[0]/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100,2)) + "%")   
        # Initiate Cycle_Imputation_Steps
        Cycle_Imputation_Steps = [[]] * 3
        # Initiate State_Inversions
        State_Inversions = np.zeros(Gene_Cardinality)
        # Initiate Cycle_Cell_Uncertainties
        Cycle_Cell_Uncertainties = np.zeros(Cell_Cardinality)
        print("Cycle Number " + str(Imputation_Cycle))         
        # Convert suggested imputation points from previous cycle to correct state.
        Minority_Group_Matrix = copy.copy(Binarised_Input_Matrix)
        Minority_Group_Matrix[All_Impute_Inds] = (Minority_Group_Matrix[All_Impute_Inds] - 1) * -1
        ### Step 1 of FFAVES is to identify and temporarily remove spurious Minority Group expression states
        with np.errstate(divide='ignore',invalid='ignore'):
            Step_1_Type_1_Error_Inds, Switch_State_Inidicies_1, Cell_Uncertainties = FFAVES_Step_1(Min_Clust_Size,Divergences_Significance_Cut_Off,Use_Cores,Cell_Cardinality,Gene_Cardinality)
        # Track Cell_Uncertainties
        Cycle_Cell_Uncertainties = Cycle_Cell_Uncertainties + Cell_Uncertainties
        #
        State_Inversions[Switch_State_Inidicies_1] = (State_Inversions[Switch_State_Inidicies_1] * -1) + 1
        State_Inversion_Inds = np.where(State_Inversions == 1)[0]
        Flat_All_Impute_Inds = np.ravel_multi_index(All_Impute_Inds, Binarised_Input_Matrix.shape)
        Flat_Step_1_Inds = np.ravel_multi_index(Step_1_Type_1_Error_Inds, Binarised_Input_Matrix.shape)
        Void_Type_1_Errors = np.where(np.isin(Step_1_Type_1_Error_Inds[1],State_Inversion_Inds)==0)[0]
        Impute_Type_1_Errors = np.where(np.isin(Step_1_Type_1_Error_Inds[1],State_Inversion_Inds)==1)[0]
        if Void_Type_1_Errors.shape[0] > 0:
            Void_Type_1_Errors = np.ravel_multi_index((Step_1_Type_1_Error_Inds[0][Void_Type_1_Errors],Step_1_Type_1_Error_Inds[1][Void_Type_1_Errors]), Binarised_Input_Matrix.shape)
            Ignore_Imputations = np.where(np.isin(Flat_All_Impute_Inds,Flat_Step_1_Inds))[0]
            Flat_All_Impute_Inds = np.delete(Flat_All_Impute_Inds,Ignore_Imputations)       
        if Impute_Type_1_Errors.shape[0] > 0:
            Impute_Type_1_Errors = np.ravel_multi_index((Step_1_Type_1_Error_Inds[0][Impute_Type_1_Errors],Step_1_Type_1_Error_Inds[1][Impute_Type_1_Errors]), Binarised_Input_Matrix.shape)
            Flat_All_Impute_Inds = np.unique(np.append(Flat_All_Impute_Inds,Impute_Type_1_Errors))
        All_Impute_Inds = np.unravel_index(Flat_All_Impute_Inds,Binarised_Input_Matrix.shape)
        Minority_Group_Matrix[Step_1_Type_1_Error_Inds] = (Minority_Group_Matrix[Step_1_Type_1_Error_Inds] - 1) * -1
        ### Step 2 of FFAVES is to identify which majority states points are spurious
        with np.errstate(divide='ignore',invalid='ignore'):
            Type_2_Error_Inds, Switch_State_Inidicies_2, Cell_Uncertainties, Average_Imputed_Divergence = FFAVES_Step_2(Min_Clust_Size,Divergences_Significance_Cut_Off,Use_Cores,Cell_Cardinality,Gene_Cardinality)        
        # Track Cell_Uncertainties
        Cycle_Cell_Uncertainties = Cycle_Cell_Uncertainties + Cell_Uncertainties
        #
        State_Inversions[Switch_State_Inidicies_2] = (State_Inversions[Switch_State_Inidicies_2] * -1) + 1
        State_Inversion_Inds = np.where(State_Inversions == 1)[0]
        Flat_All_Impute_Inds = np.ravel_multi_index(All_Impute_Inds, Binarised_Input_Matrix.shape)
        Flat_Step_2_Inds = np.ravel_multi_index(Type_2_Error_Inds, Binarised_Input_Matrix.shape)
        Void_Type_2_Errors = np.where(np.isin(Type_2_Error_Inds[1],State_Inversion_Inds)==1)[0]
        Impute_Type_2_Errors = np.where(np.isin(Type_2_Error_Inds[1],State_Inversion_Inds)==0)[0]
        if Void_Type_2_Errors.shape[0] > 0:
            Void_Type_2_Errors = np.ravel_multi_index((Type_2_Error_Inds[0][Void_Type_2_Errors],Type_2_Error_Inds[1][Void_Type_2_Errors]), Binarised_Input_Matrix.shape)
            Ignore_Imputations = np.where(np.isin(Flat_All_Impute_Inds,Flat_Step_2_Inds))[0]
            Flat_All_Impute_Inds = np.delete(Flat_All_Impute_Inds,Ignore_Imputations) 
        if Impute_Type_2_Errors.shape[0] > 0:
            Impute_Type_2_Errors = np.ravel_multi_index((Type_2_Error_Inds[0][Impute_Type_2_Errors],Type_2_Error_Inds[1][Impute_Type_2_Errors]), Binarised_Input_Matrix.shape)
            Flat_All_Impute_Inds = np.unique(np.append(Flat_All_Impute_Inds,Impute_Type_2_Errors))
        All_Impute_Inds = np.unravel_index(Flat_All_Impute_Inds,Binarised_Input_Matrix.shape)
        ### Track Erroneous points
        Cycle_Imputation_Steps[0] = Step_1_Type_1_Error_Inds
        Cycle_Imputation_Steps[1] = Type_2_Error_Inds
        Cycle_Imputation_Steps[2] = All_Impute_Inds
        Track_Imputation_Steps[Imputation_Cycle] = Cycle_Imputation_Steps
        Track_Cell_Uncertainties[(Imputation_Cycle-1),:] = Cycle_Cell_Uncertainties 
        print("Finished")
        Track_Percentage_Imputation[0,Imputation_Cycle] = (Track_Imputation_Steps[Imputation_Cycle][0][0].shape[0]/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100
        Track_Percentage_Imputation[1,Imputation_Cycle] = (Track_Imputation_Steps[Imputation_Cycle][2][0].shape[0]/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100
        Track_Percentage_Imputation[2,Imputation_Cycle] = Average_Imputed_Divergence
        if Imputation_Cycle < Num_Cycles:
            Imputed_Difference = np.sum(np.absolute(Track_Percentage_Imputation[0:2,Imputation_Cycle] - Track_Percentage_Imputation[0:2,Imputation_Cycle-1]))
            if Imputed_Difference <= Tolerance:
                Covergence_Counter = Covergence_Counter + 1
            else:
                Covergence_Counter = 0
        Imputation_Cycle = Imputation_Cycle + 1
    if Imputation_Cycle < Num_Cycles:
        Track_Imputation_Steps = Track_Imputation_Steps[0:Imputation_Cycle]
        Track_Percentage_Imputation = Track_Percentage_Imputation[:,0:Imputation_Cycle]
        Track_Cell_Uncertainties = Track_Cell_Uncertainties[0:(Imputation_Cycle-1),:]
    # Re-align inds to original data prior to Min_Clust_Size subsetting
    for i in np.arange(1,Track_Imputation_Steps.shape[0]):
        Track_Imputation_Steps[i][0] = (Track_Imputation_Steps[i][0][0],Keep_Features[Track_Imputation_Steps[i][0][1]])
        Track_Imputation_Steps[i][1] = (Track_Imputation_Steps[i][1][0],Keep_Features[Track_Imputation_Steps[i][1][1]])
        Track_Imputation_Steps[i][2] = (Track_Imputation_Steps[i][2][0],Keep_Features[Track_Imputation_Steps[i][2][1]])
    if Auto_Save == 1:
        np.save("Track_Imputation_Steps.npy",Track_Imputation_Steps)
        np.save("Track_Cell_Uncertainties.npy",Track_Cell_Uncertainties)
        np.save("Track_Percentage_Imputation.npy",Track_Percentage_Imputation)
    return Track_Imputation_Steps, Track_Percentage_Imputation, Track_Cell_Uncertainties


def FFAVES_Step_1(Min_Clust_Size,Divergences_Significance_Cut_Off,Use_Cores,Cell_Cardinality,Gene_Cardinality):
    print("Step 1: Quantifying Type 1 Error for each data point.")
    print("Identifying Sort Info for calculations.") 
    # Create Minority_Group_Matrix objects, Permutables and Switch_State_Inidicies objects.
    Permutables, Switch_State_Inidicies = Find_Permutations(Minority_Group_Matrix,Cell_Cardinality)
    # Switch Minority/Majority states to 0/1 where necessary.
    Minority_Group_Matrix[:,Switch_State_Inidicies] = (Minority_Group_Matrix[:,Switch_State_Inidicies] * -1) + 1  
    # Calculate minority group overlap matrix 
    Reference_Gene_Minority_Group_Overlaps = Parallel_Find_Minority_Group_Overlaps(Use_Cores,Gene_Cardinality)
    Permutables[Permutables < Min_Clust_Size] = np.nan
    print("Calculating Divergence Matrix.")
    Type_1_Error_Divergences = Parallel_Calculate_Cell_Divergences("1",Cell_Cardinality,Gene_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores)
    Type_1_Error_Inds_1, Cell_Uncertainties_1_1 = Extract_Divergence_Info("1", Type_1_Error_Divergences, Divergences_Significance_Cut_Off)
    Type_1_Error_Inds_1 = np.unravel_index(Type_1_Error_Inds_1,Minority_Group_Matrix.shape)
    Cell_Uncertainties = Cell_Uncertainties_1_1
    return Type_1_Error_Inds_1, Switch_State_Inidicies, Cell_Uncertainties


def FFAVES_Step_2(Min_Clust_Size,Divergences_Significance_Cut_Off,Use_Cores,Cell_Cardinality,Gene_Cardinality): 
    print("Step 2: Quantifying Type 2 Error for each data point.")
    print("Identifying Sort Info for calculations.")
    # Create Minority_Group_Matrix objects, Permutables and Switch_State_Inidicies objects.
    Permutables, Switch_State_Inidicies = Find_Permutations(Minority_Group_Matrix,Cell_Cardinality)
    # Switch Minority/Majority states to 0/1 where necessary. 
    Minority_Group_Matrix[:,Switch_State_Inidicies] = (Minority_Group_Matrix[:,Switch_State_Inidicies] * -1) + 1
    # Calculate minority group overlap matrix
    Reference_Gene_Minority_Group_Overlaps = Parallel_Find_Minority_Group_Overlaps(Use_Cores,Gene_Cardinality)
    Permutables[Permutables < Min_Clust_Size] = np.nan
    print("Calculating Divergence Matrix.")   
    Type_2_Error_Divergences = Parallel_Calculate_Cell_Divergences("2",Cell_Cardinality,Gene_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores)
    Type_2_Error_Inds, Cell_Uncertainties = Extract_Divergence_Info("2", Type_2_Error_Divergences, Divergences_Significance_Cut_Off)
    Type_2_Error_Inds = np.unravel_index(Type_2_Error_Inds,Minority_Group_Matrix.shape)
    Average_Imputed_Divergence = np.mean(Type_2_Error_Divergences[Type_2_Error_Inds])
    return Type_2_Error_Inds, Switch_State_Inidicies, Cell_Uncertainties, Average_Imputed_Divergence


def Track_Changes(Switch_State_Inidicies_1,Switch_State_Inidicies_2,Binarised_Input_Matrix):
    # Flip genes states back where required
    Minority_Group_Matrix[:,Switch_State_Inidicies_1] = (Minority_Group_Matrix[:,Switch_State_Inidicies_1] * -1) + 1
    Minority_Group_Matrix[:,Switch_State_Inidicies_2] = (Minority_Group_Matrix[:,Switch_State_Inidicies_2] * -1) + 1
    # Identify what has changed compared to initial data
    State_Changes = Binarised_Input_Matrix - Minority_Group_Matrix
    # Identify spurious active expression points
    False_Positive_Inds = np.where(State_Changes == 1)
    # Identify spurious inactive expression points
    False_Negative_Inds = np.where(State_Changes == -1)
    # Track Cell Uncertainties
    Cell_Uncertainties = np.sum(State_Changes != 0,axis=1)
    return False_Positive_Inds, False_Negative_Inds, Cell_Uncertainties


### Here we have all of FFAVES subfunctions that are needed to calculate ES scores. ###

### Find the partition basis for each reference feature.
def Find_Permutations(Minority_Group_Matrix,Cell_Cardinality):
    Permutables = np.sum(Minority_Group_Matrix,axis=0).astype("f")
    Switch_State_Inidicies = np.where(Permutables >= (Cell_Cardinality/2))[0]
    Permutables[Switch_State_Inidicies] = Cell_Cardinality - Permutables[Switch_State_Inidicies]  
    return Permutables, Switch_State_Inidicies


def Extract_Divergence_Info(Error_Type, Error_Divergences, Divergences_Significance_Cut_Off):
    Cell_Uncertainties = np.sum(Error_Divergences,axis=1)
    # Use half normal distribution of normalised divergent points to suggest which points should be re-evaluated
    if Error_Type != "2":
        Use_Inds = np.where(Minority_Group_Matrix != 0)
    else:
        Use_Inds = np.where(Minority_Group_Matrix == 0)
    Divergences = Error_Divergences[Use_Inds]
    # Get zscores for observed divergences    
    zscores = zscore(Divergences)
    zscores = zscores + np.absolute(np.min(zscores))
    # Identify points that diverge in a statistically significant way
    Pass_Threshold = np.where(halfnorm.cdf(zscores) >= Divergences_Significance_Cut_Off)[0]
    Error_Inds = (Use_Inds[0][Pass_Threshold],Use_Inds[1][Pass_Threshold])
    Error_Inds = np.ravel_multi_index(Error_Inds, Minority_Group_Matrix.shape)
    return Error_Inds, Cell_Uncertainties


### Find minority group overlapping inds for each feature. Calculating this now streamlines future calculations
def Parallel_Find_Minority_Group_Overlaps(Use_Cores,Gene_Cardinality):
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
    Overlaps = np.dot(Reference_Gene,Minority_Group_Matrix)
    return Overlaps


def Parallel_Calculate_Cell_Divergences(Error_Type,Cell_Cardinality,Gene_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores):
    Feature_Inds = np.arange(Gene_Cardinality)
    Pass_Info_To_Cores = np.concatenate((Feature_Inds.reshape(1,Feature_Inds.shape[0]),Reference_Gene_Minority_Group_Overlaps))
    Pass_Info_To_Cores = np.transpose(Pass_Info_To_Cores)
    # Parrallel calculate information gains matrix
    pool = multiprocessing.Pool(processes = Use_Cores)
    Result = pool.map(partial(Calculate_Cell_Divergences,Error_Type=Error_Type,Cell_Cardinality=Cell_Cardinality,Permutables=Permutables), Pass_Info_To_Cores)
    pool.close()
    pool.join()
    if Error_Type != "2_Intentional_Error":
        Divergence_Matrix = np.stack(Result,axis=1)
    else:
        Divergence_Matrix = np.asarray(Result)
    return Divergence_Matrix

### Calculate Divergence Matrix
def Calculate_Cell_Divergences(Pass_Info_To_Cores,Error_Type,Cell_Cardinality,Permutables):   
    ####################
    #Pass_Info_To_Cores_Temp = Pass_Info_To_Cores[40,:]
    ######################
    # Extract which gene calculations are centred around
    Feature_Inds = int(Pass_Info_To_Cores[0])
    if np.isnan(Permutables[Feature_Inds]) == 0:
        # Extract the gene
        Gene_States = Minority_Group_Matrix[:,Feature_Inds]
        Fixed_Gene_Minority_States = np.where(Gene_States == 1)[0]
        Fixed_Gene_Majority_States = np.where(Gene_States == 0)[0]
        # Remove the Query Gene ind from the data vector
        Reference_Gene_Minority_Group_Overlaps = np.delete(Pass_Info_To_Cores,0)
        # Note which RG features cannot be used (probably because their minority group cardinality does not meet the Min_Clust_Size threshold)
        if Error_Type == "1": # Caluclate divergences for each Type 1 (false positive) error scenarios.
            ##### Get Inspected Gene Information #####
            # Extract the group 1 and group 2 cardinalities. Group 1 is always the minority group in this set up.
            Minority_Group_Cardinality = Permutables[Feature_Inds]
            Majority_Group_Cardinality = Cell_Cardinality - Minority_Group_Cardinality
            # Identify in which scenarios the inspected gene is the Partitioning Gene and which it is the Permutable Gene
            # Whenever the inspected gene minority state cardinality is smaller than any other gene's, it is the RG
            Partitioning_Inds = np.where(Permutables >= Minority_Group_Cardinality)[0]
            # Whenever the inspected gene minority state cardinality is larger than any other gene's, it is the QG
            Permuting_Inds = np.where(Permutables < Minority_Group_Cardinality)[0]
            ##### Inspected Gene is RG Caclulations #####
            # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESE)
            Max_Entropy_Permutation = (Minority_Group_Cardinality * Permutables[Partitioning_Inds])/(Minority_Group_Cardinality + Majority_Group_Cardinality)
            # The maximum and minimum points of the ESE are identified from the boundaries of the ESE curve.
            Min_Entropy_ID_1 = np.zeros(Partitioning_Inds.shape[0])
            Min_Entropy_ID_2 = copy.copy(Permutables[Partitioning_Inds])
            Check_Fits_Group_1 = Minority_Group_Cardinality - Min_Entropy_ID_2
            # If the minority group of the QG is larger than the minority group of the RG then the boundary point is the cardinality of the RG minority group.
            Min_Entropy_ID_2[np.where(Check_Fits_Group_1 < 0)[0]] = Minority_Group_Cardinality #### This may be redundent since we pre-select curves
            # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
            Split_Permute_Value = Reference_Gene_Minority_Group_Overlaps[Partitioning_Inds]
            # Num_Divergent_Cell
            Num_Divergent_Cell = Minority_Group_Cardinality - Split_Permute_Value
            # Identify Split Direction (whether the observed arrangment is sorting towards the global minimum entropy or not. I.e. when
            # can type 1 error occour?
            Sort_Out_Of_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) < 0)[0]
            Split_Direction = -1
            Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies = Calculate_Fixed_RG_Sort_Values(2,Split_Direction,Max_Entropy_Permutation[Sort_Out_Of_Inds],Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables[Partitioning_Inds[Sort_Out_Of_Inds]],Min_Entropy_ID_1[Sort_Out_Of_Inds],Min_Entropy_ID_2[Sort_Out_Of_Inds],Split_Permute_Value[Sort_Out_Of_Inds])
            ## Calculate Divergence Information
            Divergences = Split_Permute_Entropies - Minimum_Entropies
            # Find the average divergence for each cell that is diverging from the optimal sort.
            Cell_Divergences = Divergences / Split_Permute_Value[Sort_Out_Of_Inds]
            # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
            Max_Num_Cell_Divergences = Min_Entropy_ID_2[Sort_Out_Of_Inds] - Max_Entropy_Permutation[Sort_Out_Of_Inds]
            Minimum_Background_Noise = (Max_Permuation_Entropies)/Max_Num_Cell_Divergences
            # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
            RG_QG_Divergences = Cell_Divergences - Minimum_Background_Noise
            # Null/Ignore points that aren't usable.
            RG_QG_Divergences[np.isinf(RG_QG_Divergences)] = 0
            RG_QG_Divergences[np.isnan(RG_QG_Divergences)] = 0
            # Featues whose RG_QG_Divergences are less than 0 would add more entropy to the system per data point imputed,
            # hence ignore them.
            Informative_Genes = np.where(RG_QG_Divergences >= 0)[0]
            # Get Overlap Matrix
            Sort_Genes = Minority_Group_Matrix[np.ix_(Fixed_Gene_Minority_States,Partitioning_Inds[Sort_Out_Of_Inds][Informative_Genes])]
            Sort_Out_Of_Partitioning_Divergences = np.zeros(Cell_Cardinality)
            Sort_Out_Of_Partitioning_Divergences[Fixed_Gene_Minority_States] = np.dot(Sort_Genes,RG_QG_Divergences[Informative_Genes])        
            ##### Inspected Gene is RG Caclulations #####
            # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESE)
            Max_Entropy_Permutation = (Minority_Group_Cardinality * Permutables[Partitioning_Inds])/(Minority_Group_Cardinality + Majority_Group_Cardinality)
            # The maximum and minimum points of the ESE are identified from the boundaries of the ESE curve.
            Min_Entropy_ID_1 = np.zeros(Partitioning_Inds.shape[0])
            Min_Entropy_ID_2 = copy.copy(Permutables[Partitioning_Inds])
            Check_Fits_Group_1 = Minority_Group_Cardinality - Min_Entropy_ID_2
            # If the minority group of the QG is larger than the minority group of the RG then the boundary point is the cardinality of the RG minority group.
            Min_Entropy_ID_2[np.where(Check_Fits_Group_1 < 0)[0]] = Minority_Group_Cardinality #### This may be redundent since we pre-select curves
            # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
            Split_Permute_Value = Reference_Gene_Minority_Group_Overlaps[Partitioning_Inds]
            # Num_Divergent_Cell
            Num_Divergent_Cell = Minority_Group_Cardinality - Split_Permute_Value
            # Identify Split Direction (whether the observed arrangment is sorting towards the global minimum entropy or not. I.e. when
            # can type 1 error occour?
            Sort_Into_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) >= 0)[0]
            Split_Direction = 1
            Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies = Calculate_Fixed_RG_Sort_Values(2,Split_Direction,Max_Entropy_Permutation[Sort_Into_Inds],Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables[Partitioning_Inds[Sort_Into_Inds]],Min_Entropy_ID_1[Sort_Into_Inds],Min_Entropy_ID_2[Sort_Into_Inds],Split_Permute_Value[Sort_Into_Inds])
            ## Calculate Divergence Information
            Divergences = Split_Permute_Entropies - Minimum_Entropies
            # Find the average divergence for each cell that is diverging from the optimal sort.
            Cell_Divergences = Divergences / Num_Divergent_Cell[Sort_Into_Inds]
            # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
            Max_Num_Cell_Divergences = Min_Entropy_ID_2[Sort_Into_Inds] - Max_Entropy_Permutation[Sort_Into_Inds]
            Minimum_Background_Noise = (Max_Permuation_Entropies)/Max_Num_Cell_Divergences
            # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
            RG_QG_Divergences = Cell_Divergences - Minimum_Background_Noise
            # Null/Ignore points that aren't usable.
            RG_QG_Divergences[np.isinf(RG_QG_Divergences)] = 0
            RG_QG_Divergences[np.isnan(RG_QG_Divergences)] = 0
            # Featues whose RG_QG_Divergences are less than 0 would add more entropy to the system per data point imputed,
            # hence ignore them.
            Informative_Genes = np.where(RG_QG_Divergences >= 0)[0]
            # Get Overlap Matrix
            Sort_Genes = Minority_Group_Matrix[np.ix_(Fixed_Gene_Minority_States,Partitioning_Inds[Sort_Into_Inds][Informative_Genes])]
            Sort_Into_Partitioning_Divergences = np.zeros(Cell_Cardinality)
            Sort_Into_Partitioning_Divergences[Fixed_Gene_Minority_States] = np.dot(Sort_Genes==0,RG_QG_Divergences[Informative_Genes])                      
            ##### Inspected Gene is QG Caclulations #####
            # Extract the group 1 and group 2 cardinalities. Group 1 is always the minority group in this set up.
            Minority_Group_Cardinality = Permutables[Permuting_Inds]
            Majority_Group_Cardinality = Cell_Cardinality - Permutables[Permuting_Inds]
            Permutable = Permutables[Feature_Inds]
            # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESQ)
            Max_Entropy_Permutation = (Minority_Group_Cardinality * Permutable)/(Minority_Group_Cardinality + Majority_Group_Cardinality)
            # The maximum and minimum points of the ESQ are identified from the boundaries of the ESQ curve.
            Min_Entropy_ID_1 = np.zeros(Permuting_Inds.shape[0])
            Min_Entropy_ID_2 = np.repeat(Permutable,Permuting_Inds.shape[0])
            Check_Fits_Group_1 = Minority_Group_Cardinality - Min_Entropy_ID_2
            # If the minority group of the QG is larger than the minority group of the RG then the boundary point is the cardinality of the RG minority group.
            Min_Entropy_ID_2[np.where(Check_Fits_Group_1 < 0)[0]] = Minority_Group_Cardinality[np.where(Check_Fits_Group_1 < 0)[0]]
            # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
            Split_Permute_Value = Reference_Gene_Minority_Group_Overlaps[Permuting_Inds]
            # Num_Divergent_Cell
            Num_Divergent_Cell = Minority_Group_Cardinality - Split_Permute_Value            
            # Identify Split Direction (whether the observed arrangment is sorting towards the global minimum entropy or not. I.e. when
            # can type 1 error occour?
            Sort_Out_Of_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) < 0)[0]
            Split_Direction = -1
            Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies = Calculate_Fixed_QG_Sort_Values(2,Split_Direction,Permutable,Max_Entropy_Permutation[Sort_Out_Of_Inds],Minority_Group_Cardinality[Sort_Out_Of_Inds],Majority_Group_Cardinality[Sort_Out_Of_Inds],Min_Entropy_ID_1[Sort_Out_Of_Inds],Min_Entropy_ID_2[Sort_Out_Of_Inds],Split_Permute_Value[Sort_Out_Of_Inds])
            ## Calculate Divergence Information
            # In scenario 6 we include the gap
            Divergences = Split_Permute_Entropies - Minimum_Entropies
            # Find the average divergence for each cell that is diverging from the optimal sort.
            Cell_Divergences = Divergences / Split_Permute_Value[Sort_Out_Of_Inds]
            # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
            Max_Num_Cell_Divergences = Max_Entropy_Permutation[Sort_Out_Of_Inds]
            Minimum_Background_Noise = (Max_Permuation_Entropies)/Max_Num_Cell_Divergences
            # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
            RG_QG_Divergences = Cell_Divergences - Minimum_Background_Noise
            # Null/Ignore points that aren't usable.
            RG_QG_Divergences[np.isinf(RG_QG_Divergences)] = 0
            RG_QG_Divergences[np.isnan(RG_QG_Divergences)] = 0
            # Featues whose RG_QG_Divergences are less than 0 would add more entropy to the system per data point imputed,
            # hence ignore them.
            Informative_Genes = np.where(RG_QG_Divergences >= 0)[0]
            # Get Overlap Matrix
            Sort_Genes = Minority_Group_Matrix[np.ix_(Fixed_Gene_Minority_States,Permuting_Inds[Sort_Out_Of_Inds][Informative_Genes])]
            Sort_Out_Of_Permuting_Divergences = np.zeros(Cell_Cardinality)
            Sort_Out_Of_Permuting_Divergences[Fixed_Gene_Minority_States] = np.dot(Sort_Genes,RG_QG_Divergences[Informative_Genes])
            return Sort_Out_Of_Partitioning_Divergences + Sort_Into_Partitioning_Divergences + Sort_Out_Of_Permuting_Divergences
        if Error_Type == "2": # Caluclate divergences for each Type 2 (false negative) error scenarios.
            ##### Get Inspected Gene Information #####
            # Extract the group 1 and group 2 cardinalities. Group 1 is always the minority group in this set up.
            Minority_Group_Cardinality = Permutables[Feature_Inds]
            Majority_Group_Cardinality = Cell_Cardinality - Minority_Group_Cardinality
            # Identify in which scenarios the inspected gene is the Partitioning Gene and which it is the Permutable Gene
            # Whenever the inspected gene minority state cardinality is smaller than any other gene's, it is the RG
            Partitioning_Inds = np.where(Permutables >= Minority_Group_Cardinality)[0]
            # Whenever the inspected gene minority state cardinality is larger than any other gene's, it is the QG
            Permuting_Inds = np.where(Permutables < Minority_Group_Cardinality)[0]
            ##### Inspected Gene is QG Caclulations #####
            # Extract the group 1 and group 2 cardinalities. Group 1 is always the minority group in this set up.
            Minority_Group_Cardinality = Permutables[Permuting_Inds]
            Majority_Group_Cardinality = Cell_Cardinality - Permutables[Permuting_Inds]
            Permutable = Permutables[Feature_Inds]
            # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESQ)
            Max_Entropy_Permutation = (Minority_Group_Cardinality * Permutable)/(Minority_Group_Cardinality + Majority_Group_Cardinality)
            # The maximum and minimum points of the ESQ are identified from the boundaries of the ESQ curve.
            Min_Entropy_ID_1 = np.zeros(Permuting_Inds.shape[0])
            Min_Entropy_ID_2 = np.repeat(Permutable,Permuting_Inds.shape[0])
            Check_Fits_Group_1 = Minority_Group_Cardinality - Min_Entropy_ID_2
            # If the minority group of the QG is larger than the minority group of the RG then the boundary point is the cardinality of the RG minority group.
            Min_Entropy_ID_2[np.where(Check_Fits_Group_1 < 0)[0]] = Minority_Group_Cardinality[np.where(Check_Fits_Group_1 < 0)[0]]
            # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
            Split_Permute_Value = Reference_Gene_Minority_Group_Overlaps[Permuting_Inds]
            # Num_Divergent_Cell
            Num_Divergent_Cell = Minority_Group_Cardinality - Split_Permute_Value
            # Identify Split Direction (whether the observed arrangment is sorting towards the global minimum entropy or not. I.e. is the QG sorting into the
            # minority or majority group of the RG.)
            Sort_Into_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) >= 0)[0]
            Split_Direction = 1
            Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies = Calculate_Fixed_QG_Sort_Values(2,Split_Direction,Permutable,Max_Entropy_Permutation[Sort_Into_Inds],Minority_Group_Cardinality[Sort_Into_Inds],Majority_Group_Cardinality[Sort_Into_Inds],Min_Entropy_ID_1[Sort_Into_Inds],Min_Entropy_ID_2[Sort_Into_Inds],Split_Permute_Value[Sort_Into_Inds])
            ## Calculate Divergence Information
            # In 
            Divergences = Split_Permute_Entropies - Minimum_Entropies
            # Find the average divergence for each cell that is diverging from the optimal sort.
            Cell_Divergences = Divergences / Num_Divergent_Cell[Sort_Into_Inds]
            # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
            Max_Num_Cell_Divergences = Min_Entropy_ID_2[Sort_Into_Inds] - Max_Entropy_Permutation[Sort_Into_Inds]
            Minimum_Background_Noise = (Max_Permuation_Entropies)/Max_Num_Cell_Divergences
            # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
            RG_QG_Divergences = Cell_Divergences - Minimum_Background_Noise
            # Null/Ignore points that aren't usable.
            RG_QG_Divergences[np.isinf(RG_QG_Divergences)] = 0
            RG_QG_Divergences[np.isnan(RG_QG_Divergences)] = 0
            # Featues whose RG_QG_Divergences are less than 0 would add more entropy to the system per data point imputed,
            # hence ignore them.
            Informative_Genes = np.where(RG_QG_Divergences >= 0)[0]
            # Get Overlap Matrix
            Sort_Genes = Minority_Group_Matrix[np.ix_(Fixed_Gene_Majority_States,Permuting_Inds[Sort_Into_Inds][Informative_Genes])]
            Sort_Into_Permuting_Divergences = np.zeros(Cell_Cardinality)
            Sort_Into_Permuting_Divergences[Fixed_Gene_Majority_States] = np.dot(Sort_Genes,RG_QG_Divergences[Informative_Genes])
            return Sort_Into_Permuting_Divergences
        if Error_Type == "2_Intentional_Error": # Caluclate divergences for each Type 2 (false negative) error scenarios.
            ##### Get Inspected Gene Information #####
            # Extract the group 1 and group 2 cardinalities. Group 1 is always the minority group in this set up.
            Minority_Group_Cardinality = Permutables[Feature_Inds]
            Majority_Group_Cardinality = Cell_Cardinality - Minority_Group_Cardinality
            # Identify in which scenarios the inspected gene is the Partitioning Gene and which it is the Permutable Gene
            # Whenever the inspected gene minority state cardinality is smaller than any other gene's, it is the RG
            Partitioning_Inds = np.where(Permutables >= Minority_Group_Cardinality)[0]
            # Whenever the inspected gene minority state cardinality is larger than any other gene's, it is the QG
            Permuting_Inds = np.where(Permutables < Minority_Group_Cardinality)[0]
            ##### Inspected Gene is QG Caclulations #####
            # Extract the group 1 and group 2 cardinalities. Group 1 is always the minority group in this set up.
            Minority_Group_Cardinality = Permutables[Permuting_Inds]
            Majority_Group_Cardinality = Cell_Cardinality - Permutables[Permuting_Inds]
            Permutable = Permutables[Feature_Inds]
            # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESQ)
            Max_Entropy_Permutation = (Minority_Group_Cardinality * Permutable)/(Minority_Group_Cardinality + Majority_Group_Cardinality)
            # The maximum and minimum points of the ESQ are identified from the boundaries of the ESQ curve.
            Min_Entropy_ID_1 = np.zeros(Permuting_Inds.shape[0])
            Min_Entropy_ID_2 = np.repeat(Permutable,Permuting_Inds.shape[0])
            Check_Fits_Group_1 = Minority_Group_Cardinality - Min_Entropy_ID_2
            # If the minority group of the QG is larger than the minority group of the RG then the boundary point is the cardinality of the RG minority group.
            Min_Entropy_ID_2[np.where(Check_Fits_Group_1 < 0)[0]] = Minority_Group_Cardinality[np.where(Check_Fits_Group_1 < 0)[0]]
            # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
            Split_Permute_Value = Reference_Gene_Minority_Group_Overlaps[Permuting_Inds]
            # Num_Divergent_Cell
            Num_Divergent_Cell = Minority_Group_Cardinality - Split_Permute_Value
            # Identify Split Direction (whether the observed arrangment is sorting towards the global minimum entropy or not. I.e. is the QG sorting into the
            # minority or majority group of the RG.)
            Sort_Into_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) >= 0)[0]
            Split_Direction = 1
            Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies = Calculate_Fixed_QG_Sort_Values(2,Split_Direction,Permutable,Max_Entropy_Permutation[Sort_Into_Inds],Minority_Group_Cardinality[Sort_Into_Inds],Majority_Group_Cardinality[Sort_Into_Inds],Min_Entropy_ID_1[Sort_Into_Inds],Min_Entropy_ID_2[Sort_Into_Inds],Split_Permute_Value[Sort_Into_Inds])
            ## Calculate Divergence Information
            # In 
            Divergences = Split_Permute_Entropies - Minimum_Entropies
            # Find the average divergence for each cell that is diverging from the optimal sort.
            Cell_Divergences = Divergences / Num_Divergent_Cell[Sort_Into_Inds]
            # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
            Max_Num_Cell_Divergences = Min_Entropy_ID_2[Sort_Into_Inds] - Max_Entropy_Permutation[Sort_Into_Inds]
            Minimum_Background_Noise = (Max_Permuation_Entropies)/Max_Num_Cell_Divergences
            # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
            RG_QG_Divergences = Cell_Divergences - Minimum_Background_Noise
            # Null/Ignore points that aren't usable.
            RG_QG_Divergences[np.isinf(RG_QG_Divergences)] = 0
            RG_QG_Divergences[np.isnan(RG_QG_Divergences)] = 0
            # Featues whose RG_QG_Divergences are less than 0 would add more entropy to the system per data point imputed,
            # hence ignore them.
            Informative_Genes = np.where(RG_QG_Divergences >= 0)[0]
            # Get feature divergences
            Feature_Divergences = np.zeros(Permutables.shape[0])
            if Informative_Genes.shape[0] > 0:
                Feature_Divergence = np.mean(RG_QG_Divergences[Informative_Genes])
            else:
                Feature_Divergence = 0
            return Feature_Divergence
    else:
        # When a feature cannot be used just give all points a value of 0.
        if Error_Type != "2_Intentional_Error":
            return np.zeros(Cell_Cardinality)
        else:
            return 0


def Calculate_Fixed_QG_Sort_Values(Outputs,Split_Direction,Permutable,Max_Entropy_Permutation,Minority_Group_Cardinality,Majority_Group_Cardinality,Min_Entropy_ID_1,Min_Entropy_ID_2,Split_Permute_Value):
    # Calculate critical points on the ES curve
    Max_Permuation_Entropies = Calc_QG_Entropies(Max_Entropy_Permutation,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutable)
    if Split_Direction == -1 and Outputs != 1:
        # The minimum entropy if none of the QG minority states are in the RG minority group.
        Minimum_Entropies = Calc_QG_Entropies(Min_Entropy_ID_1,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutable)
    if Split_Direction == 1 and Outputs != 1:
        # The minimum entropy if the RG minority group has as many of the QG minority state samples in it as possible.
        Minimum_Entropies = Calc_QG_Entropies(Min_Entropy_ID_2,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutable)
    # The entropy of the arrangment observed in the data set.
    Split_Permute_Entropies = Calc_QG_Entropies(Split_Permute_Value,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutable)
    if Outputs == 1:
        return Split_Permute_Entropies, Max_Permuation_Entropies
    if Outputs == 2:
        return Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies
    if Outputs == 3:
        # Calculate ES parabola properties
        Max_Entropy_Differences = Max_Permuation_Entropies - Minimum_Entropies
        Entropy_Losses = Max_Permuation_Entropies - Split_Permute_Entropies
        # Vector of Information Gain values for each QG/RG pair.
        Information_Gains = Entropy_Losses/Max_Entropy_Differences
        # Vector of Split Weights values for each QG/RG pair.
        Split_Weights = (Max_Permuation_Entropies - Minimum_Entropies) / Max_Permuation_Entropies
        return Information_Gains, Split_Weights


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


def Calculate_Fixed_RG_Sort_Values(Outputs,Split_Direction,Max_Entropy_Permutation,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables,Min_Entropy_ID_1,Min_Entropy_ID_2,Split_Permute_Value):
    # Calculate critical points on the ES curve
    Max_Permuation_Entropies = Calc_RG_Entropies(Max_Entropy_Permutation,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables)
    if Split_Direction == -1 and Outputs != 1:
        # The minimum entropy if none of the QG minority states are in the RG minority group.
        Minimum_Entropies = Calc_RG_Entropies(Min_Entropy_ID_1,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables)
    if Split_Direction == 1 and Outputs != 1:
        # The minimum entropy if the RG minority group has as many of the QG minority state samples in it as possible.
        Minimum_Entropies = Calc_RG_Entropies(Min_Entropy_ID_2,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables)
    # The entropy of the arrangment observed in the data set.
    Split_Permute_Entropies = Calc_RG_Entropies(Split_Permute_Value,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables)
    if Outputs == 1:
        return Split_Permute_Entropies, Max_Permuation_Entropies
    if Outputs == 2:
        return Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies
    if Outputs == 3:
        # Calculate ES parabola properties
        Max_Entropy_Differences = Max_Permuation_Entropies - Minimum_Entropies
        Entropy_Losses = Max_Permuation_Entropies - Split_Permute_Entropies
        # Vector of Information Gain values for each RG/QG pair.
        Information_Gains = Entropy_Losses/Max_Entropy_Differences
        # Vector of Split Weights values for each RG/QG pair.
        Split_Weights = (Max_Permuation_Entropies - Minimum_Entropies) / Max_Permuation_Entropies
        return Information_Gains, Split_Weights


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


#### Here we have additional functions to help with analysis of the results ####

## Retreive the Informations Gains Matrix and Split Weights Matrix for a given cycle ##

def Calculate_ES_Sort_Matricies(Binarised_Input_Matrix, Track_Imputation_Steps, Chosen_Cycle = -1, Min_Clust_Size = 5, Use_Cores = -1, Auto_Save = 1, Observe_Directionality = 0):
    # Set number of cores to use
    Cores_Available = multiprocessing.cpu_count()
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Avaiblable: " + str(Cores_Available))
    print("Cores Used: " + str(Use_Cores))
    # Define data dimensions
    #global Cell_Cardinality
    Cell_Cardinality = Binarised_Input_Matrix.shape[0]
    #global Gene_Cardinality
    Gene_Cardinality = Binarised_Input_Matrix.shape[1]
    # Convert Binarised_Input_Matrix into Minority_Group_Matrix
    Permutables, Switch_State_Inidicies = Find_Permutations(Binarised_Input_Matrix,Cell_Cardinality)
    Binarised_Input_Matrix[:,Switch_State_Inidicies] = (Binarised_Input_Matrix[:,Switch_State_Inidicies] * -1) + 1  
    # Set up Minority_Group_Matrix
    global Minority_Group_Matrix
    Minority_Group_Matrix = copy.copy(Binarised_Input_Matrix)
    # Convert suggested imputation points to correct state for chosen cycle.
    if Chosen_Cycle >= 0:
        Suggested_Impute_Inds = Track_Imputation_Steps[Chosen_Cycle][2]
        Minority_Group_Matrix[Suggested_Impute_Inds] = (Minority_Group_Matrix[Suggested_Impute_Inds] - 1) * -1 
        Step_1_Type_1_Error_Inds = Track_Imputation_Steps[Chosen_Cycle+1][0]
        Minority_Group_Matrix[Step_1_Type_1_Error_Inds] = (Minority_Group_Matrix[Step_1_Type_1_Error_Inds] - 1) * -1
    if Chosen_Cycle < 0:
        Suggested_Impute_Inds = Track_Imputation_Steps[Chosen_Cycle-1][2]
        Minority_Group_Matrix[Suggested_Impute_Inds] = (Minority_Group_Matrix[Suggested_Impute_Inds] - 1) * -1 
        Step_1_Type_1_Error_Inds = Track_Imputation_Steps[Chosen_Cycle][0]
        Minority_Group_Matrix[Step_1_Type_1_Error_Inds] = (Minority_Group_Matrix[Step_1_Type_1_Error_Inds] - 1) * -1
    Cycle_Suggested_Imputations = np.where(Binarised_Input_Matrix != Minority_Group_Matrix)
    # Remove genes below Min_Clust_Size
    Keep_Features = np.where(np.sum(Minority_Group_Matrix,axis=0) >= Min_Clust_Size)[0]
    Ignore = np.where(np.sum(Minority_Group_Matrix,axis=0) >= (Minority_Group_Matrix.shape[1]-Min_Clust_Size))[0] 
    Keep_Features = np.delete(Keep_Features,np.where(np.isin(Keep_Features,Ignore))[0])
    if Keep_Features.shape[0] < Minority_Group_Matrix.shape[1]:
        print("Ignoring " + str(Minority_Group_Matrix.shape[1]-Keep_Features.shape[0]) + " features which are below the Min_Clust_Threshold")
        Minority_Group_Matrix = Minority_Group_Matrix[:,Keep_Features]
    #global Cell_Cardinality
    Cell_Cardinality = Minority_Group_Matrix.shape[0]
    #global Gene_Cardinality
    Gene_Cardinality = Minority_Group_Matrix.shape[1]   
    print("Number of cells: " + str(Cell_Cardinality))
    print("Number of genes: " + str(Gene_Cardinality))
    Permutables, Switch_State_Inidicies = Find_Permutations(Minority_Group_Matrix,Cell_Cardinality)
    # Switch Minority/Majority states to 0/1 where necessary. 
    Minority_Group_Matrix[:,Switch_State_Inidicies] = (Minority_Group_Matrix[:,Switch_State_Inidicies] * -1) + 1
    # Calculate minority group overlap matrix
    Reference_Gene_Minority_Group_Overlaps = Parallel_Find_Minority_Group_Overlaps(Use_Cores,Gene_Cardinality)
    Permutables[Permutables < Min_Clust_Size] = np.nan
    print("Performing Sort Calculations")
    with np.errstate(divide='ignore',invalid='ignore'):
        Information_Gains, Split_Weights = Parallel_Calculate_ES_Matricies(Cell_Cardinality,Gene_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores,Observe_Directionality)
    if Observe_Directionality == 0:
        Double_Counts = np.where(np.logical_and(Information_Gains != 0, Information_Gains.T != 0))
        Information_Gains = Information_Gains + Information_Gains.T
        Information_Gains[Double_Counts] = Information_Gains[Double_Counts] / 2
        Split_Weights = Split_Weights + Split_Weights.T
        Split_Weights[Double_Counts] = Split_Weights[Double_Counts] / 2
    if Auto_Save == 1:
        np.save("Information_Gains.npy",Information_Gains)
        np.save("Split_Weights.npy",Split_Weights)
        np.save("Cycle_Suggested_Imputations.npy",Cycle_Suggested_Imputations)
        np.save("ES_Matrices_Features_Used_Inds.npy",Keep_Features)
    return Information_Gains, Split_Weights, Cycle_Suggested_Imputations, Keep_Features


def Parallel_Calculate_ES_Matricies(Cell_Cardinality,Gene_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores,Observe_Directionality):
    Feature_Inds = np.arange(Gene_Cardinality)
    Pass_Info_To_Cores = np.concatenate((Feature_Inds.reshape(1,Feature_Inds.shape[0]),Reference_Gene_Minority_Group_Overlaps))
    Pass_Info_To_Cores = np.transpose(Pass_Info_To_Cores)
    # Parrallel calculate information gains matrix
    pool = multiprocessing.Pool(processes = Use_Cores)
    Results = pool.map(partial(Calculate_ES_Matricies,Cell_Cardinality=Cell_Cardinality,Permutables=Permutables,Observe_Directionality=Observe_Directionality), Pass_Info_To_Cores)
    pool.close()
    pool.join()
    Results = np.asarray(Results)
    # Retreive Information_Gain_Matrix
    # Retreive Information_Gains and put the features back in the original feature ordering.
    Information_Gains = Results[:,0]
    Information_Gains[np.isnan(Information_Gains)] = 0
    Information_Gains[np.isinf(Information_Gains)] = 0
    # Retreive Informative_Genes and put the features back in the original feature ordering.
    Split_Weights = Results[:,1]
    Split_Weights[np.isnan(Split_Weights)] = 0
    Split_Weights[np.isinf(Split_Weights)] = 0
    return Information_Gains, Split_Weights


def Calculate_ES_Matricies(Pass_Info_To_Cores,Cell_Cardinality,Permutables,Observe_Directionality):
    with np.errstate(divide='ignore',invalid='ignore'):
        # Extract which gene calculations are centred around
        Feature_Inds = int(Pass_Info_To_Cores[0])
        # Remove the Query Gene ind from the data vector
        Reference_Gene_Minority_Group_Overlaps = np.delete(Pass_Info_To_Cores,0)
        Results = []
        if np.isnan(Permutables[Feature_Inds]) == 0:
            ##### Fixed RG Caclulations #####
            # Extract the group 1 and group 2 cardinalities. Group 1 is always the minority group in this set up.
            Minority_Group_Cardinality = Permutables[Feature_Inds]
            Majority_Group_Cardinality = Cell_Cardinality - Minority_Group_Cardinality
            #Permutable = Permutables
            # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESQ)
            Max_Entropy_Permutation = (Minority_Group_Cardinality * Permutables)/(Minority_Group_Cardinality + Majority_Group_Cardinality)
            # The maximum and minimum points of the ESQ are identified from the boundaries of the ESQ curve.
            Min_Entropy_ID_1 = np.zeros(Permutables.shape[0])
            Min_Entropy_ID_2 = copy.copy(Permutables)
            Check_Fits_Group_1 = Minority_Group_Cardinality - Min_Entropy_ID_2
            # If the minority group of the QG is larger than the minority group of the RG then the boundary point is the cardinality of the RG minority group.
            Min_Entropy_ID_2[np.where(Check_Fits_Group_1 < 0)[0]] = Minority_Group_Cardinality
            # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
            Split_Permute_Value = Reference_Gene_Minority_Group_Overlaps
            # Identify Split Direction (whether the observed arrangment is sorting towards the global minimum entropy or not. I.e. is the QG sorting into the
            # minority or majority group of the RG.)
            Sort_Out_Of_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) < 0)[0]
            # Assign Split Directions for each QG/RG pair to a vector.
            Split_Directions = np.repeat(1,Permutables.shape[0])
            Split_Directions[Sort_Out_Of_Inds] = -1
            #
            if Observe_Directionality == 0:
                Scenario_Inds = np.where(Minority_Group_Cardinality <= Permutables)[0]
                if Scenario_Inds.shape[0] > 0:
                    Information_Gains_Temp, Split_Weights_Temp = Calculate_All_Fixed_RG_Sort_Values(Split_Directions[Scenario_Inds],Max_Entropy_Permutation[Scenario_Inds],Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables[Scenario_Inds],Min_Entropy_ID_1[Scenario_Inds],Min_Entropy_ID_2[Scenario_Inds],Split_Permute_Value[Scenario_Inds])
                    Information_Gains = np.zeros(Permutables.shape[0])
                    Information_Gains[Scenario_Inds] = Information_Gains_Temp
                    Information_Gains[Scenario_Inds] = Information_Gains[Scenario_Inds] * Split_Directions[Scenario_Inds]
                    Split_Weights = np.zeros(Permutables.shape[0])
                    Split_Weights[Scenario_Inds] = Split_Weights_Temp
                else:
                    Information_Gains = np.zeros(Reference_Gene_Minority_Group_Overlaps.shape[0])
                    Split_Weights = np.zeros(Reference_Gene_Minority_Group_Overlaps.shape[0])
            else:
                Information_Gains, Split_Weights = Calculate_All_Fixed_RG_Sort_Values(Split_Directions,Max_Entropy_Permutation,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables,Min_Entropy_ID_1,Min_Entropy_ID_2,Split_Permute_Value)
                Information_Gains = Information_Gains * Split_Directions
        else:
            Information_Gains = np.zeros(Reference_Gene_Minority_Group_Overlaps.shape[0])
            Split_Weights = np.zeros(Reference_Gene_Minority_Group_Overlaps.shape[0])
        Results.append(Information_Gains)
        Results.append(Split_Weights)
    return Results


def Calculate_All_Fixed_RG_Sort_Values(Split_Directions,Max_Entropy_Permutation,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables,Min_Entropy_ID_1,Min_Entropy_ID_2,Split_Permute_Value):
    # Calculate critical points on the ES curve
    Max_Permuation_Entropies = Calc_RG_Entropies(Max_Entropy_Permutation,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables)
    Sort_Into_Inds = np.where(Split_Directions == 1)[0]
    Sort_Out_Of_Inds = np.where(Split_Directions == -1)[0]
    Minimum_Entropies = np.zeros(Permutables.shape[0])
    if Sort_Into_Inds.shape[0] > 0:
        # The minimum entropy if none of the QG minority states are in the RG minority group.
        Sort_Into_Minimum_Entropies = Calc_RG_Entropies(Min_Entropy_ID_2[Sort_Into_Inds],Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables[Sort_Into_Inds])
        Minimum_Entropies[Sort_Into_Inds] = Sort_Into_Minimum_Entropies
    if Sort_Out_Of_Inds.shape[0] > 0:
        # The minimum entropy if the RG minority group has as many of the QG minority state samples in it as possible.
        Sort_Out_Of_Minimum_Entropies = Calc_RG_Entropies(Min_Entropy_ID_1[Sort_Out_Of_Inds],Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables[Sort_Out_Of_Inds])  
        Minimum_Entropies[Sort_Out_Of_Inds] = Sort_Out_Of_Minimum_Entropies       
    # The entropy of the arrangment observed in the data set.
    Split_Permute_Entropies = Calc_RG_Entropies(Split_Permute_Value,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables)
    # Calculate ES parabola properties
    Max_Entropy_Differences = Max_Permuation_Entropies - Minimum_Entropies
    Entropy_Losses = Max_Permuation_Entropies - Split_Permute_Entropies
    # Vector of Information Gain values for each RG/QG pair.
    Information_Gains = Entropy_Losses/Max_Entropy_Differences
    # Vector of Split Weights values for each RG/QG pair.
    Split_Weights = (Max_Permuation_Entropies - Minimum_Entropies) / Max_Permuation_Entropies
    return Information_Gains, Split_Weights

###

## Optimise the Thresholds by reducing the error of discrete states between the original data set and the converged structure ##

def Parallel_Optimise_Discretisation_Thresholds(Original_Data,Binarised_Input_Matrix,Cycle_Suggested_Imputations,Use_Cores=-1,Auto_Save=1):
     # Set number of cores to use
    Cores_Available = multiprocessing.cpu_count()
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Avaiblable: " + str(Cores_Available))
    print("Cores Used: " + str(Use_Cores))
    # Convert suggested imputation points to unknown state.
    Binarised_Input_Matrix[Cycle_Suggested_Imputations] = np.nan
    Paired = [[]] * Original_Data.shape[1]
    for i in np.arange(Original_Data.shape[1]):
        Paired[i] = np.stack((Original_Data[:,i],Binarised_Input_Matrix[:,i]))       
    pool = multiprocessing.Pool(processes = Use_Cores)
    Result = pool.map(Optimise_Discretisation_Thresholds, Paired)
    pool.close()
    pool.join()
    Result = np.asarray(Result,dtype=object)
    Optimised_Imputations = np.stack(Result[:,0],axis=1)
    Optimised_Imputations = np.where(Optimised_Imputations == 1)
    Thresholds = Result[:,1].astype("f")
    Starting_False_Negatives = Thresholds = Result[:,2]
    Starting_False_Positives = Thresholds = Result[:,3]
    Optimal_False_Negatives = Thresholds = Result[:,4]
    Optimal_False_Positives = Thresholds = Result[:,5]
    Track_Errors = Result[:,6]
    if Auto_Save == 1:
        np.save("Optimised_Imputations.npy",Optimised_Imputations)
        np.save("Thresholds.npy",Thresholds)
        np.save("Starting_False_Negatives.npy",Starting_False_Negatives)
        np.save("Starting_False_Positives.npy",Starting_False_Positives)
        np.save("Optimal_False_Negatives.npy",Optimal_False_Negatives)
        np.save("Optimal_False_Positives.npy",Optimal_False_Positives)
        np.save("Track_Errors.npy",Track_Errors)
    return Optimised_Imputations, Thresholds, Starting_False_Negatives, Starting_False_Positives, Optimal_False_Negatives, Optimal_False_Positives, Track_Errors


def Optimise_Discretisation_Thresholds(Paired):
    Original_Gene = Paired[0,:]
    Imputed_Gene = Paired[1,:]
    Imputed_Cells = np.where(np.isnan(Imputed_Gene))[0]
    Results = [[]] * 7
    if Imputed_Cells.shape[0] > 0:
        False_Negatives = Imputed_Cells[np.where(Original_Gene[Imputed_Cells] == 0)[0]]
        False_Positives = Imputed_Cells[np.where(Original_Gene[Imputed_Cells] != 0)[0]]
        # Save starting number of False Negatives
        Results[2] = False_Negatives.shape[0]
        # Save starting number of False Positives
        Results[3] = False_Positives.shape[0]
        Target_Expression_States = copy.copy(Original_Gene)
        Target_Expression_States[Target_Expression_States > 0] = 1
        # 0 = Inactive, 1 = Active. Unlike in ES, the Active/Inactive definition matter.
        Target_Expression_States[Imputed_Cells] = (Target_Expression_States[Imputed_Cells] * -1) + 1
        Unique_Exspression = np.unique(Original_Gene)
        Min_Error = np.inf
        Track_Errors = np.zeros(Unique_Exspression.shape[0])
        for Thresh in np.arange(Unique_Exspression.shape[0]):
            Threshold = Unique_Exspression[Thresh]
            Threshold_Expression_States = copy.copy(Original_Gene)
            Threshold_Expression_States[Threshold_Expression_States < Threshold] = 0
            Threshold_Expression_States[Threshold_Expression_States != 0] = 1
            False_Negatives = Imputed_Cells[np.where(Threshold_Expression_States[Imputed_Cells] == 0)[0]]
            False_Positives = Imputed_Cells[np.where(Threshold_Expression_States[Imputed_Cells] == 1)[0]]
            Differences = np.absolute(Target_Expression_States-Threshold_Expression_States)
            Error = np.sum(Differences)
            Error_Inds = np.where(Differences != 0)[0]
            if Error < Min_Error:
                Min_Error = Error
                Min_Thresh = Thresh
                Impute_Cells = Imputed_Cells[np.where(np.isin(Imputed_Cells,Error_Inds) == 1)[0]]
                Impute_Vector = np.zeros(Original_Gene.shape[0])
                Impute_Vector[Impute_Cells] = 1
                # Save minimum error imputation vector
                Results[0] = Impute_Vector
                # Save Thresholds for minimum error
                Results[1] = Unique_Exspression[Thresh]
                # Save optimal number of False Negatives
                Results[4] = False_Negatives.shape[0]
                # Save optimal number of False Positives
                Results[5] = False_Positives.shape[0]
            Track_Errors[Thresh] =  Error
        # Save Tracked Errors
        Results[6] = Track_Errors
        #plt.figure()
        #plt.plot(Unique_Exspression,Errors[0,:])
        #plt.vlines(Min_Error_Ind,min(Errors[0,:]),max(Errors[0,:]),color="r")        
        #plt.figure()
        #plt.scatter(np.arange(Original_Gene.shape[0]),Original_Gene)
        #plt.scatter(np.where(np.isnan(Imputed_Data[:,Ind]))[0],Original_Gene[np.where(np.isnan(Imputed_Data[:,Ind]))[0]])
        #plt.scatter(Results[1],Original_Data[Results[1],Ind])
    else:
        Results[0] = np.repeat(-1,Original_Gene.shape[0])
        Results[1] = -1
        Results[2] = -1
        Results[3] = -1
        Results[4] = -1
        Results[5] = -1
        Results[6] = -1
    return Results


###

def Estimate_Feature_Importance(Intended_Divergence, Binarised_Input_Matrix, Cycle_Suggested_Imputations, Num_Cycles = 5, Min_Clust_Size = 5, Use_Cores = -1, Auto_Save = 1):
    # Set number of cores to use
    Cores_Available = multiprocessing.cpu_count()
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Avaiblable: " + str(Cores_Available))
    print("Cores Used: " + str(Use_Cores))
    # Define data dimensions
    #global Cell_Cardinality
    Cell_Cardinality = Binarised_Input_Matrix.shape[0]
    #global Gene_Cardinality
    Gene_Cardinality = Binarised_Input_Matrix.shape[1]
    # Convert Binarised_Input_Matrix into Minority_Group_Matrix
    Permutables, Switch_State_Inidicies = Find_Permutations(Binarised_Input_Matrix,Cell_Cardinality)
    Binarised_Input_Matrix[:,Switch_State_Inidicies] = (Binarised_Input_Matrix[:,Switch_State_Inidicies] * -1) + 1  
    # Set up Minority_Group_Matrix
    global Minority_Group_Matrix
    print("Number of cells: " + str(Cell_Cardinality))
    print("Number of genes: " + str(Gene_Cardinality))
    Feature_Divergences = []
    for Cycle in np.arange(Num_Cycles):
        print("Cycle Number: " + str(Cycle+1) + " of " + str(Num_Cycles))
        Minority_Group_Matrix = copy.copy(Binarised_Input_Matrix)
        Minority_Group_Matrix[Cycle_Suggested_Imputations] = (Minority_Group_Matrix[Cycle_Suggested_Imputations] * -1) + 1
        # Remove genes below Min_Clust_Size
        Keep_Features = np.where(np.sum(Minority_Group_Matrix,axis=0) >= Min_Clust_Size)[0]
        Ignore = np.where(np.sum(Minority_Group_Matrix,axis=0) >= (Minority_Group_Matrix.shape[1]-Min_Clust_Size))[0] 
        Keep_Features = np.delete(Keep_Features,np.where(np.isin(Keep_Features,Ignore))[0])
        if Keep_Features.shape[0] < Minority_Group_Matrix.shape[1]:
            print("Ignoring " + str(Minority_Group_Matrix.shape[1]-Keep_Features.shape[0]) + " features which are below the Min_Clust_Threshold")
            Minority_Group_Matrix = Minority_Group_Matrix[:,Keep_Features]
        #global Cell_Cardinality
        Cell_Cardinality = Minority_Group_Matrix.shape[0]
        #global Gene_Cardinality
        Gene_Cardinality = Minority_Group_Matrix.shape[1]   
        print("Number of cells: " + str(Cell_Cardinality))
        print("Number of genes: " + str(Gene_Cardinality))
        Permutables, Switch_State_Inidicies = Find_Permutations(Minority_Group_Matrix,Cell_Cardinality)
        # Switch Minority/Majority states to 0/1 where necessary. 
        Minority_Group_Matrix[:,Switch_State_Inidicies] = (Minority_Group_Matrix[:,Switch_State_Inidicies] * -1) + 1
        # Intentionally Add Error
        for i in np.arange(Minority_Group_Matrix.shape[1]):
            Feature_Minority_Inds = np.where(Minority_Group_Matrix[:,i] == 1)[0]
            Null = np.random.choice(Feature_Minority_Inds.shape[0], int(Feature_Minority_Inds.shape[0]*Intended_Divergence),replace=False)
            Minority_Group_Matrix[Feature_Minority_Inds[Null],i] = 0
        Permutables, Switch_State_Inidicies = Find_Permutations(Minority_Group_Matrix,Cell_Cardinality)
        # Switch Minority/Majority states to 0/1 where necessary. 
        Minority_Group_Matrix[:,Switch_State_Inidicies] = (Minority_Group_Matrix[:,Switch_State_Inidicies] * -1) + 1
        # Calculate minority group overlap matrix
        Reference_Gene_Minority_Group_Overlaps = Parallel_Find_Minority_Group_Overlaps(Use_Cores,Gene_Cardinality)
        Permutables[Permutables < Min_Clust_Size] = np.nan
        with np.errstate(divide='ignore',invalid='ignore'):
            Feature_Divergence = Parallel_Calculate_Cell_Divergences("2_Intentional_Error",Cell_Cardinality,Gene_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores)
        Feature_Divergences.append(Feature_Divergence)
    Feature_Divergences = np.asarray(Feature_Divergences)
    if Auto_Save == 1:
        np.save("Feature_Divergences.npy",Feature_Divergences)
        np.save("Feature_Divergences_Used_Inds.npy",Keep_Features)
    return Feature_Divergences, Keep_Features

## Get the Information Gain and Split Weights for each gene in the dataset, for a subset of cells in the data.
# Main use is to find which gene exists in a specific region of an embedding.

def Gene_Overlays_With_Cell_Subset(Cell_IDs, Binarised_Input_Matrix, Min_Clust_Size = 10):
    Cell_Inds = np.where(np.isin(Binarised_Input_Matrix.index,Cell_IDs))[0]
    Cell_Vector = np.zeros(Binarised_Input_Matrix.shape[0]).astype("i")
    Cell_Vector[Cell_Inds] = 1
    Binarised_Input_Matrix["Cell_Subset"] = Cell_Vector
    Cell_Cardinality = Binarised_Input_Matrix.shape[0]
    global Minority_Group_Matrix
    Minority_Group_Matrix = np.asarray(Binarised_Input_Matrix)
    # Create Minority_Group_Matrix objects, Permutables and Switch_State_Inidicies objects.
    Permutables, Switch_State_Inidicies = Find_Permutations(Minority_Group_Matrix,Cell_Cardinality)
    Permutables[Permutables < Min_Clust_Size] = np.nan
    # Switch Minority/Majority states to 0/1 where necessary.
    Minority_Group_Matrix[:,Switch_State_Inidicies] = (Minority_Group_Matrix[:,Switch_State_Inidicies] * -1) + 1  
    # Find minority state overlaps with cell subset
    Overlaps = Find_Minority_Group_Overlaps(-1)
    Pass_Info_To_Cores = np.append(-1,Overlaps)
    Results = Calculate_ES_Matricies(Pass_Info_To_Cores,Cell_Cardinality,Permutables,Observe_Directionality=1)
    Information_Gains = Results[0]
    Information_Gains[np.isnan(Information_Gains)] = 0
    Information_Gains[np.isinf(Information_Gains)] = 0
    # Retreive Informative_Genes and put the features back in the original feature ordering.
    Split_Weights = Results[1]
    Split_Weights[np.isnan(Split_Weights)] = 0
    Split_Weights[np.isinf(Split_Weights)] = 0
    return Information_Gains, Split_Weights


