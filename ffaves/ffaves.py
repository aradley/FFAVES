### Dependencies ###

import numpy as np
from functools import partial 
import multiprocessing
import copy
from scipy.stats import halfnorm, zscore

### Dependencies ###

### Here we have the FFAVES wrapper function that executes all the steps of FFAVES. ###

def FFAVES(Binarised_Input_Matrix, Min_Clust_Size = 5, Divergences_Significance_Cut_Off = 0.99, Use_Cores= -1, Num_Cycles = 1, Tolerance = 0.05, Auto_Save = 1):
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
    Track_Percentage_Imputation = np.zeros((2,Num_Cycles+1))
    Track_Imputation_Steps = np.empty(((Num_Cycles + 1),),dtype="object")
    Track_Imputation_Steps[...]=[([[]]*4) for _ in range((Num_Cycles + 1))]  
    # Cell Uncertainties
    Track_Cell_Uncertainties = np.zeros((Num_Cycles,Cell_Cardinality))
    # Set up tolerance tracker
    Covergence_Counter = 0
    while Imputation_Cycle <= Num_Cycles and Covergence_Counter < 3:
        if Imputation_Cycle > 1:
            print("Percentage of original data suggested as Type 1 Error: " + str(np.round((Track_Imputation_Steps[Imputation_Cycle-1][0][0].shape[0]/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100,2)) + "%")
            print("Percentage of original data suggested as Type 2 Error: " + str(np.round((Track_Imputation_Steps[Imputation_Cycle-1][3][0].shape[0]/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100,2)) + "%")   
        # Initiate Cycle_Imputation_Steps
        Cycle_Imputation_Steps = [[]] * 4
        # Initiate Cycle_Cell_Uncertainties
        Cycle_Cell_Uncertainties = np.zeros(Cell_Cardinality)
        print("Cycle Number " + str(Imputation_Cycle))         
        Minority_Group_Matrix = copy.copy(Binarised_Input_Matrix)
        # Convert suggested imputation points from previous cycle to correct state.
        All_Type_2_Error_Impute_Inds = Track_Imputation_Steps[Imputation_Cycle-1][3]
        Minority_Group_Matrix[All_Type_2_Error_Impute_Inds] = (Minority_Group_Matrix[All_Type_2_Error_Impute_Inds] - 1) * -1 
        ### Step 1 of FFAVES is to identify and temporarily remove spurious Minority Group expression states
        with np.errstate(divide='ignore',invalid='ignore'):
            Step_1_Type_1_Error_Inds, Switch_State_Inidicies_1, Cell_Uncertainties = FFAVES_Step_1(Min_Clust_Size,Divergences_Significance_Cut_Off,Use_Cores,Cell_Cardinality,Gene_Cardinality)  
        # Track Cell_Uncertainties
        Cycle_Cell_Uncertainties = Cycle_Cell_Uncertainties + Cell_Uncertainties
        Minority_Group_Matrix[Step_1_Type_1_Error_Inds] = (Minority_Group_Matrix[Step_1_Type_1_Error_Inds] - 1) * -1
        ### Step 2 of FFAVES is to identify which majority states points are spurious
        with np.errstate(divide='ignore',invalid='ignore'):
            Type_2_Error_Inds, Switch_State_Inidicies_2, Cell_Uncertainties = FFAVES_Step_2(Min_Clust_Size,Divergences_Significance_Cut_Off,Use_Cores,Cell_Cardinality,Gene_Cardinality)       
        # Track Cell_Uncertainties
        Cycle_Cell_Uncertainties = Cycle_Cell_Uncertainties + Cell_Uncertainties
        Minority_Group_Matrix[Type_2_Error_Inds] = (Minority_Group_Matrix[Type_2_Error_Inds] - 1) * -1
        # If Majority State imputation has led to a state inversion, we declare this set of imputations as leading to
        # ambuguity of feature Minority State locality in the data. As such, we ignore these imputations
        # under the assumption that if the Minority State of a feature is structured in the original dataset,
        # it's locality in the data would not completely switch. This is a consequence of the fact that the accruacy
        # of Entropy Sorting struggles when the cardinality of the minority state of a feature approaches half the cardinality
        # of the number of samples.
        State_Inversions = np.where(np.sum(Minority_Group_Matrix,axis=0) > (Cell_Cardinality/float(2)))[0]
        if State_Inversions.shape[0] > 0:
            Remove_Imputations = np.where(np.isin(Type_2_Error_Inds[1],State_Inversions))[0]
            Remove_Imputations = (Type_2_Error_Inds[0][Remove_Imputations],Type_2_Error_Inds[1][Remove_Imputations])
            Minority_Group_Matrix[Remove_Imputations] = (Minority_Group_Matrix[Remove_Imputations] - 1) * -1
            Keep_Imputations = np.where(np.isin(Type_2_Error_Inds[1],State_Inversions) == 0)[0]
            Type_2_Error_Inds = (Type_2_Error_Inds[0][Keep_Imputations],Type_2_Error_Inds[1][Keep_Imputations])
        ### Step 3 of FFAVES is to identify and remove spurious suggested imputations
        with np.errstate(divide='ignore',invalid='ignore'):
            Step_3_Type_1_Error_Inds, Switch_State_Inidicies_1 = FFAVES_Step_3(Min_Clust_Size,Divergences_Significance_Cut_Off,Use_Cores,Cell_Cardinality,Gene_Cardinality)  
        ### Null suggested imputations that significantly disagree with aspects of the existing structure
        Flat_Step_1_Inds = np.ravel_multi_index(Step_1_Type_1_Error_Inds, Binarised_Input_Matrix.shape)
        Flat_Step_2_Inds = np.ravel_multi_index(Type_2_Error_Inds, Binarised_Input_Matrix.shape)
        Flat_Step_3_Inds = np.ravel_multi_index(Step_3_Type_1_Error_Inds, Binarised_Input_Matrix.shape)
        if Imputation_Cycle > 1:
            All_Type_2_Error_Impute_Inds = np.unique(np.append(np.ravel_multi_index(All_Type_2_Error_Impute_Inds, Binarised_Input_Matrix.shape), Flat_Step_2_Inds))
        else:
            All_Type_2_Error_Impute_Inds = Flat_Step_2_Inds
        # Ignore double imputations (i.e removed in step 1 and are then re-imputed in step 2)
        Ignore_Imputations = np.where(np.isin(All_Type_2_Error_Impute_Inds,Flat_Step_1_Inds))[0]
        All_Type_2_Error_Impute_Inds = np.delete(All_Type_2_Error_Impute_Inds,Ignore_Imputations)
        # Remove spurious imputations identified in step 3
        Ignore_Imputations = np.where(np.isin(All_Type_2_Error_Impute_Inds,Flat_Step_3_Inds))[0]
        All_Type_2_Error_Impute_Inds = np.delete(All_Type_2_Error_Impute_Inds,Ignore_Imputations)
        All_Type_2_Error_Impute_Inds = np.unravel_index(All_Type_2_Error_Impute_Inds,Binarised_Input_Matrix.shape)
        ### Track Erroneous points
        Cycle_Imputation_Steps[0] = Step_1_Type_1_Error_Inds
        Cycle_Imputation_Steps[1] = Type_2_Error_Inds
        Cycle_Imputation_Steps[2] = Step_3_Type_1_Error_Inds
        Cycle_Imputation_Steps[3] = All_Type_2_Error_Impute_Inds
        Track_Imputation_Steps[Imputation_Cycle] = Cycle_Imputation_Steps
        Track_Cell_Uncertainties[(Imputation_Cycle-1),:] = Cycle_Cell_Uncertainties 
        print("Finished")
        Track_Percentage_Imputation[0,Imputation_Cycle] = (Track_Imputation_Steps[Imputation_Cycle][0][0].shape[0]/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100
        Track_Percentage_Imputation[1,Imputation_Cycle] = (Track_Imputation_Steps[Imputation_Cycle][3][0].shape[0]/(Binarised_Input_Matrix.shape[0]*Binarised_Input_Matrix.shape[1]))*100
        if Imputation_Cycle < Num_Cycles:
            Imputed_Difference = np.absolute(Track_Percentage_Imputation[1,Imputation_Cycle] - Track_Percentage_Imputation[1,Imputation_Cycle-1])
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
        Track_Imputation_Steps[i][3] = (Track_Imputation_Steps[i][3][0],Keep_Features[Track_Imputation_Steps[i][3][1]])
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
    Type_1_Error_Divergences = Parallel_Calculate_Cell_Divergences(1,Cell_Cardinality,Gene_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores)
    Cell_Uncertainties = np.sum(Type_1_Error_Divergences,axis=1)
    print("Identifying unreliable data points via half normal distribution.")
    # Use half normal distribution of normalised divergent points to suggest which points should be re-evaluated
    Use_Inds = np.where(Minority_Group_Matrix != 0)
    Divergences = Type_1_Error_Divergences[Use_Inds]    
    # Get zscores for observed divergences    
    zscores = zscore(Divergences)
    zscores = zscores + np.absolute(np.min(zscores))
    # Identify points that diverge in a statistically significant way
    Pass_Threshold = np.where(halfnorm.cdf(zscores) >= Divergences_Significance_Cut_Off)[0]
    Type_1_Error_Inds = (Use_Inds[0][Pass_Threshold],Use_Inds[1][Pass_Threshold])
    return Type_1_Error_Inds, Switch_State_Inidicies, Cell_Uncertainties


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
    Type_2_Error_Divergences = Parallel_Calculate_Cell_Divergences(2,Cell_Cardinality,Gene_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores)     
    Cell_Uncertainties = np.sum(Type_2_Error_Divergences,axis=1)
    print("Identifying data points for imputation via half normal distribution")
    # Use half normal distribution of normalised divergent points to suggest which points should be re-evaluated
    Use_Inds = np.where(Minority_Group_Matrix == 0)
    Divergences = Type_2_Error_Divergences[Use_Inds]
    zscores = zscore(Divergences)
    zscores = zscores + np.absolute(np.min(zscores))              
    # Identify points that diverge in a statistically significant way
    Pass_Threshold = np.where(halfnorm.cdf(zscores) >= Divergences_Significance_Cut_Off)[0]
    Type_2_Error_Inds = (Use_Inds[0][Pass_Threshold],Use_Inds[1][Pass_Threshold])
    return Type_2_Error_Inds, Switch_State_Inidicies, Cell_Uncertainties


def FFAVES_Step_3(Min_Clust_Size,Divergences_Significance_Cut_Off,Use_Cores,Cell_Cardinality,Gene_Cardinality):
    print("Step 3: Cleaning up untrustworthy imputed values.")
    print("Identifying Sort Info for calculations.")
    # Create Minority_Group_Matrix objects, Permutables and Switch_State_Inidicies objects.
    Permutables, Switch_State_Inidicies = Find_Permutations(Minority_Group_Matrix,Cell_Cardinality)
    # Switch Minority/Majority states to 0/1 where necessary.
    Minority_Group_Matrix[:,Switch_State_Inidicies] = (Minority_Group_Matrix[:,Switch_State_Inidicies] * -1) + 1
    # Calculate minority group overlap matrix
    Reference_Gene_Minority_Group_Overlaps = Parallel_Find_Minority_Group_Overlaps(Use_Cores,Gene_Cardinality)
    Permutables[Permutables < Min_Clust_Size] = np.nan   
    print("Calculating Divergence Matrix")
    Type_1_Error_Divergences = Parallel_Calculate_Cell_Divergences(1,Cell_Cardinality,Gene_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores)            
    #Fixed_RG_Neg_SD_Divergences, Information_Gains_Matrix, Weights_Matrix = Parallel_Fixed_RG_Neg_SD(Cell_Cardinality,Permutables,Reference_Gene_Minority_Group_Overlaps,Use_Cores)        
    print("Identifying unreliable imputed data points via half normal distribution.")
    Use_Inds = np.where(Minority_Group_Matrix != 0)
    Divergences = Type_1_Error_Divergences[Use_Inds]
    zscores = zscore(Divergences)
    zscores = zscores + np.absolute(np.min(zscores))    
    # Identify points that diverge in a statistically significant way
    Pass_Threshold = np.where(halfnorm.cdf(zscores) >= Divergences_Significance_Cut_Off)[0]
    Type_1_Error_Inds = (Use_Inds[0][Pass_Threshold],Use_Inds[1][Pass_Threshold])
    return Type_1_Error_Inds, Switch_State_Inidicies


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
    # Retreive Information_Gain_Matrix
    # Retreive Fixed_QG_Neg_SD_Divergences and put the features back in the original feature ordering.
    Divergence_Matrix = np.stack(Result,axis=1)
    return Divergence_Matrix

### Calculate Divergence Matrix
def Calculate_Cell_Divergences(Pass_Info_To_Cores,Error_Type,Cell_Cardinality,Permutables):   
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
        if Error_Type == 1: # Caluclate divergences for each Type 1 (false positive) error scenarios.
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
            # Num_Divergent_Cell
            Num_Divergent_Cell = Minority_Group_Cardinality - Split_Permute_Value
            # Identify Split Direction (whether the observed arrangment is sorting towards the global minimum entropy or not. I.e. is the QG sorting into the
            # minority or majority group of the RG.)
            Sort_Out_Of_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) < 0)[0]
            # Assign Split Directions for each QG/RG pair to a vector.
            Split_Directions = np.repeat(1,Permutables.shape[0])
            Split_Directions[Sort_Out_Of_Inds] = -1     
            #### For each scenario, G1 and G2 refer to the cardinality of the minority group of Gene 1 and Gene 2 ####
            ### Type 1 Error Scenario 1) Fixed RG, SD = +1 and G1 <= G2 ###
            Scenario_1_Inds = np.where(np.logical_and(Split_Directions == 1, Minority_Group_Cardinality <= Permutables))[0]
            Split_Direction = 1
            Split_Permute_Entropies, Max_Permuation_Entropies = Calculate_Fixed_RG_Sort_Values(1,Split_Direction,Max_Entropy_Permutation[Scenario_1_Inds],Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables[Scenario_1_Inds],Min_Entropy_ID_1[Scenario_1_Inds],Min_Entropy_ID_2[Scenario_1_Inds],Split_Permute_Value[Scenario_1_Inds])
            ## Calculate Divergence Information
            # In scenario 1 we include the gap
            Divergences = Split_Permute_Entropies
            # Find the average divergence for each cell that is diverging from the optimal sort.
            Cell_Divergences = Divergences / Num_Divergent_Cell[Scenario_1_Inds]
            # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
            Max_Num_Cell_Divergences = Min_Entropy_ID_2[Scenario_1_Inds] - Max_Entropy_Permutation[Scenario_1_Inds]
            Minimum_Background_Noise = Max_Permuation_Entropies/Max_Num_Cell_Divergences
            # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
            RG_QG_Divergences = Cell_Divergences - Minimum_Background_Noise
            # Null/Ignore points that aren't usable.
            RG_QG_Divergences[np.isinf(RG_QG_Divergences)] = 0
            RG_QG_Divergences[np.isnan(RG_QG_Divergences)] = 0
            # Featues whose RG_QG_Divergences are less than 0 would add more entropy to the system per data point imputed,
            # hence ignore them.
            Informative_Genes = np.where(RG_QG_Divergences >= 0)[0]
            # Get Overlap Matrix
            Sort_Genes = Minority_Group_Matrix[np.ix_(Fixed_Gene_Minority_States,Scenario_1_Inds[Informative_Genes])]
            Scenario_1_Divergences = np.zeros(Cell_Cardinality)
            Scenario_1_Divergences[Fixed_Gene_Minority_States] = np.dot(Sort_Genes==0,RG_QG_Divergences[Informative_Genes])
            ### Type 1 Error Scenario 2) Fixed RG, SD = -1 and G1 > G2 ###
            Scenario_2_Inds = np.where(np.logical_and(Split_Directions == -1, Minority_Group_Cardinality > Permutables))[0]
            Split_Direction = -1
            Split_Permute_Entropies, Max_Permuation_Entropies = Calculate_Fixed_RG_Sort_Values(1,Split_Direction,Max_Entropy_Permutation[Scenario_2_Inds],Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables[Scenario_2_Inds],Min_Entropy_ID_1[Scenario_2_Inds],Min_Entropy_ID_2[Scenario_2_Inds],Split_Permute_Value[Scenario_2_Inds])
            # ## Calculate Divergence Information
            # # In scenario 2 we include the gap
            # Divergences = Split_Permute_Entropies
            # # Find the average divergence for each cell that is diverging from the optimal sort.
            # Cell_Divergences = Divergences / Num_Divergent_Cell[Scenario_2_Inds]
            # # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
            # Max_Num_Cell_Divergences = Max_Entropy_Permutation[Scenario_2_Inds]
            # Minimum_Background_Noise = Max_Permuation_Entropies/Max_Num_Cell_Divergences
            # # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
            # RG_QG_Divergences = Cell_Divergences - Minimum_Background_Noise
            # # Null/Ignore points that aren't usable.
            # RG_QG_Divergences[np.isinf(RG_QG_Divergences)] = 0
            # RG_QG_Divergences[np.isnan(RG_QG_Divergences)] = 0
            # # Featues whose RG_QG_Divergences are less than 0 would add more entropy to the system per data point imputed,
            # # hence ignore them.
            # Informative_Genes = np.where(RG_QG_Divergences >= 0)[0]
            # # Get Overlap Matrix
            # Sort_Genes = Minority_Group_Matrix[np.ix_(Fixed_Gene_Minority_States,Scenario_2_Inds[Informative_Genes])]
            # Scenario_2_Divergences = np.zeros(Cell_Cardinality)
            # Scenario_2_Divergences[Fixed_Gene_Minority_States] = np.dot(Sort_Genes,RG_QG_Divergences[Informative_Genes])
            ### Type 1 Error Scenario 3) Fixed RG, SD = -1 and G1 <= G2 ###
            Scenario_3_Inds = np.where(np.logical_and(Split_Directions == -1, Minority_Group_Cardinality <= Permutables))[0]
            Split_Direction = -1
            Split_Permute_Entropies, Max_Permuation_Entropies = Calculate_Fixed_RG_Sort_Values(1,Split_Direction,Max_Entropy_Permutation[Scenario_3_Inds],Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables[Scenario_3_Inds],Min_Entropy_ID_1[Scenario_3_Inds],Min_Entropy_ID_2[Scenario_3_Inds],Split_Permute_Value[Scenario_3_Inds])
            ## Calculate Divergence Information
            # In scenario 3 we include the gap
            Divergences = Split_Permute_Entropies
            # Find the average divergence for each cell that is diverging from the optimal sort.
            Cell_Divergences = Divergences / Split_Permute_Value[Scenario_3_Inds]
            # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
            Max_Num_Cell_Divergences = Max_Entropy_Permutation[Scenario_3_Inds]
            Minimum_Background_Noise = Max_Permuation_Entropies/Max_Num_Cell_Divergences
            # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
            RG_QG_Divergences = Cell_Divergences - Minimum_Background_Noise
            # Null/Ignore points that aren't usable.
            RG_QG_Divergences[np.isinf(RG_QG_Divergences)] = 0
            RG_QG_Divergences[np.isnan(RG_QG_Divergences)] = 0
            # Featues whose RG_QG_Divergences are less than 0 would add more entropy to the system per data point imputed,
            # hence ignore them.
            Informative_Genes = np.where(RG_QG_Divergences >= 0)[0]
            # Get Overlap Matrix
            Sort_Genes = Minority_Group_Matrix[np.ix_(Fixed_Gene_Minority_States,Scenario_3_Inds[Informative_Genes])]
            Scenario_3_Divergences = np.zeros(Cell_Cardinality)
            Scenario_3_Divergences[Fixed_Gene_Minority_States] = np.dot(Sort_Genes,RG_QG_Divergences[Informative_Genes])
            # ##### Fixed QG Caclulations #####
            # Extract the group 1 and group 2 cardinalities. Group 1 is always the minority group in this set up.
            Minority_Group_Cardinality = Permutables
            Majority_Group_Cardinality = Cell_Cardinality - Permutables
            Permutable = Permutables[Feature_Inds]
            # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESQ)
            Max_Entropy_Permutation = (Minority_Group_Cardinality * Permutable)/(Minority_Group_Cardinality + Majority_Group_Cardinality)
            # The maximum and minimum points of the ESQ are identified from the boundaries of the ESQ curve.
            Min_Entropy_ID_1 = np.zeros(Permutables.shape[0])
            Min_Entropy_ID_2 = np.repeat(Permutable,Permutables.shape[0])
            Check_Fits_Group_1 = Minority_Group_Cardinality - Min_Entropy_ID_2
            # If the minority group of the QG is larger than the minority group of the RG then the boundary point is the cardinality of the RG minority group.
            Min_Entropy_ID_2[np.where(Check_Fits_Group_1 < 0)[0]] = Minority_Group_Cardinality[np.where(Check_Fits_Group_1 < 0)[0]]
            # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
            Split_Permute_Value = Reference_Gene_Minority_Group_Overlaps
            # Num_Divergent_Cell
            Num_Divergent_Cell = Minority_Group_Cardinality - Split_Permute_Value            
            # ## Type 1 Error Scenario 4) Fixed QG, SD = 1 and G1 <= G2 ###
            # Scenario_4_Inds = np.where(np.logical_and(Split_Directions == 1, Minority_Group_Cardinality > Permutable))[0]
            # Split_Direction = 1
            # Split_Permute_Entropies, Max_Permuation_Entropies = Calculate_Fixed_QG_Sort_Values(1,Split_Direction,Permutable,Max_Entropy_Permutation[Scenario_4_Inds],Minority_Group_Cardinality[Scenario_4_Inds],Majority_Group_Cardinality[Scenario_4_Inds],Min_Entropy_ID_1[Scenario_4_Inds],Min_Entropy_ID_2[Scenario_4_Inds],Split_Permute_Value[Scenario_4_Inds])
            # # In scenario 4 we include the gap
            # Divergences = Split_Permute_Entropies
            # # Find the average divergence for each cell that is diverging from the optimal sort.
            # Cell_Divergences = Divergences / Split_Permute_Value[Scenario_4_Inds]
            # # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
            # Max_Num_Cell_Divergences = Min_Entropy_ID_2[Scenario_4_Inds] - Max_Entropy_Permutation[Scenario_4_Inds]
            # Minimum_Background_Noise = Max_Permuation_Entropies/Max_Num_Cell_Divergences
            # # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
            # RG_QG_Divergences = Cell_Divergences - Minimum_Background_Noise
            # # Null/Ignore points that aren't usable.
            # RG_QG_Divergences[np.isinf(RG_QG_Divergences)] = 0
            # RG_QG_Divergences[np.isnan(RG_QG_Divergences)] = 0
            # # Featues whose RG_QG_Divergences are less than 0 would add more entropy to the system per data point imputed.
            # # Null these data points by setting all overlaps to 0.
            # Informative_Genes = np.where(RG_QG_Divergences >= 0)[0]
            # # Get Overlap Matrix
            # Sort_Genes = Minority_Group_Matrix[np.ix_(Fixed_Gene_Minority_States,Scenario_4_Inds[Informative_Genes])]
            # Scenario_4_Divergences = np.zeros(Cell_Cardinality)
            # Scenario_4_Divergences[Fixed_Gene_Minority_States] = np.dot(Sort_Genes==0,RG_QG_Divergences[Informative_Genes])
            # ### Type 1 Error Scenario 5) Fixed QG, SD = -1 and G1 <= G2 ###
            # Scenario_5_Inds = np.where(np.logical_and(Split_Directions == -1, Minority_Group_Cardinality > Permutable))[0]
            # Split_Direction = -1
            # Split_Permute_Entropies, Max_Permuation_Entropies = Calculate_Fixed_QG_Sort_Values(1,Split_Direction,Permutable,Max_Entropy_Permutation[Scenario_5_Inds],Minority_Group_Cardinality[Scenario_5_Inds],Majority_Group_Cardinality[Scenario_5_Inds],Min_Entropy_ID_1[Scenario_5_Inds],Min_Entropy_ID_2[Scenario_5_Inds],Split_Permute_Value[Scenario_5_Inds])
            # ## Calculate Divergence Information
            # # In scenario 5 we include the gap
            # Divergences = Split_Permute_Entropies
            # # Find the average divergence for each cell that is diverging from the optimal sort.
            # Cell_Divergences = Divergences / Split_Permute_Value[Scenario_5_Inds]
            # # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
            # Max_Num_Cell_Divergences = Max_Entropy_Permutation[Scenario_5_Inds]
            # Minimum_Background_Noise = Max_Permuation_Entropies/Max_Num_Cell_Divergences
            # # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
            # RG_QG_Divergences = Cell_Divergences - Minimum_Background_Noise
            # # Null/Ignore points that aren't usable.
            # RG_QG_Divergences[np.isinf(RG_QG_Divergences)] = 0
            # RG_QG_Divergences[np.isnan(RG_QG_Divergences)] = 0
            # # Featues whose RG_QG_Divergences are less than 0 would add more entropy to the system per data point imputed.
            # # Null these data points by setting all overlaps to 0.
            # Informative_Genes = np.where(RG_QG_Divergences >= 0)[0]
            # # Get Overlap Matrix
            # Sort_Genes = Minority_Group_Matrix[np.ix_(Fixed_Gene_Minority_States,Scenario_5_Inds[Informative_Genes])]
            # Scenario_5_Divergences = np.zeros(Cell_Cardinality)
            # Scenario_5_Divergences[Fixed_Gene_Minority_States] = np.dot(Sort_Genes,RG_QG_Divergences[Informative_Genes])
            ## Type 1 Error Scenario 6) Fixed QG, SD = -1 and G1 <= G2 ###
            Scenario_6_Inds = np.where(np.logical_and(Split_Directions == -1, Minority_Group_Cardinality <= Permutable))[0]
            Split_Direction = -1
            Split_Permute_Entropies, Max_Permuation_Entropies = Calculate_Fixed_QG_Sort_Values(1,Split_Direction,Permutable,Max_Entropy_Permutation[Scenario_6_Inds],Minority_Group_Cardinality[Scenario_6_Inds],Majority_Group_Cardinality[Scenario_6_Inds],Min_Entropy_ID_1[Scenario_6_Inds],Min_Entropy_ID_2[Scenario_6_Inds],Split_Permute_Value[Scenario_6_Inds])
            ## Calculate Divergence Information
            # In scenario 6 we include the gap
            Divergences = Split_Permute_Entropies
            # Find the average divergence for each cell that is diverging from the optimal sort.
            Cell_Divergences = Divergences / Split_Permute_Value[Scenario_6_Inds]
            # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
            Max_Num_Cell_Divergences = Max_Entropy_Permutation[Scenario_6_Inds]
            Minimum_Background_Noise = Max_Permuation_Entropies/Max_Num_Cell_Divergences
            # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
            RG_QG_Divergences = Cell_Divergences - Minimum_Background_Noise
            # Null/Ignore points that aren't usable.
            RG_QG_Divergences[np.isinf(RG_QG_Divergences)] = 0
            RG_QG_Divergences[np.isnan(RG_QG_Divergences)] = 0
            # Featues whose RG_QG_Divergences are less than 0 would add more entropy to the system per data point imputed,
            # hence ignore them.
            Informative_Genes = np.where(RG_QG_Divergences >= 0)[0]
            # Get Overlap Matrix
            Sort_Genes = Minority_Group_Matrix[np.ix_(Fixed_Gene_Minority_States,Scenario_6_Inds[Informative_Genes])]
            Scenario_6_Divergences = np.zeros(Cell_Cardinality)
            Scenario_6_Divergences[Fixed_Gene_Minority_States] = np.dot(Sort_Genes,RG_QG_Divergences[Informative_Genes])
             ##
            Type_1_Error_Divergences = (Scenario_1_Divergences+Scenario_3_Divergences+Scenario_6_Divergences)
            #Type_1_Error_Divergences = (Scenario_1_Divergences+Scenario_2_Divergences+Scenario_3_Divergences)
            #Type_1_Error_Divergences = (Scenario_4_Divergences+Scenario_5_Divergences+Scenario_6_Divergences)
            #Type_1_Error_Divergences = (Scenario_1_Divergences+Scenario_2_Divergences+Scenario_3_Divergences+Scenario_4_Divergences+Scenario_5_Divergences+Scenario_6_Divergences)
            return Type_1_Error_Divergences
        if Error_Type == 2: # Caluclate divergences for each Type 2 (false negative) error scenarios.
            # ##### Fixed RG Caclulations #####
            # # Extract the group 1 and group 2 cardinalities. Group 1 is always the minority group in this set up.
            # Minority_Group_Cardinality = Permutables[Feature_Inds]
            # Majority_Group_Cardinality = Cell_Cardinality - Minority_Group_Cardinality
            # #Permutable = Permutables
            # # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESQ)
            # Max_Entropy_Permutation = (Minority_Group_Cardinality * Permutables)/(Minority_Group_Cardinality + Majority_Group_Cardinality)
            # # The maximum and minimum points of the ESQ are identified from the boundaries of the ESQ curve.
            # Min_Entropy_ID_1 = np.zeros(Permutables.shape[0])
            # Min_Entropy_ID_2 = copy.copy(Permutables)
            # Check_Fits_Group_1 = Minority_Group_Cardinality - Min_Entropy_ID_2
            # # If the minority group of the QG is larger than the minority group of the RG then the boundary point is the cardinality of the RG minority group.
            # Min_Entropy_ID_2[np.where(Check_Fits_Group_1 < 0)[0]] = Minority_Group_Cardinality
            # # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
            # Split_Permute_Value = Reference_Gene_Minority_Group_Overlaps
            # # Num_Divergent_Cell
            # Num_Divergent_Cell = Minority_Group_Cardinality - Split_Permute_Value
            # # Identify Split Direction (whether the observed arrangment is sorting towards the global minimum entropy or not. I.e. is the QG sorting into the
            # # minority or majority group of the RG.)
            # Sort_Out_Of_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) < 0)[0]
            # # Assign Split Directions for each QG/RG pair to a vector.
            # Split_Directions = np.repeat(1,Permutables.shape[0])
            # Split_Directions[Sort_Out_Of_Inds] = -1     
            # #### For each scenario, G1 and G2 refer to the cardinality of the minority group of Gene 1 and Gene 2 ####
            # ### Type 2 Error Scenario 1) Fixed RG, SD = +1 and G1 > G2 ###
            # Scenario_1_Inds = np.where(np.logical_and(Split_Directions == 1, Minority_Group_Cardinality > Permutables))[0]
            # Split_Direction = 1
            # Split_Permute_Entropies, Max_Permuation_Entropies, Minimum_Entropies = Calculate_Fixed_RG_Sort_Values(2,Split_Direction,Max_Entropy_Permutation[Scenario_1_Inds],Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables[Scenario_1_Inds],Min_Entropy_ID_1[Scenario_1_Inds],Min_Entropy_ID_2[Scenario_1_Inds],Split_Permute_Value[Scenario_1_Inds])
            # # In scenario 1 we don't include the gap
            # Divergences = Split_Permute_Entropies - Minimum_Entropies
            # # Find the average divergence for each cell that is diverging from the optimal sort.
            # Cell_Divergences = Divergences / Num_Divergent_Cell[Scenario_1_Inds]
            # # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
            # Max_Num_Cell_Divergences = Min_Entropy_ID_2[Scenario_1_Inds] - Max_Entropy_Permutation[Scenario_1_Inds]
            # Minimum_Background_Noise = Max_Permuation_Entropies/Max_Num_Cell_Divergences
            # # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
            # RG_QG_Divergences = Cell_Divergences - Minimum_Background_Noise
            # # Featues whose RG_QG_Divergences are less than 0 would add more entropy to the system per data point imputed.
            # # Null these data points by setting all overlaps to 0.
            # Informative_Genes = np.where(RG_QG_Divergences >= 0)[0]
            # # Get Overlap Matrix
            # Sort_Genes = Minority_Group_Matrix[np.ix_(Fixed_Gene_Majority_States,Scenario_1_Inds[Informative_Genes])]
            # Scenario_1_Divergences = np.zeros(Cell_Cardinality)
            # Scenario_1_Divergences[Fixed_Gene_Majority_States] = np.dot(Sort_Genes,RG_QG_Divergences[Informative_Genes])
            #### Fixed QG Caclulations #####
            # Extract the group 1 and group 2 cardinalities. Group 1 is always the minority group in this set up.
            Minority_Group_Cardinality = Permutables
            Majority_Group_Cardinality = Cell_Cardinality - Permutables
            Permutable = Permutables[Feature_Inds]
            # Maximum entropy of the system is identified from the derivative of the Entropy Sorting Equation (ESQ)
            Max_Entropy_Permutation = (Minority_Group_Cardinality * Permutable)/(Minority_Group_Cardinality + Majority_Group_Cardinality)
            # The maximum and minimum points of the ESQ are identified from the boundaries of the ESQ curve.
            Min_Entropy_ID_1 = np.zeros(Permutables.shape[0])
            Min_Entropy_ID_2 = np.repeat(Permutable,Permutables.shape[0])
            Check_Fits_Group_1 = Minority_Group_Cardinality - Min_Entropy_ID_2
            # If the minority group of the QG is larger than the minority group of the RG then the boundary point is the cardinality of the RG minority group.
            Min_Entropy_ID_2[np.where(Check_Fits_Group_1 < 0)[0]] = Minority_Group_Cardinality[np.where(Check_Fits_Group_1 < 0)[0]]
            # Split_Permute_Value is the overlap of minority states that we actually observe in the data.
            Split_Permute_Value = Reference_Gene_Minority_Group_Overlaps
            # Num_Divergent_Cell
            Num_Divergent_Cell = Minority_Group_Cardinality - Split_Permute_Value
            # Identify Split Direction (whether the observed arrangment is sorting towards the global minimum entropy or not. I.e. is the QG sorting into the
            # minority or majority group of the RG.)
            Sort_Out_Of_Inds = np.where((Split_Permute_Value - Max_Entropy_Permutation) < 0)[0]
            # Assign Split Directions for each QG/RG pair to a vector.
            Split_Directions = np.repeat(1,Permutables.shape[0])
            Split_Directions[Sort_Out_Of_Inds] = -1
            ### Type 2 Error Scenario 2) Fixed RG, SD = +1 and G1 <= G2 ###
            Scenario_2_Inds = np.where(np.logical_and(Split_Directions == 1, Minority_Group_Cardinality < Permutable))[0]
            Split_Direction = 1
            Split_Permute_Entropies, Max_Permuation_Entropies,Minimum_Entropies = Calculate_Fixed_QG_Sort_Values(2,Split_Direction,Permutable,Max_Entropy_Permutation[Scenario_2_Inds],Minority_Group_Cardinality[Scenario_2_Inds],Majority_Group_Cardinality[Scenario_2_Inds],Min_Entropy_ID_1[Scenario_2_Inds],Min_Entropy_ID_2[Scenario_2_Inds],Split_Permute_Value[Scenario_2_Inds])
            ## Calculate Divergence Information
            # In scenario 2 we exclude the gap
            Divergences = Split_Permute_Entropies - Minimum_Entropies
            # Find the average divergence for each cell that is diverging from the optimal sort.
            Cell_Divergences = Divergences / Num_Divergent_Cell[Scenario_2_Inds]
            # Calculate how much divergence each cell would have if the RG/QG system was at the maximum entropy arrangment.
            Max_Num_Cell_Divergences = Min_Entropy_ID_2[Scenario_2_Inds] - Max_Entropy_Permutation[Scenario_2_Inds]
            Minimum_Background_Noise = Max_Permuation_Entropies/Max_Num_Cell_Divergences
            # Deduct the observed average divergence per cell from average divergence per cell in the maximum entorpy arrangment.
            RG_QG_Divergences = Cell_Divergences - Minimum_Background_Noise
            # Null/Ignore points that aren't usable.
            RG_QG_Divergences[np.isinf(RG_QG_Divergences)] = 0
            RG_QG_Divergences[np.isnan(RG_QG_Divergences)] = 0
            # Featues whose RG_QG_Divergences are less than 0 would add more entropy to the system per data point imputed,
            # hence ignore them.
            Informative_Genes = np.where(RG_QG_Divergences >= 0)[0]
            # Get Overlap Matrix
            Sort_Genes = Minority_Group_Matrix[np.ix_(Fixed_Gene_Majority_States,Scenario_2_Inds[Informative_Genes])]
            Scenario_2_Divergences = np.zeros(Cell_Cardinality)
            Scenario_2_Divergences[Fixed_Gene_Majority_States] = np.dot(Sort_Genes,RG_QG_Divergences[Informative_Genes])
            ##
            #Type_2_Error_Divergences = (Scenario_1_Divergences)
            Type_2_Error_Divergences = (Scenario_2_Divergences)
            return Type_2_Error_Divergences
    else:
        # When a feature cannot be used just give all points a value of 0.
        return np.zeros(Cell_Cardinality)


def Calculate_Fixed_QG_Sort_Values(Outputs,Split_Direction,Permutable,Max_Entropy_Permutation,Minority_Group_Cardinality,Majority_Group_Cardinality,Min_Entropy_ID_1,Min_Entropy_ID_2,Split_Permute_Value):
    # Calculate critical points on the ES curve
    Max_Permuation_Entropies = Calc_QG_Entropies(Max_Entropy_Permutation,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutable)
    if Split_Direction == -1:
        # The minimum entropy if none of the QG minority states are in the RG minority group.
        Minimum_Entropies = Calc_QG_Entropies(Min_Entropy_ID_1,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutable)
    if Split_Direction == 1:
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
    if Split_Direction == -1:
        # The minimum entropy if none of the QG minority states are in the RG minority group.
        Minimum_Entropies = Calc_RG_Entropies(Min_Entropy_ID_1,Minority_Group_Cardinality,Majority_Group_Cardinality,Permutables)
    if Split_Direction == 1:
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
    # Track what cycle FFAVES is on.
    print("Number of cells: " + str(Cell_Cardinality))
    print("Number of genes: " + str(Gene_Cardinality))
    Minority_Group_Matrix = copy.copy(Binarised_Input_Matrix)
    # Convert suggested imputation points to correct state for chosen cycle.
    if Chosen_Cycle >= 0:
        Suggested_Impute_Inds = Track_Imputation_Steps[Chosen_Cycle][3]
        Minority_Group_Matrix[Suggested_Impute_Inds] = (Minority_Group_Matrix[Suggested_Impute_Inds] - 1) * -1 
        Step_1_Type_1_Error_Inds = Track_Imputation_Steps[Chosen_Cycle+1][0]
        Minority_Group_Matrix[Step_1_Type_1_Error_Inds] = (Minority_Group_Matrix[Step_1_Type_1_Error_Inds] - 1) * -1
    if Chosen_Cycle < 0:
        Suggested_Impute_Inds = Track_Imputation_Steps[Chosen_Cycle-1][3]
        Minority_Group_Matrix[Suggested_Impute_Inds] = (Minority_Group_Matrix[Suggested_Impute_Inds] - 1) * -1 
        Step_1_Type_1_Error_Inds = Track_Imputation_Steps[Chosen_Cycle][0]
        Minority_Group_Matrix[Step_1_Type_1_Error_Inds] = (Minority_Group_Matrix[Step_1_Type_1_Error_Inds] - 1) * -1
    Cycle_Suggested_Imputations = np.where(Binarised_Input_Matrix != Minority_Group_Matrix)
    # Type_2_Error_Inds = Track_Imputation_Steps[Chosen_Cycle][1]
    # Flat_Step_2_Inds = np.ravel_multi_index(Type_2_Error_Inds, Binarised_Input_Matrix.shape)
    # Step_3_Type_1_Error_Inds = Track_Imputation_Steps[Chosen_Cycle][2]
    # Flat_Step_3_Inds = np.ravel_multi_index(Step_3_Type_1_Error_Inds, Binarised_Input_Matrix.shape)
    # Flat_Start_Of_Cycle_Imputations = np.ravel_multi_index(Suggested_Impute_Inds, Binarised_Input_Matrix.shape)
    # All_Type_2_Error_Impute_Inds = np.unique(np.append(Flat_Start_Of_Cycle_Imputations, Flat_Step_2_Inds))
    # Ignore_Imputations = np.where(np.isin(All_Type_2_Error_Impute_Inds,Flat_Step_3_Inds))[0]
    # All_Type_2_Error_Impute_Inds = np.delete(All_Type_2_Error_Impute_Inds,Ignore_Imputations)
    # Ignore_Imputations = np.where(np.isin(All_Type_2_Error_Impute_Inds,Flat_Start_Of_Cycle_Imputations))[0]
    # All_Type_2_Error_Impute_Inds = np.delete(All_Type_2_Error_Impute_Inds,Ignore_Imputations)
    # All_Type_2_Error_Impute_Inds = np.unravel_index(All_Type_2_Error_Impute_Inds,Binarised_Input_Matrix.shape)    
    # Minority_Group_Matrix[All_Type_2_Error_Impute_Inds] = (Minority_Group_Matrix[All_Type_2_Error_Impute_Inds] - 1) * -1
    # Create Minority_Group_Matrix objects, Permutables and Switch_State_Inidicies objects.
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
    return Information_Gains, Split_Weights, Cycle_Suggested_Imputations


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

def Parallel_Optimise_Discretisation_Thresholds(Original_Data,Binarised_Input_Matrix,Track_Imputation_Steps,Use_Cores=-1,Auto_Save=1):
     # Set number of cores to use
    Cores_Available = multiprocessing.cpu_count()
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Avaiblable: " + str(Cores_Available))
    print("Cores Used: " + str(Use_Cores))
    # Convert suggested imputation points to unknown state.
    Binarised_Input_Matrix[Track_Imputation_Steps[0]] = (Binarised_Input_Matrix[Track_Imputation_Steps[0]]*-1) + 1
    Binarised_Input_Matrix[Track_Imputation_Steps[1]] = (Binarised_Input_Matrix[Track_Imputation_Steps[1]]*-1) + 1
    Binarised_Input_Matrix[Track_Imputation_Steps[2]] = (Binarised_Input_Matrix[Track_Imputation_Steps[2]]*-1) + 1
    Paired = [[]] * Original_Data.shape[1]
    for i in np.arange(Original_Data.shape[1]):
        Paired[i] = np.stack((Original_Data[:,i],Binarised_Input_Matrix[:,i]))       
    pool = multiprocessing.Pool(processes = Use_Cores)
    Result = pool.map(Optimise_Discretisation_Thresholds, Paired)
    pool.close()
    pool.join()
    Result = np.asarray(Result,dtype=object)
    Thresholds = Result[:,0].astype("f")
    Optimised_Imputations = np.stack(Result[:,1],axis=1)
    Optimised_Imputations = np.where(Optimised_Imputations == 1)
    if Auto_Save == 1:
        np.save("Thresholds.npy",Thresholds)
        np.save("Optimised_Imputations.npy",Optimised_Imputations)
    return Thresholds, Optimised_Imputations


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


###

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
    Results = Calculate_ES_Matricies(Pass_Info_To_Cores,Cell_Cardinality,Permutables)
    Information_Gains = Results[0]
    Information_Gains[np.isnan(Information_Gains)] = 0
    Information_Gains[np.isinf(Information_Gains)] = 0
    # Retreive Informative_Genes and put the features back in the original feature ordering.
    Split_Weights = Results[1]
    Split_Weights[np.isnan(Split_Weights)] = 0
    Split_Weights[np.isinf(Split_Weights)] = 0
    return Information_Gains, Split_Weights


