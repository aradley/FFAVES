# The following code will generate synthetic data similar to the data used the Entropy Sorting paper.
# If you accidently save over the data in the Objects_For_Paper folder, you may need to re-download it.
# This is obviously a relitively contrived and rigid way of creating the synthetic data. The main purpose
# of this script is to show clearly how the synthetic data used in the paper was created, and how it was
# new synthetic data can be quickly created with different false negative drop out probabilities.

### Import packages ###
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
###

# Num_Branches = the number of branches/cell types
# Cell_Branch_Sizes = the number of samples/branches in each cell type

def Create_Synthetic_Data(Num_Branches = 5, Cell_Branch_Sizes = 200, Dropout_Probabiliy = 0.3, Num_Noise_Features = 500, Save_Data = 0):
    Total_Cells = Num_Branches * Cell_Branch_Sizes
    Branch_Gene_Cardinalities = np.flip(np.arange(20,Cell_Branch_Sizes+10,10))
    Num_Genes_Per_Branch = Branch_Gene_Cardinalities.shape[0]
    Num_Genes = Num_Branches * Num_Genes_Per_Branch
    Synthetic_Data = []
    # Create initial cell type chunks and gradients.
    Start_Cell = 0
    for i in np.arange(Num_Branches):
        for j in np.arange(Num_Genes_Per_Branch):
            Synthetic_Gene = np.zeros(Num_Branches*Cell_Branch_Sizes)
            for k in np.arange(np.random.randint(5,15)):
                Shift_Start_Cell = Start_Cell #+ np.random.randint(-15,15)
                if Shift_Start_Cell < 0:
                    Shift_Start_Cell = 0
                Synthetic_Gene[Shift_Start_Cell:(Shift_Start_Cell+Branch_Gene_Cardinalities[j])] = 1
                Synthetic_Data.append(Synthetic_Gene)
        Start_Cell = Start_Cell + Cell_Branch_Sizes
    #
    Synthetic_Data = np.stack(Synthetic_Data,axis=0).T
    # Overlap with two cell types
    Overlap_Gene = np.zeros(Synthetic_Data.shape[0])
    Overlap_Gene[0:((Cell_Branch_Sizes*2))] = 1
    for i in np.arange(50):
        Synthetic_Data = np.column_stack((Overlap_Gene,Synthetic_Data))
    #
    # Track the ground truth expression for plotting in the future.
    Structured_Expression = np.where(Synthetic_Data==1)
    # Randomly upregulated in two cell types
    for i in np.arange(50):
        Random_Int = np.random.randint(2,10)
        Random_Gene_1 = np.random.choice([0, 1], size=(Cell_Branch_Sizes*2), p=[(10-Random_Int)/10, (Random_Int)/10])
        Random_Int = int(Random_Int/2)
        Random_Gene_2 = np.random.choice([0, 1], size=(Cell_Branch_Sizes*(Num_Branches-2)), p=[(10-Random_Int)/10, (Random_Int)/10])
        Random_Gene = np.append(Random_Gene_2,Random_Gene_1)
        Synthetic_Data = np.column_stack((Synthetic_Data,Random_Gene))
    # Random Noisy Genes
    for i in np.arange(Num_Noise_Features):
        Random_Int = 20
        while Random_Int >= 10 or Random_Int <= 0:
            Random_Int = int(np.random.normal(3,3))
        Random_Gene = np.random.choice([0, 1], size=(Synthetic_Data.shape[0]), p=[(10-Random_Int)/10, (Random_Int)/10])
        Synthetic_Data = np.column_stack((Synthetic_Data,Random_Gene))
    # Create multiple sequenced cells in single samples.
    Num_Multiples = int(Synthetic_Data.shape[0]*0.2) # 20% doublet rate
    Cell_Combos = [[]] * Num_Multiples
    for i in np.arange(Num_Multiples):
        Multiple = 0
        while Multiple < 2:
            Multiple = int(np.random.poisson(1, 1))
        Cells = np.zeros(Multiple).astype("i")
        Multiple_Cell = np.zeros(Synthetic_Data.shape[1])
        for j in np.arange(Multiple):
            Cells[j] = np.random.randint(0,(Synthetic_Data.shape[0]-1))
            Multiple_Cell = Multiple_Cell + Synthetic_Data[Cells[j],:]
        Cell_Combos[i] = Cells
        Synthetic_Data = np.concatenate((Synthetic_Data,Multiple_Cell.reshape(1,Multiple_Cell.shape[0])))
    #
    Discrete_Complete_Synthetic_Data = copy.copy(Synthetic_Data)
    #### Add FNs ###
    # Two pool batch effects where odd or evening numbered samples/cells have a higher faction of dropouts than
    # the corresponding odd or even samples. The drop out bias is determined by sampling from a normal distribution
    # with mean 1 and standard deviation 0.4, and using the distance from the mean to determine whether the
    # bias will be in the odd or even samples, and the magnitude of the bias.
    Batch_1 = np.arange(0,Total_Cells,2)
    Batch_2 = np.arange(1,Total_Cells,2)
    for i in np.arange(Synthetic_Data.shape[1]):
        Gene = Synthetic_Data[:,i]
        Batch_Bias = 5
        while Batch_Bias >= 2 or Batch_Bias <= 0:
            Batch_Bias = np.random.normal(1,0.4)
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
    #
    # Track the introduced FNs for plotting in the future.
    FNs = np.where(np.logical_and(np.asarray(Synthetic_Data)==0,np.asarray(Discrete_Complete_Synthetic_Data)==1))
    #### Add FPs ###
    Temp_Discrete_Complete_Synthetic_Data = copy.copy(Discrete_Complete_Synthetic_Data)
    # Leaky Expression. This represents false positives in the data where expression appears outside of the cell types where it is expected
    # according to the ground truth. Leaky expression is proportional to the minority state cardinality of the gene. Having determined how
    # many points of leaky FP expression will be introduced for a particular gene/feature, it is randomly added to samples for that gene/feature.
    for i in np.arange(Synthetic_Data.shape[1]):
        Gene_Cardinality = np.sum(Synthetic_Data[:,i])
        Leaky_Probablility = (Gene_Cardinality/Synthetic_Data.shape[0]) * (np.random.randint(2,10)/100)
        Leaky_Expression = np.random.choice([0, 1], size=(Synthetic_Data.shape[0]), p=[(1-Leaky_Probablility), Leaky_Probablility])
        Temp_Discrete_Complete_Synthetic_Data[:,i] = Temp_Discrete_Complete_Synthetic_Data[:,i] - (Leaky_Expression)
    #
    FPs = np.where(Temp_Discrete_Complete_Synthetic_Data == -1)
    Synthetic_Data[FPs] = Synthetic_Data[FPs] + 1
    Discrete_Complete_Synthetic_Data[FPs] = Discrete_Complete_Synthetic_Data[FPs] + 1
    #Discrete_Complete_Synthetic_Data[Discrete_Complete_Synthetic_Data==3]
    # Make overlapping genes multimodal
    Discrete_Complete_Synthetic_Data[np.ix_(np.arange(Cell_Branch_Sizes,Cell_Branch_Sizes*2),np.arange(50))] = Discrete_Complete_Synthetic_Data[np.ix_(np.arange(Cell_Branch_Sizes,Cell_Branch_Sizes*2),np.arange(50))] * 2
    # Introduce Continuous expression scale
    Unique_Multiexpression = np.unique(Discrete_Complete_Synthetic_Data)
    for i in np.arange(1,Unique_Multiexpression.shape[0]):
        Active_Points = np.where(Discrete_Complete_Synthetic_Data == i)
        Continuous_Data = np.zeros(Active_Points[0].shape[0])
        for j in np.arange(i):
            Continuous_Data = Continuous_Data + np.random.normal(5,1,Continuous_Data.shape[0])
        #
        Discrete_Complete_Synthetic_Data[Active_Points] = Continuous_Data
    #
    Synthetic_Data[np.where(Synthetic_Data!=0)] = Discrete_Complete_Synthetic_Data[np.where(Synthetic_Data!=0)]
    # Consolodate data
    Cell_Combos = np.asarray(Cell_Combos,dtype=object)
    Complete_Synthetic_Data = pd.DataFrame(Discrete_Complete_Synthetic_Data)
    Drop_Out_Synthetic_Data = pd.DataFrame(Synthetic_Data)
    ### Save data and FPs/FNs
    if Save_Data == 1:
        np.save("Cell_Combos.npy",Cell_Combos)
        Complete_Synthetic_Data.to_csv("Complete_Synthetic_Data.csv")
        Drop_Out_Synthetic_Data.to_csv("Drop_Out_Synthetic_Data.csv")
        np.save("Structured_Expression.npy",Structured_Expression)
        np.save("FNs.npy", FNs)
        np.save("FPs.npy", FPs)
    return Complete_Synthetic_Data, Drop_Out_Synthetic_Data, Structured_Expression, FNs, FPs


# Having loaded the code, you can create your own comparable data as was used in the Entropy Sorting paper.
Complete_Synthetic_Data, Drop_Out_Synthetic_Data, Structured_Expression, FNs, FPs = Create_Synthetic_Data(Num_Branches = 5, Cell_Branch_Sizes = 200, Dropout_Probabiliy = 0.3, Num_Noise_Features = 500, Save_Data = 0)

# And you can take a look at it

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Ground Truth Synthetic Data",fontsize=16)
plt.imshow(np.log10(Complete_Synthetic_Data+1),cmap="magma")
plt.xlabel("Features/Genes", fontsize=12)
plt.ylabel("Samples/Cells", fontsize=12)
plt.colorbar(shrink=0.75)
plt.subplot(1,2,2)
plt.title("Noisy/Dropout Synthetic Data",fontsize=16)
plt.imshow(np.log10(Drop_Out_Synthetic_Data+1),cmap="magma")
plt.xlabel("Features/Genes", fontsize=12)
plt.ylabel("Samples/Cells", fontsize=12)
plt.colorbar(shrink=0.75)

plt.show()

