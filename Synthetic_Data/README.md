# Synthetic Data
Here we present the reader with the initial synthetic data sets that were used in our publication, and all the code required to generate the results presented in our work.

### Code
Within the Code folder there are 5 workflows:
1. Run_FFAVES_ESFW_Synthetic_Data.py takes the synthetic datasets in the Objects_From_Paper folder and applies FFAVES and ESFW to re-create the results in the paper. You will need to run these scripts before you can plot some of the figures generated in the next workflow (2.).
2. Generate_Imputed_Expression_Matricies.r uses the outputs from FFAVES/ESFW, and a selection of previously published imputation software, to impute the noisy drop out synthetic data so that their performances may be compared.
3. Generate_Gene_Rankings.r uses the outputs from FFAVES/ESFW, and a selection of previously published feature selection software, to rank gene importance in the drop out synthetic data so that their performances may be compared.
4. Synthetic_Data_Plotting.py plots the above results.
5. Generate_Simple_Synthetic_Data.py allows the user to create a new simple synthetic dataset with known ground truth in the same manner that the provided synthetic dataset was generated. 
