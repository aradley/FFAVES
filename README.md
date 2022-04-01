
# FFAVES and ESFW
This repository contains the code for our software, Functional Feature Amplification Via Entropy Sorting (FFAVES) and Entropy Sorting Feature Weighting (ESFW). A detailed description of these software can be found in our paper, Functional feature selection reveals the inner cell mass in human pre-implantation embryo single cell RNA sequencing data.

### Installation
1. Retreive the ripository with: `git clone https://github.com/aradley/FFAVES.git`
2. Navigate to the directory where the clone was downloaded to, for example: `cd FFAVES/`
3. Run the following on the command line: `python setup.py install`

Dependencies for FFAVES and ESFW are outlined in the requirements.txt file. Dependencies will be automatically installed by `python setup.py install`.

### Data and example workflows
The Synthetic_Data folder in this repository provides the synthetic data described in our paper, and all the code needed to re-create the results. The process of re-creating the synthetic data results should be relitively quick.

The human pre-implantation embryo data used in our publication is available for download at https://data.mendeley.com/datasets/689pm8s7jc/draft?a=cc12423c-c19c-49b9-8cd9-d883064c048f. Likewise, the code required for re-creating the results can be found in the linked directory. Because re-creating the results for the human pre-implantation embryo data would require significant computational resources, the linked repository also contains a minimal set of FFAVES and ESFW outputs that are required for re-creating the plots presented in the paper.

### Usage
The main input for FFAVES and ESFW is a discretised state matrix (rows are samples and columns are features) where the samples of a feature can be represented as existing in one of two states by 0's or 1's. In our paper we discretise scRNA-seq data such that genes can be considered as active or inactive (it does not matter whether 1 or 0 indicates active or inactive, but we chose 0 to indicate inactive). Discretisation in this manner is often appropriate. Other examples include chromatin accessibility/inaccessibility or genome sequence methylation/non-methylation. We remind potential users that discretisation of your data need not be perfect. As long as discretisation is carried out in a reasonable and rational manner, FFAVES has built in methodology to attempt to correct sub-optimal discretisation.

### Authors note on code optimisation
FFAVES and ESFW have been parallelised which helps them be tractable on large datasets. However, this implementation of Entroy Sorting into FFAVES and ESFW is by no means the most efficient or compute optimised. Some estimates suggest that FFAVES and ESFW could be made 500-1000 times faster if implemented in a compute optimised language or through compute optimised packages such a numba. If anyone in the community would like to try and increase the speed of FFAVES/ESFW it would be most welcome. If you would like to undetake this together such that our work be associated with each other, please get in contact. If you decide to go it alone, please cite our paper.
