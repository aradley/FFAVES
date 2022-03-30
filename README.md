
# FFAVES and ESFW
This repository contains the code for our software, Functional Feature Amplification Via Entropy Sorting (FFAVES) and Entropy Sorting Feature Weighting (ESFW). A detailed description of these software can be found in our paper, Functional feature selection reveals the inner cell mass in human pre-implantation embryo single cell RNA sequencing data.

### Installation
1. Retreive the ripository with: `git clone https://github.com/aradley/FFAVES.git`
2. Navigate to the directory where the clone was downloaded to, for example: `cd FFAVES/`
3. Run the following on the command line: `python setup.py install`

Dependencies for FFAVES and ESFW are outlined in the requirements.txt file. Dependencies will be automatically installed by `python setup.py install`.

### Usage
The Synthetic_Data folder in this repository provides the synthetic data described in our paper, and all the code needed to re-create the results. The process of re-creating the synthetic data results should be relitively quick.

The human pre-implantation embryo data used in our publication is available for download at https://data.mendeley.com/datasets/3zcwg7dmh5/draft?a=bb564341-6c15-4d8a-8c69-5dbc5e8c64df. Likewise, the code required for re-creating the results can be found in the linked directory. Because re-creating the results for the human pre-implantation embryo data would require significant computational resources, the linked repository also contains a minimal set of FFAVES and ESFW outputs that are required for re-creating the plots presented in the paper.
