# The following code takes the synthetic data from the paper and uses ESFW, FFAVES + ESFW, Scran,
# Seurat and scry to rank all of the features (genes) as to whether they convery correlations 
# between samples (cells) in the data. These are all R software.

### Set folder path as required ###
path = "/mnt/c/Users/arthu/OneDrive\ -\ University\ of\ Cambridge/Entropy_Sorting_Paper_2022/"
path_2 = "Synthetic_Data/Objects_For_Paper/"
###

### R Code for SCRAN Highly Variable Genes ####
library(SingleCellExperiment)
library(scran)

counts <- read.csv("Drop_Out_Synthetic_Data.csv")
counts <- counts[,-1]
# Columns are cells, rows are genes
counts <- t(counts)
colnames(counts) <- as.character(1:1200)
sce <- SingleCellExperiment(list(counts=counts),
    metadata=list(study="Synthetic Data"))

logcounts(sce) <- log(counts+1)
rownames(sce) <- as.character(1:1519)
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
#SCRAN_HVG_Order <- order(-dec$bio)
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
Seurat_HVGs <- gsub("-", "_", Seurat_HVGs)
Seurat_HVG_Order <- match(Seurat_HVGs,Orig_Order)
Seurat_HVG_Order = append(Seurat_HVG_Order,which((Orig_Order %in% Seurat_HVGs)==0))

## Un-comment to save results ##
#write.csv(Seurat_HVG_Order, "Seurat_HVG_Order.csv")

#### R code for scry Highly Variable Genes ####
library(scry)

m <- data.frame((matrix(0, ncol = 2, nrow = 1519)))
rowData(sce) <- m
deviances <- rowData(devianceFeatureSelection(sce))$binomial_deviance
scry_HVG_Order <- as.numeric(order(-deviances))

## Un-comment to save results ##
#write.csv(scry_HVG_Order, "scry_HVG_Order.csv")