# DeepExpression
We propose DeepExpression, a densely connected convolutional neural network to predict gene expression using both promoter sequences and enhancer-promoter interactions. We demonstrate that our model consistently outperforms baseline methods not only in the classification of binary gene expression status but also in the regression of continuous gene expression levels, in both cross-validation experiments and cross-cell lines predictions. We show that sequential promoter information is more informative than experimental enhancer information while enhancer-promoter interactions are most beneficial from those within ±100 kbp around the TSS of a gene. We finally visualize motifs in both promoter and enhancer regions and show the match of identified sequence signatures and known motifs. We expect to see a wide spectrum of applications using HiChIP data in deciphering the mechanism of gene regulation.

# Requirements
Keras
numpy
sklearn
random
