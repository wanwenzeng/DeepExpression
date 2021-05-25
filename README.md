# DeepExpression

## Introduction
We propose DeepExpression, a densely connected convolutional neural network to predict gene expression using both promoter sequences and enhancer-promoter interactions. We demonstrate that our model consistently outperforms baseline methods not only in the classification of binary gene expression status but also in the regression of continuous gene expression levels, in both cross-validation experiments and cross-cell lines predictions. We show that sequential promoter information is more informative than experimental enhancer information while enhancer-promoter interactions are most beneficial from those within ±100 kbp around the TSS of a gene. We finally visualize motifs in both promoter and enhancer regions and show the match of identified sequence signatures and known motifs. We expect to see a wide spectrum of applications using HiChIP data in deciphering the mechanism of gene regulation.

## Preprocessing
**DeepExpression receive 3 input files, and each row represents an gene:

* RNA-seq file: gene expression value of each gene.

* HiChIP file: Bins of HiChIP signal surrounding each gene.

* Sequence file: one-hot encoding matrix of each 2000bp promoter sequence.

All the preprocessed data can be found in Zenodo: https://zenodo.org/record/3040059

## Running coupleNMF
**DeepExpression receives 8 parameters:**

* -cl        the testing cell line

* -s         the testing species

* -plen      the length of promoter sequence

* -elen      the length of enhancer sequence

* -c         the pesudo count added to Y

* -lr        the learning rate 

* -batch     the size of one batch 


### Example

```
THEANO_FLAGS="device=cuda1" python deepexpression.py -cl mES -s mouse -c 0.1 -plen 200 -elen 40 -c 1 -batch 16 -lr 1e-5

```

## Requirements
* Keras
* numpy
* sklearn
* random

## License
This project is licensed under the MIT License - see the LICENSE.md file for details
