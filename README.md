# MAFA

Codes for IJCNN'22 paper 'Chinese Sentence Matching with Multiple Alignments and Feature Augmentation'

This repo contains the implementation of "Chinese Sentence Matching with Multiple Alignments and Feature Augmentation" in Keras & Tensorflow.

# Usage for python code

## 0. Requirement

python 3.6  
numpy==1.16.4  
pandas==0.22.0  
tensorboard==1.12.0  
tensorflow-gpu==1.12.0  
keras==2.2.4  
gensim==3.0.0

## 1. Data preparation

The dataset is BQ & LCQMC.  
"The BQ Corpus: A Large-scale Domain-specific Chinese Corpus For Sentence Semantic Equivalence Identification", https://www.aclweb.org/anthology/D18-1536/.

"LCQMC: A Large-scale Chinese Question Matching Corpus", https://www.aclweb.org/anthology/C18-1166/.

## 2. Start the training process

python train.py

## 3. Start the prediction process

python predict.py
