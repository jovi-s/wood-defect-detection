# Wood Defect Detection

Defect Detection is a method where automated visual evaluation systems can be used
to control the product quality in an industry to meet its objective.

## Description

During various stages of a manufacturing process or product lifecycle, surface 
level defects may occur, such as missing and misaligned components, physical 
damages like scratches, cracks, holes, etc. To find these defects, either visual 
inspection is done by human experts, or Automatic Optical Inspection (AOI) is 
conducted using computer algorithms. Surface analysis technique is an algorithmic 
attempt to inspect a surface for possible defects. Defects could be of various 
types and many different methods are used for inspection.

This work is a demonstration of a potential wood industrial project involving 
the development of AOI techniques in a manufacturing or recycling environment. 
The set-up potentially could contain a high quality camera which sends images 
of components to a computer running defect detection algorithm. 

Problem Statement: Anomaly Detection and Localization

### Dataset

**MVTec Anomaly Detection (MVTec AD)**.  

Fine-tuning training for the wood category of this dataset was done using Google Colab.

### Model
This repository utilizes the following model:  
[PaDiM](https://www.arxiv-vanity.com/papers/2011.08785/) stands for Patch Distribution 
Modeling and it generates patch embeddings for anomaly localization. 
It makes use of a pretrained convolutional neural network (CNN: ResNet-18) for embedding extraction 
and has the two following properties:  
 1. Each patch position is described by a multivariate Gaussian distribution
 2. PaDiM takes into account the correlations between different semantic levels of a pretrained CNN

The following features are embedded within the model and `anomalib` package:
 1. Embedding Extraction
 2. Normality Learning
 3. Anomaly Map Computation  
    - Anomaly Score using Mahalanobis distance where the final anomaly score of the entire image is the maximum of the anomaly map M


## Getting Started

### Installing

1. git clone/ download zip
2. cd into dir
3.
```
pip install -r /path/to/requirements.txt
```

### Executing program

- Dashboard
```
python main.py
```

## Inference

Repository is containerized.

## Authors

[Jovinder Singh](https://github.com/jovi-s/)

## Acknowledgments

- [Anomalib](https://github.com/openvinotoolkit/anomalib)
