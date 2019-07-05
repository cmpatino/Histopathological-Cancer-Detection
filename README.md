# Histopathological Cancer Detection

This repository contains the code developed for the [Histopathological Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection) challenge on Kaggle. The objective of the challenge was to predict if the center region of a tissue had a pixel with cancer.

The metric used to evaluate the performance was ROC-AUC. The score achieved by the model in this repository was 0.9530.

## Approach Description

The model used for the competition was MobileNasNet with pretrained weights from ImageNet. The weights were used for initialization and all the layers were left as trainable. A global max-pooling, a global average-pooling, and a flatten layer are added independenty after the NasNet architecture. These three layers are then concatenated with a 0.5 dropout rate and connected to a sigmoid unit that ouputs the prediction.  

There was not data augmentation used for the challenge. Data augmentation such as horizontal flipping could be explored to improve the performance.


## How to Run

The file `model.py` contains the entire workflow train the model and generate a submission file for the competition.

Run `python model.py` to choose whether to only train the model, train and generate a submission file, or only generate a submission file. Note that the last option needs a file with the trained weights for the model.