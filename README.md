<h1 align="center">Implementation in MATLAB of the PLS algorithm for classification</h1>

## Authors
* **FERRARI Lorenzo**, postgraduate in Computer Engineering at University of Bergamo.
* **LEONI Lorenzo**, postgraduate in Computer Engineering at University of Bergamo.

## Descrption
Implementation of the PLS algorithm through a MATLAB class which allows:
* to estimate a classification model using the NIPALS algorithm;
* to validate and cross-validate it by providing some performance metrics;
* to predict new instances starting from the trained model;
* to compute the best reduction order;
* to perform a comparison with the PCA technique.

[Data_analysis.mlx](Scripts/Data_analysis.mlx) contains an example of how [PLS.m](Scripts/PLS.m) can be used to classify 
steel plates faults by using this [dataset](https://www.kaggle.com/datasets/uciml/faulty-steel-plates) available on Kaggle.
