<h1 align="center">Implementation in MATLAB of the PLS algorithm for classification</h1>

## Authors
* **FERRARI Lorenzo**, postgraduate in Computer Engineering at University of Bergamo.
* **LEONI Lorenzo**, postgraduate in Computer Engineering at University of Bergamo.

## Description
Implementation of the **discriminant PLS algorithm** through a MATLAB class. It provides the following features:
* *estimation* of a PLS model by using the NIPALS algorithm, both PLS1 and PLS2 versions;
* *validation* of the estimated model by providing not only the test MCE for each class, but also the test confusion matrix;
* *cross-validation* to find the best reduction order;
* *graphing* of the matrix T for orders 1, 2, and 3;
* *classification* of new data;

Moreover, [PLS.m](Scripts/PLS.m) can also estimate a PCA model, therefore it is possible to compare it with PLS.
##
