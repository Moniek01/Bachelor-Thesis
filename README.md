# Fall Detection using Machine Learning

## Dataset
This project used the FallAllD dataset, acquired via this link: https://dx.doi.org/10.21227/bnya-mn34.

## Description
The goal of this project is to train and test three machine learning models (kNN, Random Forest, Gradient Boosting) on fall detection using three different sensor placements. The test result would then be compared to find the best position for the sensors during fall detection.

## Files
- Data Exploration.ipynb: a python notebook containing the data exploration on the dataset before preprocessing
- Data pre-processing.ipynb: a python notebook containing the pre-processing of the data before training and testing the machine learning models
- GB.py: a python file containing the code for the training and testing of the Gradient Boosting Classifier
- GB results.csv: the test results after running GB.py
- RF.py: a python file containing the code for the training and testing of the Random Forest Classifier
- RF results.csv: the test results after running RF.py
- kNN.py: a python file containing the code for the training and testing of the k-Nearest Neighbors Classifier
- kNN results.csv: the test results after running kNN.py
- Results visualizations.ipynb: a python notebook containing the confusion matrices of the test results from GB.py, RF.py, and kNN.py
- Loss Decrement.ipynb: a python notebook containing the visualization of the trend of loss function decrement for gradient boosting
