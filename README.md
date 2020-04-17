# Machine Learning

## Description
Collection of scripts used to train LightGBM models. These scripts were orginally written for the Prediction Molecular Properties
competition hosted on Kaggle in 2019. They have been modified to be more general and are meant to represent the pipeline that is used 
to generate the model. In summary, first features are generated, the feature are then (optionally) selected based on feature importance.
After that, hyperparameter optimization is performed using either a grid search or random search method before k-fold splitting the data 
and training it using a gradient boosted trees algorithm. 