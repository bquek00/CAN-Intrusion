# AI based Intrusion Detection for In-vehicle networks
This project is designed to train and test machine learning models on CAN bus data for intrusion detection purposes. It includes the following files:

## The main project
- project.ipynb: This file contains the my full project where I pre-processed the data, trained the model, tuned the hyper parameters, and evaluated the final model. It is in the form of a Jupyter notebook and includes all my results. 

## Basic code I wrote when first starting on this project
The following incliudes the basic source code I wrote when I first started this project. It focuses on first pre-processing the data for both classical and deep learning models and training it on an option of various different machine learning algorithms. 

- config.py: A configuration file that allows you to set the file path and the type of training to be performed.
- trainer.py: Trains the selected machine learning model on the given data.
- preprocess.py: Performs data preprocessing.
- main.py: The main Python code.

### Usage
 Edit the config.py file to specify the path to the data file and the type of training to be performed

You can choose one of the following training types:
- decision_tree
- mlp
- lstm
- xgboost
- random_forest
- svm

Once you have edited the config.py file, you can run main.py.
