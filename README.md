# AI based Intrusion Detection for In-vehicle networks
This project is designed to train and test machine learning models on CANBUS data for intrusion detection purposes. It includes the following files:

- config.py: A configuration file that allows you to set the file path and the type of training to be performed.
- trainer.py: Trains the selected machine learning model on the given data.
- preprocess.py: Performs data preprocessing.
- main.py: The main Python code.

## Usage
 Edit the config.py file to specify the path to the data file and the type of training to be performed

You can choose one of the following training types:
- decision_tree
- mlp
- lstm
- xgboost
- random_forest
- svm

Once you have edited the config.py file, you can run main.py.
