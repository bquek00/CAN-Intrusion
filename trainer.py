import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from preprocess import load_data
from xgboost import XGBClassifier

class Trainer:
    """A class for training models on CAN bus data."""

    def __init__(self, dataset, use_IAT = False):
        self.use_IAT = use_IAT
        self.dataset = dataset

        if use_IAT == False:
            self.fn = ['Timestamp', 'Arbitration_ID', 'DLC', 'Data']
        else:
            self.fn = ['Timestamp', 'Arbitration_ID', 'DLC', 'Data', 'IAT']

        self.processed_data = None

    def mlp_train(self):
        """
        Trains the given data using MLP. Return test lost and accuracy
        """
        self.processed_data = load_data(self.dataset, self.use_IAT, NN = True, numerical=True)

        X_train, X_test, y_train, y_test = self.processed_data
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)

        X_train -= mean
        X_train /= std
        X_test -= mean
        X_test /= std

        ln = X_train.shape[1]

        # Create a perceptron NN
        model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape = ln,),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4)
        ])

        model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=3)

        loss, accuracy = model.evaluate(X_test, y_test) 
        return loss, accuracy

    def LSTM(self, randomise_window = False):
        """Train using LSTM. Return loss and accuracy"""
        self.processed_data = load_data(self.dataset, self.use_IAT, NN = True, numerical=True, window=10, 
                                        randomise_window=randomise_window)
        X_train, X_test, y_train, y_test = self.processed_data
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)

        X_train -= mean
        X_train /= std
        X_test -= mean
        X_test /= std

        output_size = 4
        input_size= X_train.shape[2]
        learning_rate = 0.001

        model = tf.keras.Sequential()
        model.add(tf.keras.Input (shape=(None, input_size)))
        model.add(tf.keras.layers.LSTM(128, activation="relu"))
        model.add(tf.keras.layers.Dense(output_size, activation="softmax"))

        print(model.summary())

        model.compile(
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"])
        model.fit(X_train, y_train, batch_size=64, epochs=5)
        print("Evaluate results")
        results = model.evaluate (X_test, y_test, batch_size=64)

        return results

    def xgboost(self, k = None):
        """
        Train using XGBoost

        Args:
        k (int): the number of folds for cross-validation. If None, the data 
          set will be split into 80/20 and the accuracy and confusion matrix 
          will be returned

        Returns:
        Cross validation score if K is provided. Accuracy and confusion matrix
        of the dataset if not. 
        """
        model = XGBClassifier()
        
        if k == None:
            self.processed_data = load_data(self.dataset, self.use_IAT, NN=False, numerical=True)
            X_train, X_test, y_train, y_test = self.processed_data

            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            return "Test set score: {:.4f}".format(model.score(X_test, y_test)), confusion_matrix(y_test, y_pred)
            

        X, y = self.processed_data = load_data(self.dataset, self.use_IAT, NN=False, numerical=True, split=False)
        kf = KFold(n_splits=k, shuffle=True, random_state=0)
        scores = cross_val_score(model, X, y, cv=kf)
        return scores

    def decision_tree_train(self):
        """
        Trains the given data.

        Args:
            Data properly formatted and split into test and train subsets

        Returns:
            The confusion matrix, classification report and an object of the
            decision tree classifier. 
        """

        self.processed_data = load_data(self.dataset, self.use_IAT)

        X_train, X_test, y_train, y_test = self.processed_data
        classifier = DecisionTreeClassifier(max_depth=4)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test) 
        return (confusion_matrix(y_test, y_pred), 
                classification_report(y_test, y_pred),
                classifier)

    def random_forest(self):
        """Train using Random Forest. Return accuracy"""
        self.processed_data = load_data(self.dataset, self.use_IAT)

        X_train, X_test, y_train, y_test = self.processed_data
        classifier = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        results = accuracy_score(y_test, y_pred)
        return results
    
    def svm(self):
        """Train using SVM. Return accuracy"""
        self.processed_data = load_data(self.dataset, self.use_IAT)

        X_train, X_test, y_train, y_test = self.processed_data
        classifier = svm.SVC(kernel='linear')
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        return accuracy_score(y_test, y_pred)   

    

 