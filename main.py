from trainer import Trainer
from config import *
import sys
import os

if __name__ == "__main__":
    trainer = Trainer(FILE, use_IAT=True)

    if TRAIN_TYPE == mlp:
        loss, accuracy = trainer.mlp_train()
        print("loss: ", loss)
        print("accuracy: ", accuracy)
        
    elif TRAIN_TYPE == decision_tree:
        confusion_matrix, classification_report, classifier = trainer.decision_tree_train()
        print('Confusion Matrix:\n', confusion_matrix)
        print('Classification Report:\n', classification_report)

    elif TRAIN_TYPE == lstm:
        results = trainer.LSTM()
        print(results)

    elif TRAIN_TYPE == xgboost:
        results, confusion_matrix = trainer.xgboost()
        print(results)
        print(confusion_matrix)

        # Cross Validation
        k = 5
        scores = trainer.xgboost(k=k)
        print(scores)
        print(scores.mean())

    elif TRAIN_TYPE == random_forest:
        results = trainer.random_forest()
        print(results)

    elif TRAIN_TYPE == svm:
        print(trainer.svm())




    




    



