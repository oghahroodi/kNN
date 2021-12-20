import csv
import numpy as np
import pandas as pd
import random

def read_data(address, target):
    col = list(pd.read_csv(address, nrows=1).columns)
    target = col.index(target)
    data = pd.read_csv(address).to_numpy()
    Y = data.T[target].T
    X = np.delete(data.T, target, 0).T
    return X, Y

def shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize]
    Y = Y[randomize]
    return X, Y

def split(X, Y, train_percentage):
    X_train = X[:int(train_percentage*len(X))]
    Y_train = Y[:int(train_percentage*len(X))]
    X_test = X[int(train_percentage*len(X)):]
    Y_test = Y[int(train_percentage*len(X)):]
    return X_train, Y_train, X_test, Y_test

def accuracy(Y_pred, Y):
    return sum(1 for x, y in zip(Y_pred, Y) if x==y) / len(Y)

def matrix_confusion(Y_pred, Y):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(Y)):
        if(Y_pred[i]==Y[i] and Y[i]==1):
            TP+=1
        elif(Y_pred[i]==Y[i] and Y[i]==0):
            TN+=1
        elif(Y_pred[i]!=Y[i] and Y[i]==1):
            FN+=1
        elif(Y_pred[i]!=Y[i] and Y[i]==0):
            FP+=1
    return TP, TN, FP, FN

def report_classification(Y_pred, Y):
    TP, TN, FP, FN = matrix_confusion(Y_pred, Y)
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    Specificity = TN/(TN+FP)
    f1 = (2*Precision*Recall)/(Precision+Recall)
    return Accuracy, Precision, Recall, Specificity, f1
    