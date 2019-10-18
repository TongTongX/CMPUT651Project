import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# def readData(fileloc):
#     dataset = pd.read_csv(fileloc)
#     # to be edited: 
#     X = dataset.drop('Class', axis=1)
#     y = dataset['Class']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
#     return X_train, X_test, y_train, y_test

class SVM_Classifier:
    def __init__(self, _kernel, _degree):
        # (kernal, degree) can be: ('linear'), ('poly', 6), ('rbf'), ('sigmoid')
        self.kernel = _kernel
        self.degree = _degree
        if _degree != None:
            self.svclassifier = SVC(kernel=_kernel, degree=_degree)
        else:
            self.svclassifier = SVC(kernel = _kernel)

    def train(self, X_train, y_train):
        self.svclassifier.fit(X_train, y_train)
        return 

    def predictOnTest(self, X_test, y_test):
        y_pred = self.svclassifier.predict(X_test)
        print(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        return
        
        
