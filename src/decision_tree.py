import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def build_DT (x_train,y_train,x_test,y_test):

    clf = DecisionTreeClassifier(max_depth= 15)
    clf.fit(x_train, y_train)

    #get train score
    y_train_pred = clf.predict(x_train)

    #get test score
    y_test_pred = clf.predict(x_test)

    return y_train_pred, y_test_pred
