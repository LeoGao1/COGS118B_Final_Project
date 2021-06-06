import numpy as np
from sklearn.svm import SVC

def svm_model(x_train, y_train, x_test, y_test):

    clf = SVC()
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)

    #get test score
    y_test_pred = clf.predict(x_test)

    return y_train_pred, y_test_pred
