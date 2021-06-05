
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler

# run knn with sklearn with input data
def knnwithsklearn(x_train,y_train,x_test,y_test):

    # implement knn using sklearn
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(x_train, y_train)

    # predict y values according to x test
    y_predict = knn.predict(x_test)
    y_train_predict = knn.predict(x_train)

    # calculate accuracy of this trail
    acc = metrics.accuracy_score(y_predict, y_test)

    print("accuracy of this trail is "+ str(acc))

    return y_train_predict, y_predict
