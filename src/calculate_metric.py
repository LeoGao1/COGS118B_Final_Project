import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#load the data for one experiment, parameter: experiment_number
from read_data import read_exp

#apply PCA, parameters: n_components,data
from pca import PCA_calcV,PCA_transform


def rgb2gray(rgb):
    return np.dot(rgb[:,:,:,:3], [0.2989, 0.5870, 0.1140])

def model_metrics (model_name, model_func,if_gray_scale, if_pca, pca_dim = 2):


    #create a panda dataframe to save the result
    result = pd.DataFrame(columns =['classifier','exp_num','train_acc','test_acc','train_f1','test_f1'])

    for exp_num in range(6):

        exp_num = exp_num +1
        #load data
        x_train,y_train,x_test,y_test = read_exp(exp_num)
        if (if_gray_scale):

            x_train= rgb2gray(x_train)
            x_test = rgb2gray(x_test)
            x_train = np.reshape(x_train, (x_train.shape[0],100*100))
            x_test = np.reshape(x_test, (x_test.shape[0],100*100))
        else:
            x_train = np.reshape(x_train, (x_train.shape[0],100*100*3))
            x_test = np.reshape(x_test, (x_test.shape[0],100*100*3))

        #apply pca
        if(if_pca):

            vecs = PCA_calcV(pca_dim,x_train)
            x_train = PCA_transform(vecs,x_train)
            x_test = PCA_transform(vecs,x_test)

        #build model
        y_train_pred, y_test_pred = model_func(x_train,y_train,x_test,y_test)

        #calculate metrics
        train_sc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred,average='micro')
        test_sc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred,average='micro')

        #save the result
        temp = {'classifier': model_name,
                'exp_num': exp_num,
                'train_acc':train_sc,
                'test_acc':test_sc,
                'train_f1':train_f1,
                'test_f1':test_f1}

        result = result.append(temp,ignore_index=True)

    return result
