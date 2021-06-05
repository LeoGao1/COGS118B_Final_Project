
import numpy as np

def read_exp(exp_num):
    name1 = '../data/x_train_'+str(exp_num)+'.npy'
    name2 = '../data/y_train_'+str(exp_num)+'.npy'
    name3 = '../data/x_test_'+str(exp_num)+'.npy'
    name4 = '../data/y_test_'+str(exp_num)+'.npy'

    x_train = np.load(name1)
    y_train = np.load(name2)
    x_test = np.load(name3)
    y_test = np.load(name4)

    print('Experiment: ' + str(exp_num))
    print('The data are x_train,y_train,x_test,y_test')

    return x_train,y_train,x_test,y_test
