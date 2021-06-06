import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#Modified from #http://www.oranlooney.com/post/ml-from-scratch-part-5-gmm/
#Works on 2 dimension 2 class data, in order to fit our datasets.
#The algorithm clusters x_train with y_train to initialize the means.
#X_test and y_test are used to do the classificationtask in calculate_metices.py
def gmm_2d_2class(x_train,y_train,x_test,y_test):
    n,m = x_train.shape
    phi = np.full(shape=2, fill_value=1/2)
    weights = np.full(shape=(n,m), fill_value=1/2)
    
    #initialize the means of data 
    class0_mean = x_train[y_train == 0].mean(axis = 0)
    class1_mean = x_train[y_train == 1].mean(axis = 0)
    means = [class0_mean,class1_mean]
    covMat = [ np.cov(x_train.T) for _ in range(2) ]
    
    # 40 iteratins should be enough for our task.
    for iteration in range(40):
        # E-Step: update weights and phi holding means and covMat constant
        likelihood = np.zeros( (n, 2) )
        for i in range(2):
            distribution = multivariate_normal(mean=means[i], cov=covMat[i])
            likelihood[:,i] = distribution.pdf(x_train)
        
        numerator = likelihood * phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        
        phi = weights.mean(axis=0)    
            
        # M-Step: update means and covMat holding phi and weights constant
        for i in range(2):
            weight = weights[:, [i]]
            total_weight = weight.sum()
            means[i] = (x_train * weight).sum(axis=0) / total_weight
            covMat[i] = np.cov(x_train.T, aweights=(weight/total_weight).flatten(), bias=True)

    training_predicted = np.argmax(weights, axis=1)
    #label test datapoint based on the likelihood of falling in which cluster
    n_test = len(x_test)
    likelihood_test = np.zeros( (n_test, 2) )
    for i in range(2):
        distribution = multivariate_normal(mean=means[i], cov=covMat[i])
        likelihood_test[:,i] = distribution.pdf(x_test)
        
    numerator = likelihood_test * phi
    denominator = numerator.sum(axis=1)[:, np.newaxis]
    weights = numerator / denominator

    test_pred = np.argmax(weights, axis=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for i in range(len(x_train)):
            if y_train[i]==0:
                ax1.scatter(x_train[i,0],x_train[i,1],color = 'b')
            else:
                ax1.scatter(x_train[i,0],x_train[i,1],color = 'r')
    ax1.set_title("PCA_data")
  
    for i in range(len(x_train)):
        if training_predicted[i]==0:
            ax2.scatter(x_train[i,0],x_train[i,1],color = 'b')
        else:
            ax2.scatter(x_train[i,0],x_train[i,1],color = 'r')
    ax2.set_title("GMM_cluster")
    plt.show()
    return training_predicted, test_pred



#For 3 dimension PCA data
def gmm_3d_2class(x_train,y_train,x_test,y_test):
    n,m = x_train.shape
    phi = np.full(shape=2, fill_value=1/2)
    weights = np.full(shape=(n,m), fill_value=1/2)
    
    #initialize the means of data 
    class0_mean = x_train[y_train == 0].mean(axis = 0)
    class1_mean = x_train[y_train == 1].mean(axis = 0)
    means = [class0_mean,class1_mean]
    covMat = [ np.cov(x_train.T) for _ in range(2) ]
    
    # 40 iteratins should be enough for our task.
    for iteration in range(40):
        # E-Step: update weights and phi holding means and covMat constant
        likelihood = np.zeros( (n, 2) )
        for i in range(2):
            distribution = multivariate_normal(mean=means[i], cov=covMat[i])
            likelihood[:,i] = distribution.pdf(x_train)
        
        numerator = likelihood * phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        
        phi = weights.mean(axis=0)    
            
        # M-Step: update means and covMat holding phi and weights constant
        for i in range(2):
            weight = weights[:, [i]]
            total_weight = weight.sum()
            means[i] = (x_train * weight).sum(axis=0) / total_weight
            covMat[i] = np.cov(x_train.T, aweights=(weight/total_weight).flatten(), bias=True)

    training_predicted = np.argmax(weights, axis=1)
    #label test datapoint based on the likelihood of falling in which cluster
    n_test = len(x_test)
    likelihood_test = np.zeros( (n_test, 2) )
    for i in range(2):
        distribution = multivariate_normal(mean=means[i], cov=covMat[i])
        likelihood_test[:,i] = distribution.pdf(x_test)
        
    numerator = likelihood_test * phi
    denominator = numerator.sum(axis=1)[:, np.newaxis]
    weights = numerator / denominator

    test_pred = np.argmax(weights, axis=1)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    for i in range(len(x_train)):
        if y_train[i]==0:
            ax1.scatter(x_train[i,0],x_train[i,1],x_train[i,2],color = 'b')
        else:
            ax1.scatter(x_train[i,0],x_train[i,1],x_train[i,2],color = 'r')
    ax1.set_title("PCA_data")
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    for i in range(len(x_train)):
        if training_predicted[i]==0:
            ax2.scatter(x_train[i,0],x_train[i,1],x_train[i,2],color = 'b')
        else:
            ax2.scatter(x_train[i,0],x_train[i,1],x_train[i,2],color = 'r')
    ax2.set_title("GMM_cluster")
    plt.show()
    return training_predicted, test_pred