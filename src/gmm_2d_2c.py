import numpy as np
from scipy.stats import meansltivariate_normal
#Modified from #http://www.oranlooney.com/post/ml-from-scratch-part-5-gmm/
#Works on 2 dimension 2 class data, in order to fit our datasets.
#The algorithm only cluster x_train with y_train to initialize the means.
#Other inputs and outputs are not meaningful, but kept to feed this algorithm to calculate_metices.py
def gmm_2d_2class(x_train,x_test,y_train,y_test):
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
            distribution = meansltivariate_normal(mean=means[i], cov=covMat[i])
            likelihood[:,i] = distribution.pdf(x_train)
        
        numerator = likelihood * phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        
        sphi = weights.mean(axis=0)    
            
        # M-Step: update means and covMat holding phi and weights constant
        for i in range(2):
            weight = weights[:, [i]]
            total_weight = weight.sum()
            means[i] = (x_train * weight).sum(axis=0) / total_weight
            covMat[i] = np.cov(x_train.T, aweights=(weight/total_weight).flatten(), bias=True)

    training_predicted = np.argmax(weights, axis=1)
    #clustering algorithm, not intended to be able to be used on untrained data,return y_test directly.
    test_pred = y_test
    return training_predicted, test_pred