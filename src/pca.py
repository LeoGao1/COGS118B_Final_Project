import numpy as np
import scipy

# calculate n_componenets dimension direction vectors
def PCA_calcV(n_components,data):
    #variance calculated
    covMat = np.cov(data.T)

    #eigen decomposition
    eigvals, V = scipy.sparse.linalg.eigsh(covMat)

    #sort the value from largest to smallest, store the index
    sortIndex = np.argsort(eigvals)
    sortIndex = np.flip(sortIndex)

    #sort the eigenvectors with the indices
    sortedVectors = np.zeros((len(V), n_components))

    for i in range(n_components):
        sortedVectors[:,i] = V[:,sortIndex[i]]
    return sortedVectors


# transform the data into n_componenets dimension
def PCA_transform(sortedVectors,data):
    c = np.matmul(sortedVectors.T,data.T)
    return c.T
