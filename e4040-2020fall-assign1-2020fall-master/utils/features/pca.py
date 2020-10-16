import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    # TODO: Implement PCA by extracting        #
    # eigenvector.You may need to sort the      #
    # eigenvalues to get the top K of them.     #
    ###############################################
    ###############################################
    #          START OF YOUR CODE         #
    ###############################################

    #u,d,v = np.linalg.svd(X)
    #Index = index_lst(d, rate=0.95)  # choose how many main factors
    #P = np.dot(X, u[:,:K])
    
    #T = np.zeros(K)
    
    cov = np.cov(np.transpose(X)) #D*D
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    T = np.array(eigenvalues[:K])
    P = np.transpose(eigenvectors[:,:K])
    
    ###############################################
    #           END OF YOUR CODE         #
    ###############################################
    
    return (P, T)
