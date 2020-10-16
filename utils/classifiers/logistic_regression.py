import numpy as np
from random import shuffle

def sigmoid(x):
    h = np.zeros_like(x)
    
    #############################################################################
    # TODO: Implement sigmoid function.                            #         
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    
    h += 1.0 / (1 + np.exp(-x))
    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################
    return h 

def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = None
    # Initialize the gradient to zero
    dW = None

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################

    Dim, Cat = W.shape
    Num, Dim = X.shape
    
    loss_sum = 0
    
    new_y = np.transpose(np.vstack((y,np.ones_like(y)-y)))
    for i in range(0, Num):
        current_X = X[i,:]
        current_f = np.dot(current_X, W)
        current_loss = 0
        for j in range(0, Cat):
            current_loss += new_y[i,j]*np.log(sigmoid(current_f[j])) + (1-new_y[i,j])*np.log(1-sigmoid(current_f[j]))
        loss_sum += -current_loss
        '''
        if y[i] == 1: #class 0
            current_loss = np.log(sigmoid(current_f[0])) + np.log(1-sigmoid(current_f[1]))
        elif y[i] == 0: # class 1
            current_loss = np.log(sigmoid(current_f[1])) + np.log(1-sigmoid(current_f[0]))
            
        loss_sum += current_loss
        '''
    
    new_y = np.transpose(np.vstack((y,np.ones_like(y)-y)))
    dw = np.zeros_like(W)
    
    for i in range(0, Dim):
        for j in range(0, Cat):
            dw[i,j] = np.dot(np.transpose(X)[i,:], sigmoid(np.dot(X, W[:,j])) - new_y[:,j]) / Num
        
    loss = loss_sum / Num
    dW = dw
    
    loss += 0.5 * reg * np.sum(W * W)
    dW += 2 * reg * W
    
    
    
    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = None
    # Initialize the gradient to zero
    dW = None

    ############################################################################
    # TODO: Compute the logistic regression loss and its gradient using no    # 
    # explicit loops.                                       #
    # Store the loss in loss and the gradient in dW. If you are not careful   #
    # here, it is easy to run into numeric instability. Don't forget the     #
    # regularization!                                       #
    ############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    f = np.dot(X, W)
    h = sigmoid(f)
    new_y = np.transpose(np.vstack((y,np.ones_like(y)-y)))
    loss = (new_y * np.log(h) + (1 - new_y) * np.log(1 - h))
    loss = -(loss[:,0] + loss[:,1]).mean()
    dW = np.dot(np.transpose(X), (h - new_y)) / new_y.shape[0]
    
    loss += 0.5 * reg * np.sum(W * W)
    dW += 2 * reg * W
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW
