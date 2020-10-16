import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

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
    
    for i in range(Num):
        current_x = X[i,:]
        current_f = np.dot(current_x, W)
        current_f -= np.max(current_f)
        sum_f = np.sum(np.exp(current_f))
        
        loss += -np.log(np.exp(current_f[y[i]]) / sum_f)
       
        for k in range(Cat):
            P_k = np.exp(current_f[k]) / sum_f
            dW[:, k] += (P_k - (k == y[i])) * current_x
            
    loss /= Num
    loss += 0.5 * reg * np.sum(W * W)
    dW /= Num
    dW += 2 * reg * W
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    Dim, Cat = W.shape
    Num, Dim = X.shape
    
    f = np.dot(X, W)
    f -= np.max(f, axis=1, keepdims=True)
    sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
    P = np.exp(f) / sum_f
    
    loss = np.sum(-np.log(P[np.arange(Num), y]))
    
    p = np.zeros_like(P)
    p[np.arange(Num), y] = 1
    dW = np.dot(np.transpose(X), P-p)

    loss /= Num
    loss += 0.5 * reg * np.sum(W * W)
    dW /= Num
    dW += 2 * reg * W
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW
