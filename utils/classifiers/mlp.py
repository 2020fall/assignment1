from builtins import range
from builtins import object
import numpy as np

from utils.layer_funcs import *
from utils.layer_utils import *

class MLP(object):
    """
    MLP with an arbitrary number of dense hidden layers,
    and a softmax loss function. For a network with L layers,
    the architecture will be

    input >> DenseLayer x (L - 1) >> AffineLayer >> softmax_loss >> output

    Here "x (L - 1)" indicate to repeat L - 1 times. 
    """
    def __init__(self, input_dim=3072, hidden_dims=[200,200], num_classes=10, reg=0.0, weight_scale=1e-3):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(DenseLayer(input_dim=dims[i], output_dim=dims[i+1], weight_scale=weight_scale))
        layers.append(AffineLayer(input_dim=dims[-1], output_dim=num_classes, weight_scale=weight_scale))
        
        self.layers = layers

    def loss(self, X, y):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        
        Inputs:
        - X: input data
        - y: ground truth
        
        Return loss value(float)
        """
        loss = 0.0
        reg = self.reg
        num_layers = self.num_layers
        layers = self.layers
        ####################################################
        # TODO: Feedforward                      #
        ####################################################
        ####################################################
        #           START OF YOUR CODE           #
        ####################################################

        den_layer_input = []
        den_layer_mid = []
        den_layer_output = []
        
        den_layer_input.append(X)
        final_layer_input = np.zeros_like(X)
        
        for i in range(num_layers-1):
            x = den_layer_input[i]
            
            xx = affine_forward(x, layers[i].params[0], layers[i].params[1])
            den_layer_mid.append(xx)
            
            r = relu_forward(xx)
            den_layer_output.append(r)
            
            if i != num_layers-2:
                den_layer_input.append(r)
            elif i == num_layers-2:
                final_layer_input = r
        
        score = affine_forward(final_layer_input, layers[num_layers-1].params[0], layers[num_layers-1].params[1])
        
        loss, dx = softmax_loss(score, y)
        
        ####################################################
        # TODO: Backpropogation                   #
        ####################################################
        dfinal_layer_input, dfinalW, dfinalb = affine_backward(dx, final_layer_input, layers[num_layers-1].params[0],\
                                                               layers[num_layers-1].params[1])
        dx = []
        dx.append(dfinal_layer_input)
        dW = []
        dW.append(dfinalW)
        db = []
        db.append(dfinalb)
        
        
        for i in range(num_layers-1):
            da = relu_backward(dx[i], den_layer_mid[-i-1])
            dxx, dWW, dbb = affine_backward(da, den_layer_input[-i-1], layers[-i-2].params[0], layers[-i-2].params[1])
            
            dx.append(dxx)
            dW.append(dWW)
            db.append(dbb)
        
        dx.reverse()
        dW.reverse()
        db.reverse()
        
        for i in range(num_layers):
            self.layers[i].gradients = [dW[i], db[i]]
            
        ####################################################
        # TODO: Add L2 regularization               #
        ####################################################
        
        square_weights = 0.0
        for i in range(num_layers):
            square_weights += np.sum(self.layers[i].params[0]**2)
            
        loss += 0.5*self.reg*square_weights
        
        ####################################################
        #            END OF YOUR CODE            #
        ####################################################
        
        return loss

    def step(self, learning_rate=1e-5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
        ####################################################
        # TODO: Use SGD to update variables in layers.    #
        ####################################################
        ####################################################
        #           START OF YOUR CODE           #
        ####################################################
        params = []
        grads = []
        for i in range(self.num_layers):
            params += self.layers[i].params
            grads += self.layers[i].gradients
            
        reg = self.reg  
        grads = [grad + reg*params[i] for i, grad in enumerate(grads)]
        
        for i in range(len(params)):
            params[i] += -learning_rate * grads[i]
        
        
        ####################################################
        #            END OF YOUR CODE            #
        ####################################################
   
        # update parameters in layers
        for i in range(self.num_layers):
            self.layers[i].update_layer(params[2*i:2*(i+1)])
        

    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        num_layers = self.num_layers
        layers = self.layers
        #####################################################
        # TODO: Remember to use functions in class       #
        # SoftmaxLayer                          #
        #####################################################
        ####################################################
        #           START OF YOUR CODE           #
        ####################################################

        for i in range(num_layers-1):
            X = affine_forward(X, layers[i].params[0], layers[i].params[1])
            X = relu_forward(X)
            
        z = affine_forward(X, layers[num_layers-1].params[0], layers[num_layers-1].params[1])
        predictions = np.argmax(z, axis=1)
        
        ####################################################
        #            END OF YOUR CODE            #
        ####################################################
        
        return predictions
    
    def check_accuracy(self, X, y):
        """
        Return the classification accuracy of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        - y: (int) an array of length N. ground truth label 
        Returns: 
        - acc: (float) between 0 and 1
        """
        y_pred = self.predict(X)
        acc = np.mean(np.equal(y, y_pred))
        
        return acc