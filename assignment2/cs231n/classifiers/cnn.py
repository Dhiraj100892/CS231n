import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    W1 = np.random.normal( 0.0, weight_scale, ( num_filters, C, filter_size, filter_size ) )
    b1 = np.zeros(num_filters)
    
    stride = 1
    pad = ( filter_size - 1 ) / 2    
    
    H_2 = 1 + ( H + 2 * pad - filter_size ) / stride
    W_2 = 1 + ( W + 2 * pad - filter_size ) / stride
    
    H_2 = ( H_2 - 2 ) / 2 + 1
    W_2 = ( W_2 - 2 ) / 2 + 1
    
    W2 = np.random.normal( 0.0, weight_scale, ( num_filters * H_2 * W_2, hidden_dim))
    b2 = np.zeros( hidden_dim )
    
    W3 = np.random.normal( 0.0, weight_scale, (hidden_dim, num_classes))
    b3 = np.zeros( num_classes )
    
    self.params.update({'W1':W1})
    self.params.update({'W2':W2})
    self.params.update({'W3':W3})
    self.params.update({'b1':b1})
    self.params.update({'b2':b2})
    self.params.update({'b3':b3})
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # 1st layer    
    out_conv_1, cache_conv_1 = conv_forward_fast(X, W1, b1, conv_param)
    out_relu_1, cache_relu_1 = relu_forward( out_conv_1 )
    out_pool_1, cache_pool_1  = max_pool_forward_fast(out_relu_1, pool_param)
    
    #2nd layer
    out_affine_2, cache_affine_2 = affine_forward( out_pool_1, W2, b2)
    out_relu_2, cache_relu_2 = relu_forward( out_affine_2 )
    
    #3rd layer
    out_affine_3, cache_affine_3 = affine_forward( out_relu_2, W3, b3)
    scores = out_affine_3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dx = softmax_loss( scores, y )
    dx_affine_3, dW3, db3 = affine_backward( dx, cache_affine_3)
    
    dx_relu_2 = relu_backward(dx_affine_3, cache_relu_2)
    dx_affine_2, dW2, db2 = affine_backward( dx_relu_2, cache_affine_2)
    
    dx_pool_1 = max_pool_backward_fast( dx_affine_2, cache_pool_1)
    dx_relu_1 = relu_backward( dx_pool_1, cache_relu_1)
    dx_conv_1, dW1, db1 = conv_backward_fast(dx_relu_1, cache_conv_1)
    
    loss = loss + 0.5 * self.reg * ( np.sum(W1*W1) + np.sum(b1*b1) + np.sum(W2*W2) + np.sum(b2*b2) + np.sum(W3*W3) + np.sum(b3*b3) )
    grads.update({('W1'):( dW1 + W1 * self.reg )})
    grads.update({('b1'):( db1 + b1 * self.reg )})
    grads.update({('W2'):( dW2 + W2 * self.reg )})
    grads.update({('b2'):( db2 + b2 * self.reg )})
    grads.update({('W3'):( dW3 + W3 * self.reg )})
    grads.update({('b3'):( db3 + b3 * self.reg )})
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
