import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1] 
  num_train = X.shape[0]
  scores = X.dot(W)
  scores = scores - np.reshape(np.max(scores,axis = 1),(num_train,1))
  exp_scores = np.exp(scores)
  exp_scores_sum = np.sum(exp_scores, axis = 1)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
      sum = 0.0
      for j in range(num_classes):
          sum =  sum + np.exp(scores[i,j])
          ## example belonging to y[i] class
          if j == y[i] :
              dW[:,j] = dW[:,j] - X[i,:].transpose() * ( 1 - exp_scores[i,j] / exp_scores_sum[i] )            
          ## other than y[i] class
          else:
              dW[:,j] = dW[:,j] - X[i,:].transpose() * ( 0.0 - exp_scores[i,j] / exp_scores_sum[i] )            
      loss = loss - np.log( np.exp(scores[i,y[i]]) / sum )
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss = loss / num_train
  loss = loss + 0.5 * reg * np.sum(W * W)
  
  dW = dW / num_train
  dW = dW + reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1] 
  num_train = X.shape[0]
  scores = X.dot(W)
  scores = scores - np.reshape(np.max(scores,axis = 1),(num_train,1))
  exp_scores = np.exp(scores)
  exp_scores_sum = np.sum(exp_scores, axis = 1)
  
  class_prob = exp_scores / np.reshape(exp_scores_sum,(num_train,1))
  
  ## loss
  loss = np.sum( -1 * np.log( class_prob[ range(num_train), y ] ) )
  loss = loss / num_train  
  loss = loss + 0.5 * reg * np.sum( W * W )
  
  ## gradient 
  class_prob[range(num_train), y] = class_prob[range(num_train), y] - 1 
  dW = X.transpose().dot(class_prob)
  dW = dW / num_train
  dW = dW + reg * W
  return loss, dW

