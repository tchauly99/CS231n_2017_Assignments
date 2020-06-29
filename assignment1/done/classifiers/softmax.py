import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = np.dot(X, W)
  for i in range(num_train):
    scores_ = scores[i, :]
    scores_stable = scores_ - np.max(scores_)
    numerator = np.exp(scores_stable[y[i]])
    denominator = np.sum(np.exp(scores_stable))
    loss += -np.log(numerator/denominator)
    for j in range(num_class):
      coefficient = np.exp(scores_stable[j])/np.sum(np.exp(scores_stable))
      dW[:, j] += (coefficient - int(j==y[i]))*X[i, :]
  loss /= num_train
  loss += reg*np.sum(W*W)

  dW /= num_train
  dW += 2*reg*W



  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
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
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = np.dot(X, W)

  scores_stable = scores - np.array([np.max(scores, axis=1)]).T
  expo_scores = np.exp(scores_stable)
  sum = np.sum(expo_scores, axis=1)
  proportional = expo_scores/np.array([sum]).T

  mask_loss = np.zeros(proportional.shape, dtype=bool)
  mask_loss[range(mask_loss.shape[0]), y] = True
  pre_loss = np.array(proportional.shape)
  pre_loss = proportional
  pre_loss = proportional[mask_loss]

  loss = -np.sum(np.log(pre_loss))

  loss /= num_train
  loss += reg*np.sum(W*W)

  proportional[range(proportional.shape[0]), y] -= 1
  dW = np.dot(X.T, proportional)
  dW /= num_train
  dW += 2*reg*W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

