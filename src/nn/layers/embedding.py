from .layer import Layer
from ..engine import Tensor
from ..initializers import INITIALIZATIONS
import numpy as np

class Embedding(Layer):
   def __init__(self, vocab_size, embedding_dim, weight_init='uniform', l1_lambda=0.0, l2_lambda=0.0, seed=None):
      super().__init__()
      
      self.vocab_size = vocab_size
      self.embedding_dim = embedding_dim

      self.l1_lambda = l1_lambda
      self.l2_lambda = l2_lambda

      rng = np.random.default_rng(seed)
      initializer = INITIALIZATIONS[weight_init]
      
      self.weights = initializer((vocab_size, embedding_dim), rng=rng)

   def forward(self, inputs):
      inputs = inputs.astype(int) if isinstance(inputs, np.ndarray) else inputs.data.astype(int)
      return self.weights[inputs] # same as doing multiplication by one-hot encoded vectors 
   
   def regularization_loss(self):
      reg_loss = Tensor(0.0) 
      if self.l1_lambda > 0:
         reg_loss += self.l1_lambda * abs(self.weights).sum()
      if self.l2_lambda > 0:
         reg_loss += self.l2_lambda * (self.weights ** 2).sum()
      return reg_loss