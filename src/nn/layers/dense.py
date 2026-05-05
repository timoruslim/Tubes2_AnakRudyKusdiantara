from .layer import Layer
from ..engine import Tensor
from ..initializers import INITIALIZATIONS
import numpy as np

class Dense(Layer):
   def __init__(self, input_size, output_size, weight_init='xavier', bias_init='zero', l1_lambda=0.0, l2_lambda=0.0, seed=None):
      super().__init__()
      self.input_size = input_size
      self.output_size = output_size
      self.l1_lambda = l1_lambda
      self.l2_lambda = l2_lambda

      rng = np.random.default_rng(seed)
      weight_initializer = INITIALIZATIONS[weight_init]
      bias_initializer = INITIALIZATIONS[bias_init]
      
      self.weights = weight_initializer((input_size, output_size), rng=rng)
      self.bias = bias_initializer((1, output_size), rng=rng)

   def forward(self, inputs):
      return inputs @ self.weights + self.bias
   
   def regularization_loss(self):
      reg_loss = Tensor(0.0) 
      if self.l1_lambda > 0:
         reg_loss += self.l1_lambda * abs(self.weights).sum()
      if self.l2_lambda > 0:
         reg_loss += self.l2_lambda * (self.weights ** 2).sum()
      return reg_loss

