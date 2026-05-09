from .layer import Layer
from ..engine import Tensor
from ..activations import ACTIVATIONS
from ..initializers import INITIALIZATIONS
import numpy as np

class Dense(Layer):
   def __init__(self, output_size, activation="relu", weight_init='xavier', bias_init='zero', l1_lambda=0.0, l2_lambda=0.0, seed=None):
      super().__init__()

      if activation not in ACTIVATIONS:
         raise ValueError(f"Unknown activation '{activation}'. Available: {list(ACTIVATIONS.keys())}")
      if weight_init not in INITIALIZATIONS:
         raise ValueError(f"Unknown weight_init '{weight_init}'. Available: {list(INITIALIZATIONS.keys())}")
      if bias_init not in INITIALIZATIONS:
         raise ValueError(f"Unknown bias_init '{bias_init}'. Available: {list(INITIALIZATIONS.keys())}")
      if output_size <= 0:
         raise ValueError(f"output_size must be strictly positive, got {output_size}.")

      self.input_size = None
      self.output_size = output_size
      
      self.activation = ACTIVATIONS.get(activation)
      self.l1_lambda = l1_lambda
      self.l2_lambda = l2_lambda

      self.rng = np.random.default_rng(seed)
      self.weight_initializer = INITIALIZATIONS[weight_init]
      self.bias_initializer = INITIALIZATIONS[bias_init]
   
   def forward(self, inputs):
      h = inputs @ self.weights + self.bias
      return self.activation(h) if self.activation else h
   
   def build(self, input_shape):
      super().build(input_shape)

      self.input_size = input_shape[-1]
      self.output_shape = (*input_shape[:-1], self.output_size)

      self.weights = self.weight_initializer((self.input_size, self.output_size), rng=self.rng)
      self.bias = self.bias_initializer((1, self.output_size), rng=self.rng) 

   def regularization_loss(self):
      reg_loss = Tensor(0.0) 
      if self.l1_lambda > 0:
         reg_loss += self.l1_lambda * abs(self.weights).sum()
      if self.l2_lambda > 0:
         reg_loss += self.l2_lambda * (self.weights ** 2).sum()
      return reg_loss

