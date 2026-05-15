from .layer import Layer
from ..engine import Tensor
from ..initializers import INITIALIZATIONS
import numpy as np

class Embedding(Layer):
   def __init__(self, vocab_size, embedding_dim, weight_init='uniform', l1_lambda=0.0, l2_lambda=0.0, seed=None):
      super().__init__()

      if weight_init not in INITIALIZATIONS:
         raise ValueError(f"Unknown weight_init '{weight_init}'. Available: {list(INITIALIZATIONS.keys())}")
      if vocab_size <= 0:
         raise ValueError(f"vocab_size must be strictly positive, got {vocab_size}.")
      if embedding_dim <= 0:
         raise ValueError(f"embedding_dim must be strictly positive, got {embedding_dim}.")
      
      self.vocab_size = vocab_size
      self.embedding_dim = embedding_dim

      self.l1_lambda = l1_lambda
      self.l2_lambda = l2_lambda

      self.rng = np.random.default_rng(seed)
      self.weight_initializer = INITIALIZATIONS[weight_init]
      self.weights = self.weight_initializer((self.vocab_size, self.embedding_dim), rng=self.rng)

   def forward(self, inputs):
      indices = inputs.astype(int) if isinstance(inputs, np.ndarray) else inputs.data.astype(int)
      if np.any(indices < 0) or np.any(indices >= self.vocab_size):
         raise ValueError(f"Embedding indices must be in range [0, {self.vocab_size}), got min={indices.min()}, max={indices.max()}")
      return self.weights[indices] 
   
   def build(self, input_shape):
      super().build(input_shape)
      self.output_shape = (*input_shape, self.embedding_dim)
   
   @classmethod
   def from_weights(cls, W):
      from ..base import Module
      obj = cls.__new__(cls)
      Module.__init__(obj)
      obj.vocab_size = W.shape[0]
      obj.embedding_dim = W.shape[1]
      obj.l1_lambda = 0.0
      obj.l2_lambda = 0.0
      obj.weights = Tensor(W)
      obj.built = True
      return obj

   def regularization_loss(self):
      reg_loss = Tensor(0.0) 
      if self.l1_lambda > 0:
         reg_loss += self.l1_lambda * abs(self.weights).sum()
      if self.l2_lambda > 0:
         reg_loss += self.l2_lambda * (self.weights ** 2).sum()
      return reg_loss