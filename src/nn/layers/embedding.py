from .layer import Layer
from ..engine import Tensor
from ..initializers import INITIALIZATIONS
import numpy as np

class Embedding(Layer):
   def __init__(self, vocab_size, embedding_dim, weight_init='uniform', seed=None):
      super().__init__()
      self.vocab_size = vocab_size
      self.embedding_dim = embedding_dim

      rng = np.random.default_rng(seed)
      initializer = INITIALIZATIONS[weight_init]
      
      self.weights = initializer((vocab_size, embedding_dim), rng=rng)

   def forward(self, inputs):
      inputs = inputs.astype(int) if isinstance(inputs, np.ndarray) else inputs.data.astype(int)
      return self.weights[inputs] # same as doing multiplication by one-hot encoded vectors 