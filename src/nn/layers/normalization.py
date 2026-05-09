from .layer import Layer
from ..engine import Tensor
from ..initializers import INITIALIZATIONS
import numpy as np

class RMSNorm(Layer):
   def __init__(self, eps=1e-8):
      super().__init__()
      self.eps = eps

   def forward(self, inputs):
      n = inputs.data.shape[-1]
      RMS = ((inputs ** 2).sum(axis=-1, keepdims=True) / n + self.eps) ** 0.5 
      x_bar = (inputs / RMS) * self.gamma # following: https://openreview.net/pdf?id=SygkZ3MTJE  
      return x_bar
   
   def build(self, input_shape):
      super().build(input_shape)
      size = input_shape[-1]
      self.gamma = Tensor(np.ones((1, size)), requires_grad=True)
      self.output_shape = input_shape