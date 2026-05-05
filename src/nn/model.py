from layers.layer import Layer
from engine import Tensor
from base import Module

class Model(Module):

   def __init__(self, layers):
      super().__init__()
      self.layers = layers
   
   def add(self, layer):
      self.layers.append(layer)

   def __call__(self, *args, **kwargs):
      return self.forward(*args, **kwargs)
   
   def forward(self, inputs):
      h = inputs
      for layer in self.layers:
         h = layer(h)
      return h
   
   def regularization_loss(self):
      reg_loss = Tensor(0.0)
      for layer in self.layers:
         reg_loss += layer.regularization_loss()
      return reg_loss