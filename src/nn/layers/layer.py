from ..base import Module

class Layer(Module):
   def __init__(self):
      super().__init__()
      self.trainable = True

   def __call__(self, *args, **kwargs):
      return self.forward(*args, **kwargs)

   def forward(self, *args, **kwargs):
      raise NotImplementedError