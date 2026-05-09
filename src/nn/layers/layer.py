from ..base import Module

class Layer(Module):
   def __init__(self):
      super().__init__()

   def __call__(self, *args, **kwargs):
      if not hasattr(self, 'built') or not self.built:
         raise RuntimeError(f"{self.__class__.__name__} has not been built yet. Call build(input_shape) or use Model to build automatically.")
      return self.forward(*args, **kwargs)

   def forward(self, *args, **kwargs):
      raise NotImplementedError(f"{self.__class__.__name__} must implement the forward() method.")
   
   def build(self, input_shape):
      self.input_shape = input_shape
      self.built = True

class Input(Layer):
   def __init__(self, shape):
      super().__init__()
      self.shape = shape
      self.output_shape = shape
      
   def build(self, input_shape):
      self.built = True
      
   def forward(self, inputs):
      return inputs 
