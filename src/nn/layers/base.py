
class Layer:
   def __init__(self):
      self.trainable = True
      self.cache = None

   def __call__(self, inputs):
      return self.forward(inputs)

   def forward(self, inputs):
      raise NotImplementedError

   def backward(self, grad_outputs):
      raise NotImplementedError
   
   def get_weights(self):
      return [] 
   
   def set_weights(self, weights):
      pass