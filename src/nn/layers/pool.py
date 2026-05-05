from ..engine import Tensor
from .layer import Layer

class Flatten(Layer):

   def __init__(self):
      super().__init__()

   def forward(self, inputs):
      batch_size = inputs.data.shape[0]
      return inputs.reshape(batch_size, -1)
   
class GlobalMaxPooling2D(Layer):

   def __init__(self):
      super().__init__()
   
   def forward(self, inputs):
      return inputs.max(axis=(1,2))
   
class GlobalAveragePooling2D(Layer):

   def __init__(self):
      super().__init__()
   
   def forward(self, inputs):
      return inputs.mean(axis=(1,2))
   
class MaxPooling2D(Layer):
   def __init__(self, pool_size=2, stride=None, padding=0):
      super().__init__()
      self.pool_size = pool_size
      self.stride = pool_size if stride is None else stride 
      self.padding = padding
   
   def forward(self, inputs):

      if self.padding > 0:
         p = self.padding
         pad_width = ((0, 0), (p, p), (p, p), (0, 0))
         inputs = inputs.pad(pad_width, mode='constant', constant_values=0)

      b, h_in, w_in, c = inputs.data.shape
      k = self.pool_size
      
      if self.stride == self.pool_size and h_in % k == 0 and w_in % k == 0:
         reshaped = inputs.reshape(b, h_in // k, k, w_in // k, k, c)
         return reshaped.max(axis=(2, 4))
      
      else:
         s = self.stride
         h_out = (h_in - k) // s + 1
         w_out = (w_in - k) // s + 1
         patches = []
         for i in range(h_out):
            for j in range(w_out):
               patch = inputs[:, i*s:i*s+k, j*s:j*s+k, :]
               p_max = patch.max(axis=(1, 2), keepdims=True)
               patches.append(p_max)
         return Tensor.concatenate(patches, axis=1).reshape(b, h_out, w_out, c)
      
class AveragePooling2D(Layer):
   def __init__(self, pool_size=2, stride=None, padding=0):
      super().__init__()
      self.pool_size = pool_size
      self.stride = pool_size if stride is None else stride 
      self.padding = padding
   
   def forward(self, inputs):

      if self.padding > 0:
         p = self.padding
         pad_width = ((0, 0), (p, p), (p, p), (0, 0))
         inputs = inputs.pad(pad_width, mode='constant', constant_values=0)

      b, h_in, w_in, c = inputs.data.shape
      k = self.pool_size
      
      if self.stride == self.pool_size and h_in % k == 0 and w_in % k == 0:
         reshaped = inputs.reshape(b, h_in // k, k, w_in // k, k, c)
         return reshaped.mean(axis=(2, 4))
      
      else:
         s = self.stride
         h_out = (h_in - k) // s + 1
         w_out = (w_in - k) // s + 1
         patches = []
         for i in range(h_out):
            for j in range(w_out):
               patch = inputs[:, i*s:i*s+k, j*s:j*s+k, :]
               p_mean = patch.mean(axis=(1, 2), keepdims=True)
               patches.append(p_mean)
         return Tensor.concatenate(patches, axis=1).reshape(b, h_out, w_out, c)
