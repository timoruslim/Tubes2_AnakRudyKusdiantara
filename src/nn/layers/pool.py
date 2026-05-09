from ..engine import Tensor
from .layer import Layer
import numpy as np

class Flatten(Layer):

   def __init__(self):
      super().__init__()

   def forward(self, inputs):
      batch_size = inputs.data.shape[0]
      return inputs.reshape(batch_size, -1)
   
   def build(self, input_shape):
      if len(input_shape) < 2:
         raise ValueError(f"Flatten expects at least 2D input (batch, features), but got shape {input_shape}. Did you forget the batch dimension?")
      super().build(input_shape)
      self.output_shape = (input_shape[0], int(np.prod(input_shape[1:])))
   
class GlobalMaxPooling2D(Layer):

   def __init__(self):
      super().__init__()
   
   def forward(self, inputs):
      return inputs.max(axis=(1,2))
   
   def build(self, input_shape):
      if len(input_shape) != 4:
         raise ValueError(f"{self.__class__.__name__} expects 4D input (batch, height, width, channels), but got shape {input_shape}. Did you forget the batch dimension?")
      super().build(input_shape)
      self.output_shape = (input_shape[0], input_shape[-1])
   
class GlobalAveragePooling2D(Layer):

   def __init__(self):
      super().__init__()
   
   def forward(self, inputs):
      return inputs.mean(axis=(1,2))
   
   def build(self, input_shape):
      if len(input_shape) != 4:
         raise ValueError(f"{self.__class__.__name__} expects 4D input (batch, height, width, channels), but got shape {input_shape}. Did you forget the batch dimension?")
      super().build(input_shape)
      self.output_shape = (input_shape[0], input_shape[-1])

class MaxPooling2D(Layer):
   def __init__(self, pool_size=2, stride=None, padding=0):
      super().__init__()
      if pool_size <= 0: 
         raise ValueError("pool_size must be strictly positive.")
      if stride is not None and stride <= 0: 
         raise ValueError("stride must be strictly positive.")
      if padding < 0: 
         raise ValueError("padding cannot be negative.")

      self.pool_size = pool_size
      self.stride = pool_size if stride is None else stride 
      self.padding = padding
   
   def forward(self, inputs):

      if self.padding > 0:
         p = self.padding
         pad_width = ((0, 0), (p, p), (p, p), (0, 0))
         inputs = inputs.pad(pad_width, mode='constant', constant_values=0)

      b, h_in, w_in, c = inputs.data.shape # batch, height, width, channels
      
      if self.stride == self.pool_size and h_in % self.pool_size == 0 and w_in % self.pool_size == 0:
         reshaped = inputs.reshape(b, h_in // self.pool_size, self.pool_size, w_in // self.pool_size, self.pool_size, c)
         return reshaped.max(axis=(2, 4))
      
      else:
         patches = []
         for i in range(self.h_out):
            for j in range(self.w_out):
               patch = inputs[:, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size, :]
               p_max = patch.max(axis=(1, 2), keepdims=True)
               patches.append(p_max)
         return Tensor.concatenate(patches, axis=1).reshape(b, self.h_out, self.w_out, c)
      
   def build(self, input_shape):
      if len(input_shape) != 4:
         raise ValueError(f"{self.__class__.__name__} expects 4D input (batch, height, width, channels), but got shape {input_shape}. Did you forget the batch dimension?")
      super().build(input_shape)
      self.h_out = (input_shape[1] - self.pool_size + 2 * self.padding) // self.stride + 1
      self.w_out = (input_shape[2] - self.pool_size + 2 * self.padding) // self.stride + 1
      self.output_shape = (input_shape[0], self.h_out, self.w_out, input_shape[3])

class AveragePooling2D(Layer):
   def __init__(self, pool_size=2, stride=None, padding=0):
      super().__init__()
      if pool_size <= 0:
         raise ValueError("pool_size must be strictly positive.")
      if stride is not None and stride <= 0:
         raise ValueError("stride must be strictly positive.")
      if padding < 0:
         raise ValueError("padding cannot be negative.")
   
      self.pool_size = pool_size
      self.stride = pool_size if stride is None else stride 
      self.padding = padding
   
   def forward(self, inputs):

      if self.padding > 0:
         p = self.padding
         pad_width = ((0, 0), (p, p), (p, p), (0, 0))
         inputs = inputs.pad(pad_width, mode='constant', constant_values=0)

      b, h_in, w_in, c = inputs.data.shape # batch, height, width, channels
      
      if self.stride == self.pool_size and h_in % self.pool_size == 0 and w_in % self.pool_size == 0:
         reshaped = inputs.reshape(b, h_in // self.pool_size, self.pool_size, w_in // self.pool_size, self.pool_size, c)
         return reshaped.mean(axis=(2, 4))
      
      else:
         patches = []
         for i in range(self.h_out):
            for j in range(self.w_out):
               patch = inputs[:, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size, :] # receptive field
               p_mean = patch.mean(axis=(1, 2), keepdims=True)
               patches.append(p_mean)
         return Tensor.concatenate(patches, axis=1).reshape(b, self.h_out, self.w_out, c)

   def build(self, input_shape):
      if len(input_shape) != 4:
         raise ValueError(f"{self.__class__.__name__} expects 4D input (batch, height, width, channels), but got shape {input_shape}. Did you forget the batch dimension?")
      super().build(input_shape)
      self.h_out = (input_shape[1] - self.pool_size + 2 * self.padding) // self.stride + 1
      self.w_out = (input_shape[2] - self.pool_size + 2 * self.padding) // self.stride + 1
      self.output_shape = (input_shape[0], self.h_out, self.w_out, input_shape[3])