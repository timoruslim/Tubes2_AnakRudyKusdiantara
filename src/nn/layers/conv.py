from .layer import Layer
from ..engine import Tensor
from ..initializers import INITIALIZATIONS
from ..activations import ACTIVATIONS
import numpy as np

class Conv2D(Layer):

   def __init__(self, output_channels, kernel_size, stride=1, padding=0, activation='relu', weight_init='he', bias_init='zero', l1_lambda=0.0, l2_lambda=0.0, seed=None):
      
      super().__init__()
      if activation not in ACTIVATIONS:
         raise ValueError(f"Unknown activation '{activation}'. Available: {list(ACTIVATIONS.keys())}")
      if weight_init not in INITIALIZATIONS:
         raise ValueError(f"Unknown weight_init '{weight_init}'. Available: {list(INITIALIZATIONS.keys())}")
      if bias_init not in INITIALIZATIONS:
         raise ValueError(f"Unknown bias_init '{bias_init}'. Available: {list(INITIALIZATIONS.keys())}")
      if output_channels <= 0:
         raise ValueError(f"output_channels must be strictly positive, got {output_channels}.")
      if kernel_size <= 0: 
         raise ValueError(f"kernel_size must be strictly positive, got {kernel_size}.")
      if stride <= 0: 
         raise ValueError(f"stride must be strictly positive, got {stride}.")
      if padding < 0: 
         raise ValueError(f"padding cannot be negative, got {padding}.")

      self.output_channels = output_channels
      self.kernel_size = kernel_size
      self.stride = stride
      self.padding = padding

      self.activation = ACTIVATIONS[activation] if activation else None

      self.l1_lambda = l1_lambda
      self.l2_lambda = l2_lambda

      self.rng = np.random.default_rng(seed)
      self.weight_initializer = INITIALIZATIONS[weight_init]
      self.bias_initializer = INITIALIZATIONS[bias_init]

   def build(self, input_shape):
      super().build(input_shape)

      self.h_in, self.w_in = input_shape[1], input_shape[2]
      self.input_channels = input_shape[3]

      self.h_out = (self.h_in - self.kernel_size + 2 * self.padding) // self.stride + 1
      self.w_out = (self.w_in - self.kernel_size + 2 * self.padding) // self.stride + 1
      
      self.output_shape = (input_shape[0], self.h_out, self.w_out, self.output_channels)

      self.weights = self.weight_initializer((self.kernel_size, self.kernel_size, self.input_channels, self.output_channels), rng=self.rng)
      self.bias = self.bias_initializer((1, 1, 1, self.output_channels), rng=self.rng)

   def forward(self, inputs):
      
      if self.padding > 0:
         p = self.padding
         pad_width = ((0, 0), (p, p), (p, p), (0, 0))
         inputs = inputs.pad(pad_width, mode='constant', constant_values=0)

      b = inputs.data.shape[0]

      # hybrid loop-im2col  
      patches = []
      weights_flat = self.weights.reshape(-1, self.output_channels) # flatten weights to column vector

      for i in range(self.h_out):
         for j in range(self.w_out):
            patch = inputs[:, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size, :].reshape(b, -1) # flatten receptive field to row vector
            out_patch = (patch @ weights_flat).reshape(b, 1, 1, self.output_channels) # mini im2col
            patches.append(out_patch)

      h = Tensor.concatenate(patches, axis=1).reshape(b, self.h_out, self.w_out, self.output_channels) + self.bias
      return self.activation(h) if self.activation else h
   
   def regularization_loss(self):
      reg_loss = Tensor(0.0) 
      if self.l1_lambda > 0:
         reg_loss += self.l1_lambda * abs(self.weights).sum()
      if self.l2_lambda > 0:
         reg_loss += self.l2_lambda * (self.weights ** 2).sum()
      return reg_loss
   
class LocallyConnected2D(Layer):
   
   def __init__(self, output_channels, kernel_size, stride=1, padding=0, activation='relu', weight_init='he', bias_init='zero', l1_lambda=0.0, l2_lambda=0.0, seed=None):
      
      super().__init__()
      if activation not in ACTIVATIONS:
         raise ValueError(f"Unknown activation '{activation}'. Available: {list(ACTIVATIONS.keys())}")
      if weight_init not in INITIALIZATIONS:
         raise ValueError(f"Unknown weight_init '{weight_init}'. Available: {list(INITIALIZATIONS.keys())}")
      if bias_init not in INITIALIZATIONS:
         raise ValueError(f"Unknown bias_init '{bias_init}'. Available: {list(INITIALIZATIONS.keys())}")
      if output_channels <= 0:
         raise ValueError(f"output_channels must be strictly positive, got {output_channels}.")
      if kernel_size <= 0: 
         raise ValueError(f"kernel_size must be strictly positive, got {kernel_size}.")
      if stride <= 0: 
         raise ValueError(f"stride must be strictly positive, got {stride}.")
      if padding < 0: 
         raise ValueError(f"padding cannot be negative, got {padding}.")

      self.output_channels = output_channels
      self.kernel_size = kernel_size
      self.stride = stride
      self.padding = padding

      self.activation = ACTIVATIONS[activation] if activation else None

      self.l1_lambda = l1_lambda
      self.l2_lambda = l2_lambda

      self.rng = np.random.default_rng(seed)
      self.weight_initializer = INITIALIZATIONS[weight_init]
      self.bias_initializer = INITIALIZATIONS[bias_init]

   def forward(self, inputs):
      
      if self.padding > 0:
         p = self.padding
         pad_width = ((0, 0), (p, p), (p, p), (0, 0))
         inputs = inputs.pad(pad_width, mode='constant', constant_values=0)

      b = inputs.data.shape[0]

      patches = []
      for i in range(self.h_out):
         for j in range(self.w_out):
            loc = i * self.w_out + j
            patch = inputs[:, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size, :].reshape(b, -1) # flatten receptive field to row vector
            local_w = self.weights[loc]
            local_b = self.bias[loc]  
            out_patch = (patch @ local_w + local_b).reshape(b, 1, 1, self.output_channels) 
            patches.append(out_patch)

      h = Tensor.concatenate(patches, axis=1).reshape(b, self.h_out, self.w_out, self.output_channels) 
      return self.activation(h) if self.activation else h

   def build(self, input_shape):
      super().build(input_shape)

      self.h_in, self.w_in = input_shape[1], input_shape[2]
      self.input_channels = input_shape[3]

      self.h_out = (self.h_in - self.kernel_size + 2 * self.padding) // self.stride + 1
      self.w_out = (self.w_in - self.kernel_size + 2 * self.padding) // self.stride + 1
      
      self.output_shape = (input_shape[0], self.h_out, self.w_out, self.output_channels)

      # flattened weights and biases for easier computation 
      self.weights = self.weight_initializer((self.h_out * self.w_out, self.kernel_size * self.kernel_size * self.input_channels, self.output_channels), rng=self.rng) 
      self.bias = self.bias_initializer((self.h_out * self.w_out, 1, self.output_channels), rng=self.rng)


   def regularization_loss(self):
      reg_loss = Tensor(0.0) 
      if self.l1_lambda > 0:
         reg_loss += self.l1_lambda * abs(self.weights).sum()
      if self.l2_lambda > 0:
         reg_loss += self.l2_lambda * (self.weights ** 2).sum()
      return reg_loss