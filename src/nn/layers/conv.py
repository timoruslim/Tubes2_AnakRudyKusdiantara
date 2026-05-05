from .layer import Layer
from ..engine import Tensor
from ..initializers import INITIALIZATIONS
import numpy as np

class Conv2D(Layer):

   def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, weight_init='he', bias_init='zero', l1_lambda=0.0, l2_lambda=0.0, seed=None):
      
      super().__init__()
      self.input_channels = input_channels
      self.output_channels = output_channels

      self.kernel_size = kernel_size
      self.stride = stride
      self.padding = padding

      self.l1_lambda = l1_lambda
      self.l2_lambda = l2_lambda

      rng = np.random.default_rng(seed)
      weight_initializer = INITIALIZATIONS[weight_init]
      bias_initializer = INITIALIZATIONS[bias_init]
      
      self.weights = weight_initializer((kernel_size, kernel_size, input_channels, output_channels), rng=rng)
      self.bias = bias_initializer((1, 1, 1, output_channels), rng=rng)

   def forward(self, inputs):
      
      if self.padding > 0:
         p = self.padding
         pad_width = ((0, 0), (p, p), (p, p), (0, 0))
         inputs = inputs.pad(pad_width, mode='constant', constant_values=0)

      b, h_in, w_in, _ = inputs.data.shape # batch, height, width, channels
      k = self.kernel_size
      s = self.stride
      h_out = (h_in - k) // s + 1
      w_out = (w_in - k) // s + 1

      # hybrid loop-im2col  
      patches = []
      weights_flat = self.weights.reshape(-1, self.output_channels) # flatten weights to column vector

      for i in range(h_out):
         for j in range(w_out):
            patch = inputs[:, i*s:i*s+k, j*s:j*s+k, :].reshape(b, -1) # flatten receptive field to row vector
            out_patch = (patch @ weights_flat).reshape(b, 1, 1, self.output_channels) # mini im2col
            patches.append(out_patch)

      return Tensor.concatenate(patches, axis=1).reshape(b, h_out, w_out, self.output_channels) + self.bias
   
   def regularization_loss(self):
      reg_loss = Tensor(0.0) 
      if self.l1_lambda > 0:
         reg_loss += self.l1_lambda * abs(self.weights).sum()
      if self.l2_lambda > 0:
         reg_loss += self.l2_lambda * (self.weights ** 2).sum()
      return reg_loss
   
class LocallyConnected2D(Layer):
   
   def __init__(self, h_in, w_in, input_channels, output_channels, kernel_size, stride=1, padding=0, weight_init='he', bias_init='zero', l1_lambda=0.0, l2_lambda=0.0, seed=None):
      
      super().__init__()

      self.h_in = h_in
      self.w_in = w_in

      self.input_channels = input_channels
      self.output_channels = output_channels

      self.h_out = (h_in - kernel_size + 2 * padding) // stride + 1
      self.w_out = (w_in - kernel_size + 2 * padding) // stride + 1

      self.kernel_size = kernel_size
      self.stride = stride
      self.padding = padding

      self.l1_lambda = l1_lambda
      self.l2_lambda = l2_lambda

      rng = np.random.default_rng(seed)
      weight_initializer = INITIALIZATIONS[weight_init]
      bias_initializer = INITIALIZATIONS[bias_init]
      
      # flattened weights and biases for easier computation 
      self.weights = weight_initializer((self.h_out * self.w_out, kernel_size * kernel_size * input_channels, output_channels), rng=rng) 
      self.bias = bias_initializer((self.h_out * self.w_out, 1, output_channels), rng=rng)

   def forward(self, inputs):
      
      if self.padding > 0:
         p = self.padding
         pad_width = ((0, 0), (p, p), (p, p), (0, 0))
         inputs = inputs.pad(pad_width, mode='constant', constant_values=0)

      b, _, _, _ = inputs.data.shape # batch, height, width, channels
      k = self.kernel_size
      s = self.stride
      h_out = self.h_out
      w_out = self.w_out

      patches = []
      for i in range(h_out):
         for j in range(w_out):
            loc = i * w_out + j
            patch = inputs[:, i*s:i*s+k, j*s:j*s+k, :].reshape(b, -1) # flatten receptive field to row vector
            local_w = self.weights[loc]
            local_b = self.bias[loc]  
            out_patch = (patch @ local_w + local_b).reshape(b, 1, 1, self.output_channels) 
            patches.append(out_patch)

      return Tensor.concatenate(patches, axis=1).reshape(b, h_out, w_out, self.output_channels)

   def regularization_loss(self):
      reg_loss = Tensor(0.0) 
      if self.l1_lambda > 0:
         reg_loss += self.l1_lambda * abs(self.weights).sum()
      if self.l2_lambda > 0:
         reg_loss += self.l2_lambda * (self.weights ** 2).sum()
      return reg_loss