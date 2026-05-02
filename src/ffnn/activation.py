from .engine import Tensor, unbroadcast, GradMode
import numpy as np
from scipy.special import erf

def linear(h):
   if not GradMode.enabled:
      return Tensor(h.data)
   
   id_h = Tensor(h.data, (h,))
   def _backward():
      h.grad += id_h.grad
   id_h._backward = _backward
   return id_h

def relu(h):
   if not GradMode.enabled:
      return Tensor(np.maximum(0, h.data))
   
   relu_h = Tensor(np.maximum(0, h.data), (h,))
   def _backward():
      h.grad += unbroadcast(relu_h.grad * (h.data > 0).astype(float), h.data.shape) # dL/dA = dL/dC * dC/dA = dL/dC * (A > 0)
   relu_h._backward = _backward
   return relu_h

def sigmoid(h):
   if not GradMode.enabled:
      return Tensor(1 / (1 + np.exp(-h.data)))
   
   sigmoid_h = Tensor(1 / (1 + np.exp(-h.data)), (h,))
   def _backward():
      h.grad += unbroadcast(sigmoid_h.grad * sigmoid_h.data * (1 - sigmoid_h.data), h.data.shape) # dL/dA = dL/dC * dC/dA = dL/dC * σ(A) * (1 - σ(A))
   sigmoid_h._backward = _backward
   return sigmoid_h

def tanh(h):
   if not GradMode.enabled:
      return Tensor(np.tanh(h.data))
   
   tanh_h = Tensor(np.tanh(h.data), (h,))
   def _backward():
      h.grad += unbroadcast(tanh_h.grad * (1 - tanh_h.data ** 2), h.data.shape) # dL/dA = dL/dC * dC/dA = dL/dC * (1 - tanh(A)²)
   tanh_h._backward = _backward
   return tanh_h

def softmax(h):
   exps = np.exp(h.data - np.max(h.data, axis=-1, keepdims=True)) # Numerical stability: https://cs231n.github.io/linear-classify/ 

   if not GradMode.enabled:
      return Tensor(exps / np.sum(exps, axis=-1, keepdims=True))
   
   softmax_h = Tensor(exps / np.sum(exps, axis=-1, keepdims=True), (h,))
   def _backward():
      s = softmax_h.data
      g = softmax_h.grad
      h.grad += unbroadcast(s * (g - np.sum(g * s, axis=-1, keepdims=True)), h.data.shape) # dL/dA = dL/dS * dS/dA = S * (dL/dS - (dL/dS * S) 1_(n × n))
   softmax_h._backward = _backward
   return softmax_h

def swish(h): # Swish: https://arxiv.org/abs/1710.05941
   if not GradMode.enabled:
      return Tensor(h.data / (1 + np.exp(-h.data)))
   
   swish_h = Tensor(h.data / (1 + np.exp(-h.data)), (h,))
   def _backward():
      sigmoid_h = 1 / (1 + np.exp(-h.data))
      h.grad += unbroadcast(swish_h.grad * (swish_h.data + sigmoid_h * (1 - swish_h.data)), h.data.shape) # dL/dA = dL/dC * dC/dA = dL/dC * (f(A) + σ(A) * (1 - f(A)))
   swish_h._backward = _backward
   return swish_h

def gelu(h): # Gaussian Error Linear Unit: https://alaaalatif.github.io/2019-04-11-gelu/
   phi_h = 0.5 * (1 + erf(h.data / np.sqrt(2))) 

   if not GradMode.enabled:
      return Tensor(h.data * phi_h)

   gelu_h = Tensor(h.data * phi_h, (h,))
   def _backward():
      x = h.data
      cdf_x = 0.5 * (1 + erf(x / np.sqrt(2))) 
      pdf_x = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
      gelu_grad = cdf_x + x * pdf_x # dL/dA = dL/dC * dC/dA = dL/dC * (Φ(A) + A * φ(A))
      h.grad += unbroadcast(gelu_h.grad * gelu_grad, h.data.shape) 
   gelu_h._backward = _backward
   return gelu_h

ACTIVATIONS = {
   'linear': linear,
   'relu': relu,
   'sigmoid': sigmoid,
   'tanh': tanh,
   'softmax': softmax,
   'swish': swish,
   'gelu': gelu
}