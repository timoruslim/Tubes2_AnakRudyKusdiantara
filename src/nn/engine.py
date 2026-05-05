import numpy as np
import contextlib
from functools import partial

def _unbroadcast(grad, shape):
   while len(grad.shape) > len(shape):
      grad = np.sum(grad, axis=0)
   for i, dim in enumerate(shape):
      if dim == 1:
         grad = np.sum(grad, axis=i, keepdims=True)
   return grad

def _restore_shape(grad, original_shape, axis):

   kept_shape = list(original_shape)
   
   if axis is None:
      axes = range(len(kept_shape))
   elif isinstance(axis, int):
      axes = [axis]
   else:
      axes = axis
      
   for ax in axes:
      kept_shape[ax] = 1
      
   return grad.reshape(kept_shape)

class _GradMode:
   enabled = True

class no_grad(contextlib.ContextDecorator):
   def __enter__(self):
      self.prev = _GradMode.enabled
      _GradMode.enabled = False

   def __exit__(self, exc_type, exc_val, exc_tb):
      _GradMode.enabled = self.prev

class Tensor:
   def __init__(self, data, _children=()):
      self.data = np.array(data, dtype=float)
      self.grad = np.zeros_like(self.data)
      self._children = _children
      self._backward = lambda: None

   def __add__(self, other):
      other = other if isinstance(other, Tensor) else Tensor(other)

      if not _GradMode.enabled:
         return Tensor(self.data + other.data)
      
      sum = Tensor(self.data + other.data, (self, other))
      def _backward():
         self.grad += _unbroadcast(sum.grad, self.data.shape)
         other.grad += _unbroadcast(sum.grad, other.data.shape)
      sum._backward = _backward
      return sum
   
   def __radd__(self, other):
      other = other if isinstance(other, Tensor) else Tensor(other)

      if not _GradMode.enabled:
         return Tensor(other.data + self.data)

      sum = Tensor(other.data + self.data, (other, self))
      def _backward():
         other.grad += _unbroadcast(sum.grad, other.data.shape)
         self.grad += _unbroadcast(sum.grad, self.data.shape)
      sum._backward = _backward
      return sum

   def __neg__(self):
      if not _GradMode.enabled:
         return Tensor(-self.data)
      
      neg = Tensor(-self.data, (self,))
      def _backward():
         self.grad -= neg.grad
      neg._backward = _backward
      return neg

   def __sub__(self, other):
      other = other if isinstance(other, Tensor) else Tensor(other)

      if not _GradMode.enabled:
         return Tensor(self.data - other.data)
      
      diff = Tensor(self.data - other.data, (self, other))
      def _backward():
         self.grad += _unbroadcast(diff.grad, self.data.shape)
         other.grad -= _unbroadcast(diff.grad, other.data.shape)
      diff._backward = _backward
      return diff

   def __mul__(self, other):
      other = other if isinstance(other, Tensor) else Tensor(other)

      if not _GradMode.enabled:
         return Tensor(self.data * other.data)
      
      prod = Tensor(self.data * other.data, (self, other))
      def _backward():
         self.grad += _unbroadcast(prod.grad * other.data, self.data.shape)
         other.grad += _unbroadcast(prod.grad * self.data, other.data.shape)
      prod._backward = _backward
      return prod

   def __rmul__(self, other):
      other = other if isinstance(other, Tensor) else Tensor(other)

      if not _GradMode.enabled:
         return Tensor(other.data * self.data)
      
      prod = Tensor(other.data * self.data, (other, self))
      def _backward():
         other.grad += _unbroadcast(prod.grad * self.data, other.data.shape)
         self.grad += _unbroadcast(prod.grad * other.data, self.data.shape)
      prod._backward = _backward
      return prod

   def __truediv__(self, other):
      other = other if isinstance(other, Tensor) else Tensor(other)

      if not _GradMode.enabled:
         return Tensor(self.data / other.data)
      
      quot = Tensor(self.data / other.data, (self, other))
      def _backward():
         self.grad += _unbroadcast(quot.grad / other.data, self.data.shape)
         other.grad += _unbroadcast(-quot.grad * self.data / (other.data ** 2), other.data.shape)
      quot._backward = _backward
      return quot

   def __matmul__(self, other):
      other = other if isinstance(other, Tensor) else Tensor(other)

      if not _GradMode.enabled:
         return Tensor(self.data @ other.data)
      
      prod = Tensor(self.data @ other.data, (self, other)) # C = A @ B
      def _backward():
         self.grad += _unbroadcast(prod.grad @ other.data.swapaxes(-1, -2), self.data.shape) # dL/dA = dL/dC @ dC/dA = dL/dC @ B.T 
         other.grad += _unbroadcast(self.data.swapaxes(-1, -2) @ prod.grad, other.data.shape) # dL/dB = dL/dC @ dC/dB = A.T @ dL/dC 
      prod._backward = _backward
      return prod

   def __rmatmul__(self, other):
      other = other if isinstance(other, Tensor) else Tensor(other)

      if not _GradMode.enabled:
         return Tensor(other.data @ self.data)
      
      prod = Tensor(other.data @ self.data, (other, self))
      def _backward():
         other.grad += _unbroadcast(prod.grad @ self.data.swapaxes(-1, -2), other.data.shape)
         self.grad += _unbroadcast(other.data.swapaxes(-1, -2) @ prod.grad, self.data.shape)
      prod._backward = _backward
      return prod
   
   def __pow__(self, exponent):

      if not _GradMode.enabled:
         return Tensor(self.data ** exponent)
      
      power = Tensor(self.data ** exponent, (self,)) # C = A ** n
      
      def _backward():
         self.grad += _unbroadcast(power.grad * exponent * self.data ** (exponent - 1), self.data.shape)   # dL/dA = dL/dC * dC/dA = dL/dC * n * A ** (n - 1)
      power._backward = _backward
      return power
   
   def __abs__(self):

      if not _GradMode.enabled:
         return Tensor(np.abs(self.data))
      
      abs_tensor = Tensor(np.abs(self.data), (self,))
      def _backward():
         self.grad += _unbroadcast(abs_tensor.grad * np.sign(self.data), self.data.shape) # dL/dA = dL/d|A| * d|A|/dA = dL/d|A| * sign(A)
      abs_tensor._backward = _backward
      return abs_tensor
   
   def __getitem__(self, index):
      if not _GradMode.enabled:
         return Tensor(self.data[index])
      sliced = Tensor(self.data[index], (self,))
      def _backward():
         self.grad[index] += sliced.grad
      sliced._backward = _backward
      return sliced
   
   def sum(self, axis=None, keepdims=False):

      if not _GradMode.enabled:
         return Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))
      
      sum_tensor = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,))
      def _backward():
         sum_grad = _restore_shape(sum_tensor.grad, self.data.shape, axis)
         self.grad += _unbroadcast(sum_grad, self.data.shape) 
      sum_tensor._backward = _backward
      return sum_tensor
   
   def max(self, axis=None, keepdims=False):
      
      if not _GradMode.enabled:
         return Tensor(np.max(self.data, axis=axis, keepdims=keepdims))
      
      max_tensor = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), (self,))
      def _backward():
         max_grad = _restore_shape(max_tensor.grad, self.data.shape, axis)
         mask = (self.data == np.max(self.data, axis=axis, keepdims=True))
         self.grad += _unbroadcast(max_grad * mask, self.data.shape)
      max_tensor._backward = _backward
      return max_tensor
   
   def mean(self, axis=None, keepdims=False):
      
      if not _GradMode.enabled:
         return Tensor(np.mean(self.data, axis=axis, keepdims=keepdims))
      
      mean = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,))
      def _backward():
         if axis is None:
            N = self.data.size
         elif isinstance(axis, int):
            N = self.data.shape[axis]
         else:
            N = np.prod([self.data.shape[ax] for ax in axis]) 
         mean_grad = _restore_shape(mean.grad, self.data.shape, axis)
         self.grad += _unbroadcast(mean_grad / N, self.data.shape)
      mean._backward = _backward
      return mean
   
   def reshape(self, *shape):
      if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
         shape = shape[0]
      
      if not _GradMode.enabled:
         return Tensor(np.reshape(self.data, shape))
      
      reshaped = Tensor(np.reshape(self.data, shape), (self,))
      def _backward():
         self.grad += _unbroadcast(reshaped.grad.reshape(self.data.shape), self.data.shape)
      reshaped._backward = _backward
      return reshaped
   
   def split(self, indices_or_sections, axis=0):

      if not _GradMode.enabled:
         return [Tensor(part) for part in np.split(self.data, indices_or_sections, axis)]
      
      split_tensors = [Tensor(part, (self,)) for part in np.split(self.data, indices_or_sections, axis)]
      
      if isinstance(indices_or_sections, int):
         sections = self.data.shape[axis] // indices_or_sections
         indices = [sections * i for i in range(1, indices_or_sections)]
      else:
         indices = list(indices_or_sections)
      
      start = 0
      for i, split_tensor in enumerate(split_tensors):

         end = indices[i] if i < len(indices) else self.data.shape[axis]
         
         def _backward(child=split_tensor, start=start, end=end):
            parent_view = self.grad.swapaxes(0, axis)
            child_view = child.grad.swapaxes(0, axis)
            parent_view[start:end] += child_view
         
         split_tensor._backward = _backward
         start = end
            
      return split_tensors
   
   def transpose(self, axes=None):

      if not _GradMode.enabled:
         return Tensor(np.transpose(self.data, axes=axes))
      
      transposed = Tensor(np.transpose(self.data, axes=axes), (self,))
      def _backward():
         inverse_axes = np.argsort(axes) if axes is not None else None
         self.grad += _unbroadcast(np.transpose(transposed.grad, axes=inverse_axes), self.data.shape)
      transposed._backward = _backward
      return transposed
   
   def pad(self, pad_width, mode='constant', constant_values=0):
      if not _GradMode.enabled:
         return Tensor(np.pad(self.data, pad_width, mode=mode, constant_values=constant_values)) # type: ignore

      padded = Tensor(np.pad(self.data, pad_width, mode=mode, constant_values=constant_values), (self,)) # type: ignore
      def _backward():
         slices = []
         for (pad_before, pad_after), dim in zip(pad_width, padded.grad.shape):
            slices.append(slice(pad_before, dim - pad_after))
         self.grad += _unbroadcast(padded.grad[tuple(slices)], self.data.shape)
      padded._backward = _backward
      return padded
   
   @staticmethod
   def concatenate(tensors, axis=0):

      if not _GradMode.enabled:
         return Tensor(np.concatenate([t.data for t in tensors], axis))
      
      concat = Tensor(np.concatenate([t.data for t in tensors], axis), tuple(tensors))
      def _backward():
         indices = np.cumsum([t.data.shape[axis] for t in tensors])
         split_grads = np.split(concat.grad, indices[:-1], axis=axis)
         for t, g in zip(tensors, split_grads):
            t.grad += _unbroadcast(g, t.data.shape)
      concat._backward = _backward
      return concat
   
   def backward(self):
      topo = []
      visited = set()
      def build_topo(tensor): # topological sort 
         if tensor not in visited:
            visited.add(tensor)
            for child in tensor._children:
               build_topo(child)
            topo.append(tensor)
      build_topo(self)
      self.grad = np.ones_like(self.data) # dL/dL = 1
      for tensor in reversed(topo):
         tensor._backward()

   def __repr__(self):
      return f"Tensor({self.data})"
   