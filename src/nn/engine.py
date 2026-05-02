import numpy as np

def unbroadcast(grad, shape):
   while len(grad.shape) > len(shape):
      grad = np.sum(grad, axis=0)
   for i, dim in enumerate(shape):
      if dim == 1:
         grad = np.sum(grad, axis=i, keepdims=True)
   return grad

class GradMode:
   enabled = True

class no_grad:
   def __enter__(self):
      self.prev = GradMode.enabled
      GradMode.enabled = False

   def __exit__(self, exc_type, exc_val, exc_tb):
      GradMode.enabled = self.prev

class Tensor:
   def __init__(self, data, _children=()):
      self.data = np.array(data, dtype=float)
      self.grad = np.zeros_like(self.data)
      self._children = _children
      self._backward = lambda: None

   def __add__(self, other):
      other = other if isinstance(other, Tensor) else Tensor(other)

      if not GradMode.enabled:
         return Tensor(self.data + other.data)
      
      sum = Tensor(self.data + other.data, (self, other))
      def _backward():
         self.grad += unbroadcast(sum.grad, self.data.shape)
         other.grad += unbroadcast(sum.grad, other.data.shape)
      sum._backward = _backward
      return sum
   
   def __radd__(self, other):
      other = other if isinstance(other, Tensor) else Tensor(other)

      if not GradMode.enabled:
         return Tensor(other.data + self.data)

      sum = Tensor(other.data + self.data, (other, self))
      def _backward():
         other.grad += unbroadcast(sum.grad, other.data.shape)
         self.grad += unbroadcast(sum.grad, self.data.shape)
      sum._backward = _backward
      return sum

   def __neg__(self):
      if not GradMode.enabled:
         return Tensor(-self.data)
      
      neg = Tensor(-self.data, (self,))
      def _backward():
         self.grad -= neg.grad
      neg._backward = _backward
      return neg

   def __sub__(self, other):
      other = other if isinstance(other, Tensor) else Tensor(other)

      if not GradMode.enabled:
         return Tensor(self.data - other.data)
      
      diff = Tensor(self.data - other.data, (self, other))
      def _backward():
         self.grad += unbroadcast(diff.grad, self.data.shape)
         other.grad -= unbroadcast(diff.grad, other.data.shape)
      diff._backward = _backward
      return diff

   def __mul__(self, other):
      other = other if isinstance(other, Tensor) else Tensor(other)

      if not GradMode.enabled:
         return Tensor(self.data * other.data)
      
      prod = Tensor(self.data * other.data, (self, other))
      def _backward():
         self.grad += unbroadcast(prod.grad * other.data, self.data.shape)
         other.grad += unbroadcast(prod.grad * self.data, other.data.shape)
      prod._backward = _backward
      return prod

   def __rmul__(self, other):
      other = other if isinstance(other, Tensor) else Tensor(other)

      if not GradMode.enabled:
         return Tensor(other.data * self.data)
      
      prod = Tensor(other.data * self.data, (other, self))
      def _backward():
         other.grad += unbroadcast(prod.grad * self.data, other.data.shape)
         self.grad += unbroadcast(prod.grad * other.data, self.data.shape)
      prod._backward = _backward
      return prod

   def __truediv__(self, other):
      other = other if isinstance(other, Tensor) else Tensor(other)

      if not GradMode.enabled:
         return Tensor(self.data / other.data)
      
      quot = Tensor(self.data / other.data, (self, other))
      def _backward():
         self.grad += unbroadcast(quot.grad / other.data, self.data.shape)
         other.grad += unbroadcast(-quot.grad * self.data / (other.data ** 2), other.data.shape)
      quot._backward = _backward
      return quot

   def __matmul__(self, other):
      other = other if isinstance(other, Tensor) else Tensor(other)

      if not GradMode.enabled:
         return Tensor(self.data @ other.data)
      
      prod = Tensor(self.data @ other.data, (self, other)) # C = A @ B
      def _backward():
         self.grad += unbroadcast(prod.grad @ other.data.T, self.data.shape) # dL/dA = dL/dC @ dC/dA = dL/dC @ B.T 
         other.grad += unbroadcast(self.data.T @ prod.grad, other.data.shape) # dL/dB = dL/dC @ dC/dB = A.T @ dL/dC 
      prod._backward = _backward
      return prod

   def __rmatmul__(self, other):
      other = other if isinstance(other, Tensor) else Tensor(other)

      if not GradMode.enabled:
         return Tensor(other.data @ self.data)
      
      prod = Tensor(other.data @ self.data, (other, self))
      def _backward():
         other.grad += unbroadcast(prod.grad @ self.data.T, other.data.shape)
         self.grad += unbroadcast(other.data.T @ prod.grad, self.data.shape)
      prod._backward = _backward
      return prod
   
   def __pow__(self, exponent):

      if not GradMode.enabled:
         return Tensor(self.data ** exponent)
      
      power = Tensor(self.data ** exponent, (self,)) # C = A ** n
      
      def _backward():
         self.grad += unbroadcast(power.grad * exponent * self.data ** (exponent - 1), self.data.shape)   # dL/dA = dL/dC * dC/dA = dL/dC * n * A ** (n - 1)
      power._backward = _backward
      return power
   
   def sum(self, axis=None, keepdims=False):

      if not GradMode.enabled:
         return Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))
      
      sum = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,))
      def _backward():
         self.grad += unbroadcast(sum.grad * np.ones_like(self.data), self.data.shape)
      sum._backward = _backward
      return sum
   
   def __abs__(self):

      if not GradMode.enabled:
         return Tensor(np.abs(self.data))
      
      abs_tensor = Tensor(np.abs(self.data), (self,))
      def _backward():
         self.grad += unbroadcast(abs_tensor.grad * np.sign(self.data), self.data.shape) # dL/dA = dL/d|A| * d|A|/dA = dL/d|A| * sign(A)
      abs_tensor._backward = _backward
      return abs_tensor
   
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
   