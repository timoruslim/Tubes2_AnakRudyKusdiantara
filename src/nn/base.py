from fastapi import params
import numpy as np 
from .engine import Tensor

def _collect(obj, params):
   if isinstance(obj, Tensor):
      params.append(obj)
   elif isinstance(obj, Module):
      params.extend(obj.parameters())
   elif isinstance(obj, (list, tuple)):
      for item in obj:
         _collect(item, params)
   elif isinstance(obj, dict):
      for item in obj.values():
         _collect(item, params)

class Module:

   def tensors(self):
      tensors = []
      for val in self.__dict__.values():
         _collect(val, tensors)
      return list({id(t): t for t in tensors}.values()) 
   
   def parameters(self):
      return [t for t in self.tensors() if t.requires_grad]
   
   def zero_grad(self):
      for p in self.parameters():
         p.grad = np.zeros_like(p.data)

   def regularization_loss(self):
      return Tensor(0.0)
   
   def freeze(self):
      for p in self.parameters():
         p.requires_grad = False
      
   def unfreeze(self):
      for p in self.parameters():
         p.requires_grad = True