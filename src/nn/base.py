import numpy as np 
from .engine import Tensor

class Module:

   def parameters(self):
      params = []
      for all in self.__dict__.values():
         if isinstance(all, Tensor):
            params.append(all)
         if isinstance(all, Module):
            params.extend(all.parameters())
         if isinstance(all, (list, tuple)):
            for item in all:
               if isinstance(item, Tensor):
                  params.append(item)
               if isinstance(item, Module):
                  params.extend(item.parameters())
      return params
   
   def zero_grad(self):
      for p in self.parameters():
         p.grad = np.zeros_like(p.data)