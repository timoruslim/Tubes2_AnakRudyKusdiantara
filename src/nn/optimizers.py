import numpy as np

class Optimizer:

   def __init__(self, parameters, learning_rate=0.01, clip_value=None, clip_norm=None, **kwargs):
      self.parameters = parameters
      self.alpha = learning_rate
      self.clip_value = clip_value
      self.clip_norm = clip_norm
      self.params = kwargs

   def step(self):
      raise NotImplementedError("Optimizer step method must be implemented by subclasses")

   def zero_grad(self):
      for p in self.parameters:
         p.grad = np.zeros_like(p.data)

   def _clip_gradients(self):
      if self.clip_value is not None:
         for p in self.parameters:
            p.grad = np.clip(p.grad, -self.clip_value, self.clip_value)
      if self.clip_norm is not None:
         global_norm = np.sqrt(sum(np.sum(p.grad ** 2) for p in self.parameters))
         if global_norm > self.clip_norm:
            scale = self.clip_norm / (global_norm + 1e-6)
            for p in self.parameters:
               p.grad *= scale

class SGD(Optimizer):

   def step(self):
      self._clip_gradients()
      for p in self.parameters:
         p.data -= self.alpha * p.grad

class Adam(Optimizer):

   def __init__(self, parameters, learning_rate=0.001, clip_value=None, clip_norm=None, **kwargs):
      super().__init__(parameters, learning_rate, clip_value, clip_norm, **kwargs)
      self.beta1 = self.params.get('beta1', 0.9)                           # defaul follows : https://arxiv.org/pdf/1412.6980
      self.beta2 = self.params.get('beta2', 0.999)
      self.epsilon = self.params.get('epsilon', 1e-8) 
      self.m = [np.zeros_like(p.data) for p in self.parameters]
      self.v = [np.zeros_like(p.data) for p in self.parameters]
      self.t = 0
   
   def step(self):
      self._clip_gradients()
      self.t += 1
      for i, p in enumerate(self.parameters):
         g = p.grad                                                        # get gradient
         self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g         # update biased first moment estimate
         self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)  # update biased second moment estimate
         m_hat = self.m[i] / (1 - self.beta1 ** self.t)                    # compute bias-corrected first moment estimate
         v_hat = self.v[i] / (1 - self.beta2 ** self.t)                    # compute bias-corrected second moment estimate
         p.data -= self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)    # update parameter 

OPTIMIZER = {
   'sgd': SGD,
   'adam': Adam
}