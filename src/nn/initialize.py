from .engine import Tensor
import numpy as np

def zero(rows, cols, params=None, rng=None):
   return Tensor(np.zeros((rows, cols)))

def uniform(rows, cols, params=None, rng=None):
   rng = rng or np.random.default_rng()
   params = params or {}
   low = params.get('low', -0.1)
   high = params.get('high', 0.1)
   return Tensor(rng.uniform(low, high, (rows, cols)))

def normal(rows, cols, params=None, rng=None):
   rng = rng or np.random.default_rng()
   params = params or {}
   mean = params.get('mean', 0)
   var = params.get('var', 0.01)
   if var <= 0:
      raise ValueError("Variance must be positive for normal initialization")
   return Tensor(rng.normal(mean, var ** 0.5, (rows, cols)))

def xavier(rows, cols, params=None, rng=None): # source: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
   rng = rng or np.random.default_rng()
   limit = np.sqrt(6 / (rows + cols))
   return Tensor(rng.uniform(-limit, limit, (rows, cols)))

def he(rows, cols, params=None, rng=None): # source: https://arxiv.org/abs/1502.01852
   rng = rng or np.random.default_rng()
   stddev = np.sqrt(2 / rows)
   return Tensor(rng.normal(0, stddev, (rows, cols)))

INITIALIZATIONS = {
   'zero': zero,
   'uniform': uniform,
   'normal': normal,
   'xavier': xavier,
   'he': he
}