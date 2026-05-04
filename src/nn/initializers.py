from .engine import Tensor
import numpy as np

def _compute_connections(shape):
   if len(shape) < 2:
      n_in = n_out = shape[0]
   elif len(shape) == 2:
      n_in = shape[0]
      n_out = shape[1]
   else:
      receptive_field = np.prod(shape[:-2])
      n_in = shape[-2] * receptive_field
      n_out = shape[-1] * receptive_field
      
   return n_in, n_out

def zero(shape, rng=None):
   return Tensor(np.zeros(shape))

def uniform(shape, low=None, high=None, rng=None):
   rng = rng or np.random.default_rng()
   low = low if low is not None else -0.1
   high = high if high is not None else 0.1
   return Tensor(rng.uniform(low, high, shape))

def normal(shape, mean=None, var=None, rng=None):
   rng = rng or np.random.default_rng()
   mean = mean if mean is not None else 0
   var = var if var is not None else 0.01
   if var <= 0:
      raise ValueError("Variance must be positive for normal initialization")
   return Tensor(rng.normal(mean, var ** 0.5, shape))

def xavier(shape, rng=None): # source: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
   rng = rng or np.random.default_rng()
   limit = np.sqrt(6 / (sum(_compute_connections(shape))))
   return Tensor(rng.uniform(-limit, limit, shape))

def he(shape, rng=None): # source: https://arxiv.org/abs/1502.01852
   rng = rng or np.random.default_rng()
   n_in, _ = _compute_connections(shape)
   stddev = np.sqrt(2 / n_in)
   return Tensor(rng.normal(0, stddev, shape))

INITIALIZATIONS = {
   'zero': zero,
   'uniform': uniform,
   'normal': normal,
   'xavier': xavier,
   'he': he
}