from .engine import Tensor, _unbroadcast, _GradMode
import numpy as np

def mse(pred, target):
   pred = pred if isinstance(pred, Tensor) else Tensor(pred.astype(np.float32))
   target = target if isinstance(target, Tensor) else Tensor(target.astype(np.float32))

   if not _GradMode.enabled:
      return Tensor(np.mean((pred.data - target.data) ** 2))

   mse_loss = Tensor(np.mean((pred.data - target.data) ** 2), (pred, target))
   def _backward():
      pred.grad += _unbroadcast(2 * (pred.data - target.data) / pred.data.size * mse_loss.grad, pred.data.shape) # dL/dP = dL/dMSE * dMSE/dP = 2 * (P - T) / N
      target.grad -= _unbroadcast(2 * (pred.data - target.data) / target.data.size * mse_loss.grad, target.data.shape) # dL/dT = dL/dMSE * dMSE/dT = -2 * (P - T) / N
   mse_loss._backward = _backward
   return mse_loss

def bce(pred, target):
   pred = pred if isinstance(pred, Tensor) else Tensor(pred.astype(np.float32))
   target = target if isinstance(target, Tensor) else Tensor(target.astype(np.float32))
   p_safe = np.clip(pred.data, 1e-15, 1.0 - 1e-15)

   if not _GradMode.enabled:
      return Tensor(-np.mean(target.data * np.log(p_safe) + (1 - target.data) * np.log(1 - p_safe)))
   
   bce_loss = Tensor(-np.mean(target.data * np.log(p_safe) + (1 - target.data) * np.log(1 - p_safe)), (pred, target))
   def _backward():
      pred.grad -= _unbroadcast(((target.data / p_safe) - ((1 - target.data) / (1 - p_safe))) / pred.data.size * bce_loss.grad, pred.data.shape) # dL/dP = dL/dBCE * dBCE/dP = - (T / P - (1 - T) / (1 - P)) / N
      target.grad -= _unbroadcast((np.log(p_safe) - np.log(1 - p_safe)) / target.data.size * bce_loss.grad, target.data.shape) # dL/dT = dL/dBCE * dBCE/dT = -(log(P) - log(1 - P)) / N
   bce_loss._backward = _backward
   return bce_loss

def cce(pred, target):
   pred = pred if isinstance(pred, Tensor) else Tensor(pred.astype(np.float32))
   target = target if isinstance(target, Tensor) else Tensor(target.astype(np.float32))
   p_safe = np.clip(pred.data, 1e-15, 1.0)

   if not _GradMode.enabled:
      return Tensor(-np.mean(np.sum(target.data * np.log(p_safe), axis=-1)))

   cce_loss = Tensor(-np.mean(np.sum(target.data * np.log(p_safe), axis=-1)), (pred, target))
   def _backward():
      N = pred.data.size / pred.data.shape[-1]
      pred.grad -= _unbroadcast((target.data / p_safe) / N * cce_loss.grad, pred.data.shape) # dL/dP = dL/dCCE * dCCE/dP = -T / (P * N) 
      target.grad -= _unbroadcast((np.log(p_safe)) / N * cce_loss.grad, target.data.shape) # dL/dT = dL/dCCE * dCCE/dT = -log(P) / N
   cce_loss._backward = _backward
   return cce_loss

def scce(pred, target, ignore_index=None):
   pred = pred if isinstance(pred, Tensor) else Tensor(pred.astype(np.float32))
   target = target if isinstance(target, Tensor) else Tensor(target.astype(np.int32))

   pred_safe = np.clip(pred.data, 1e-15, 1.0)

   mask = target.data != ignore_index if ignore_index is not None else np.ones_like(target.data, dtype=bool) # mask out ignored tokens
   N_valid = max(np.sum(mask), 1) # number of non-ignored tokens 
   
   pred_correct = np.take_along_axis( # select predicted probability of correct class
      pred_safe, # (B, T, C)
      target.data[..., None].astype(int), # convert (B, T) to (B, T, 1) 
      axis=-1 
   ).squeeze(-1) # (B, T)

   token_losses = -np.log(pred_correct) * mask # select non-ignored tokens 
   loss_val = np.sum(token_losses) / N_valid 

   if not _GradMode.enabled:
      return Tensor(loss_val)

   scce_loss = Tensor(loss_val, (pred, target))

   def _backward():
      grad = np.zeros_like(pred_safe) # dL/dP_incorrect = 0 
      update = (-1.0 / (pred_correct[..., None] * N_valid)) * mask[..., None] # dL/dP_correct = -1 / (P_correct * N_valid) 
      np.put_along_axis(
         grad, 
         target.data[..., None].astype(int), 
         update, 
         axis=-1
      ) # combine into full gradient matrix
      pred.grad += _unbroadcast(grad * scce_loss.grad, pred.data.shape) # dL/dP = dL/dSCCE * dSCCE/dP = grad 
      
   scce_loss._backward = _backward

   return scce_loss

LOSSES = {
   'mse': mse,
   'bce': bce,
   'cce': cce, 
   'scce': scce
}