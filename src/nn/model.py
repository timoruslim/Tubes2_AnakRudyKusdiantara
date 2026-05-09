from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
   from tqdm import tqdm

from .layers.layer import Layer
from .engine import Tensor, no_grad
from .base import Module
from .initializers import INITIALIZATIONS
from .activations import ACTIVATIONS
from .losses import LOSSES
from .optimizers import OPTIMIZER, Adam, SGD, Optimizer
import numpy as np

class Model(Module):

   def __init__(self, layers=None, seed=None):
      super().__init__()
      
      self.seed = seed
      self.rng = np.random.default_rng(seed)

      self.layers = []
      if layers is not None:
         for layer in layers:
               self.add(layer) # use add to ensure proper checks 

      if self.layers and self.layers[0].__class__.__name__ == 'Input':
         self.build(self.layers[0].shape)

   def __call__(self, *args, **kwargs):
      return self.forward(*args, **kwargs)
   
   def forward(self, inputs):
      h = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
      for layer in self.layers:
         h = layer(h)
      return h
   
   def add(self, layer):
      if not isinstance(layer, Layer): 
         raise TypeError(f"model.add() expects a Layer instance, got {type(layer).__name__}")
      self.layers.append(layer)

   def build(self, input_shape):
      current_shape = input_shape
      for layer in self.layers:
         layer.build(current_shape)
         current_shape = layer.output_shape
      if hasattr(self, 'optimizer'):
         self.optimizer.build(self.parameters())
      self.built = True

   def compile(self, optimizer, loss, learning_rate=0.001, **kwargs):

      if not self.layers:
         raise RuntimeError("Cannot compile a model with no layers. Use model.add(layer) first.")

      if loss not in LOSSES: raise ValueError(f"Unknown loss: '{loss}'. Available losses: {list(LOSSES.keys())}")
      self.loss_fn = LOSSES[loss]
      
      if isinstance(optimizer, str):
         self.optimizer = OPTIMIZER[optimizer](learning_rate=learning_rate, **kwargs)
      elif isinstance(optimizer, Optimizer):
         self.optimizer = optimizer
      else:
         raise ValueError(f"Unknown optimizer: '{optimizer}'. Available optimizers: {list(OPTIMIZER.keys())}")
      
      self.optimizer.build(self.parameters())

   def predict(self, X, batch_size=32):

      X = np.array(X)
      if len(X) == 0:
         raise ValueError("Dataset cannot be empty.")
      
      if not hasattr(self, 'built') or not self.built:
         self.build(X.shape)
      
      with no_grad():
         outputs = []
         for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_output = self.forward(batch_X)
            outputs.append(batch_output.data)
         return np.concatenate(outputs, axis=0)
      
   def _regularization_loss(self):
      reg_loss = Tensor(0.0)
      for layer in self.layers:
         reg_loss += layer.regularization_loss()
      return reg_loss
      
   def evaluate(self, X, y, batch_size=32):

      X = np.array(X)
      y = np.array(y)

      if len(X) == 0:
         raise ValueError("Dataset cannot be empty.")
      if len(X) != len(y): 
         raise ValueError(f"Input X (length {len(X)}) and target y (length {len(y)}) must have the same number of samples.")
      
      with no_grad():
         y_pred = self.predict(X, batch_size)
         pred_loss = self.loss_fn(y_pred, y).data
         reg_loss = self._regularization_loss().data
         return pred_loss + reg_loss
      
   def fit(self, X, y, epochs=1, batch_size=32, verbose=2, validation_data=None):
      
      X = np.array(X)
      y = np.array(y)
      if len(X) == 0:
         raise ValueError("Dataset cannot be empty.")
      if len(X) != len(y): 
         raise ValueError(f"Input X (length {len(X)}) and target y (length {len(y)}) must have the same number of samples.")

      if not hasattr(self, 'built') or not self.built:
         self.build(X.shape)
      if not self.optimizer.parameters:
         self.optimizer.build(self.parameters())

      if verbose == 2:
         try:
            from tqdm import tqdm
         except ImportError:
            print("tqdm is not installed. Install it to see progress bars: pip install tqdm")
            verbose = 1

      history = {'train_loss': [], 'val_loss': []}
      num_batches = (len(X) + batch_size - 1) // batch_size

      for epoch in range(epochs):
         if verbose:
            print(f"Epoch {epoch+1}/{epochs}")

         indices = self.rng.permutation(len(X))
         X = X[indices]
         y = y[indices]

         epoch_loss = 0.0

         batch_iterator = range(0, len(X), batch_size)
         pbar = tqdm(batch_iterator, desc="  Running", leave=False, bar_format='{l_bar}{bar:30}{r_bar}') if verbose == 2 else batch_iterator  # type: ignore
         
         for step, i in enumerate(pbar, 1):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            # Forward pass
            outputs = self.forward(batch_X)
            loss = self.loss_fn(outputs, batch_y) + self._regularization_loss()
            epoch_loss += loss.data

            # Backward pass 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if hasattr(pbar, 'set_postfix'):
               running_loss = epoch_loss / step
               getattr(pbar, 'set_postfix')({'loss': f"{running_loss:.4f}"})
         
         avg_loss = epoch_loss / num_batches
         history['train_loss'].append(avg_loss)
         
         log_msg = f"  - loss: {avg_loss:.4f}"
         
         if validation_data is not None:
            X_val, y_val = validation_data
            val_loss = self.evaluate(X_val, y_val, batch_size)

            history['val_loss'].append(val_loss)
            log_msg += f" - val_loss: {val_loss:.4f}"

         print(log_msg) if verbose else None

      return history
   
   def summary(self):
      if not hasattr(self, 'built') or not self.built: 
         raise RuntimeError("You must add Input layer or call build() before calling summary().")

      print("\n" + "=" * 67)
      print(f"{'Layer':<22} {'Output Shape':<23} {'Param #':<22}")
      print("=" * 67)
      
      total_params = 0
      trainable_params = 0
      
      for layer in self.layers:
         
         name = layer.__class__.__name__ + f"({layer.activation.__name__ if hasattr(layer, 'activation') and layer.activation else ''})"
         shape = str(getattr(layer, 'output_shape', 'Unknown'))
         layer_total = sum(p.data.size for p in layer.tensors())
         layer_trainable = sum(p.data.size for p in layer.parameters())

         print(f"{name:<22} {shape:<23} {layer_total:<22,}")
         total_params += layer_total
         trainable_params += layer_trainable

      print("=" * 67)
      print(f"Total parameters: {total_params:,}")
      print(f"Trainable parameters: {trainable_params:,}")
      print(f"Non-trainable params: {total_params - trainable_params:,}")
      print("=" * 67 + "\n")

   def save(self, filepath):
      if not hasattr(self, 'built') or not self.built: 
         raise RuntimeError("You must add Input layer or call build() before calling save().")
      
      params = {f"param_{i}": p.data for i, p in enumerate(self.tensors())}
      np.savez(filepath, **params)

   def load(self, filepath):
      if not hasattr(self, 'built') or not self.built: 
         raise RuntimeError("You must add Input layer or call build() before calling load().")
      
      data = np.load(filepath)
      params = self.tensors()
      if len(data.files) != len(params):
         raise ValueError(f"Number of parameters in file ({len(data.files)}) does not match model parameters ({len(params)}).")
      for i, p in enumerate(params):
         saved_data = data[f"param_{i}"]
         if p.data.shape != saved_data.shape:
            raise ValueError(f"Shape mismatch for param_{i}: model expects {p.data.shape}, but file contains {saved_data.shape}.")
         p.data = saved_data