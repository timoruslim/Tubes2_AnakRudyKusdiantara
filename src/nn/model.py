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
      self.layers = list(layers) if layers is not None else []
      self.seed = seed
      self.rng = np.random.default_rng(seed)

   def __call__(self, *args, **kwargs):
      return self.forward(*args, **kwargs)
   
   def forward(self, inputs):
      h = inputs
      for layer in self.layers:
         h = layer(h)
      return h
   
   def regularization_loss(self):
      reg_loss = Tensor(0.0)
      for layer in self.layers:
         reg_loss += layer.regularization_loss()
      return reg_loss
   
   def add(self, layer):
      if not isinstance(layer, Layer): 
         raise TypeError(f"model.add() expects a Layer instance, got {type(layer).__name__}")
      self.layers.append(layer)
   
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
      
      with no_grad():
         outputs = []
         for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_output = self.forward(batch_X)
            outputs.append(batch_output.data)
         return np.concatenate(outputs, axis=0)
      
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
         reg_loss = self.regularization_loss().data
         return pred_loss + reg_loss
      
   def fit(self, X, y, epochs=1, batch_size=32, validation_data=None):

      if not self.optimizer.parameters:
         # self.build() 
         self.optimizer.build(self.parameters())

      X = np.array(X)
      y = np.array(y)
      if len(X) == 0:
         raise ValueError("Dataset cannot be empty.")
      if len(X) != len(y): 
         raise ValueError(f"Input X (length {len(X)}) and target y (length {len(y)}) must have the same number of samples.")

      history = {'train_loss': [], 'val_loss': []}
      num_batches = (len(X) + batch_size - 1) // batch_size

      for epoch in range(epochs):

         indices = self.rng.permutation(len(X))
         X = X[indices]
         y = y[indices]

         epoch_loss = 0.0
         for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            # Forward pass
            outputs = self.forward(batch_X)
            loss = self.loss_fn(outputs, batch_y) + self.regularization_loss()
            epoch_loss += loss.data

            # Backward pass 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
         
         avg_loss = epoch_loss / num_batches
         history['train_loss'].append(avg_loss)
         print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
         
         if validation_data is not None:
            X_val, y_val = validation_data
            val_loss = self.evaluate(X_val, y_val, batch_size)
            print(f", Validation Loss: {val_loss:.4f}")
            history['val_loss'].append(val_loss)

      return history
   
   def summary(self):
      print("\n" + "=" * 67)
      print(f"{'Layer':<17} {'Output Shape':<17} {'Param #':<16}")
      print("=" * 67)
      
      total_params = 0
      layer_count = 0
      
      for layer in self.layers:
         
         # Get name
         # Get shape
         # Get parameters
         pass 
      
      print("=" * 67)
      print(f"Total parameters: {total_params:,}")
      print("=" * 67 + "\n")