from .layer import Layer
from ..engine import Tensor
from ..initializers import INITIALIZATIONS
from ..activations import ACTIVATIONS
import numpy as np

class SimpleRNNCell(Layer):
   def __init__(self, hidden_size, activation="tanh", weight_init='xavier', bias_init='zero', l1_lambda=0.0, l2_lambda=0.0, seed=None):
      super().__init__()

      if activation not in ACTIVATIONS:
         raise ValueError(f"Unknown activation '{activation}'. Available: {list(ACTIVATIONS.keys())}")
      if weight_init not in INITIALIZATIONS:
         raise ValueError(f"Unknown weight_init '{weight_init}'. Available: {list(INITIALIZATIONS.keys())}")
      if bias_init not in INITIALIZATIONS:
         raise ValueError(f"Unknown bias_init '{bias_init}'. Available: {list(INITIALIZATIONS.keys())}")
      if hidden_size <= 0:
         raise ValueError(f"hidden_size must be strictly positive, got {hidden_size}.")
      
      self.hidden_size = hidden_size
      self.activation = ACTIVATIONS[activation]

      self.l1_lambda = l1_lambda
      self.l2_lambda = l2_lambda

      self.rng = np.random.default_rng(seed)
      self.weight_initializer = INITIALIZATIONS[weight_init]
      self.bias_initializer = INITIALIZATIONS[bias_init]

   def forward(self, x_t, h_prev):
      return self.activation(x_t @ self.W_ih + h_prev @ self.W_hh + self.bias)
   
   def build(self, input_shape):
      super().build(input_shape)
      self.input_size = input_shape[-1]
      self.W_ih = self.weight_initializer((self.input_size, self.hidden_size), rng=self.rng) 
      self.W_hh = self.weight_initializer((self.hidden_size, self.hidden_size), rng=self.rng) 
      self.bias = self.bias_initializer((1, self.hidden_size), rng=self.rng) # single bias 

   
   def regularization_loss(self):
      reg_loss = Tensor(0.0) 
      if self.l1_lambda > 0:
         reg_loss += self.l1_lambda * (abs(self.W_ih).sum() + abs(self.W_hh).sum())
      if self.l2_lambda > 0:
         reg_loss += self.l2_lambda * ((self.W_ih ** 2).sum() + (self.W_hh ** 2).sum())
      return reg_loss
   
class LSTMCell(Layer):
   def __init__(self, hidden_size, activation="tanh", recurrent_activation="sigmoid", weight_init='xavier', bias_init='zero', l1_lambda=0.0, l2_lambda=0.0, seed=None):
      super().__init__()

      if activation not in ACTIVATIONS:
         raise ValueError(f"Unknown activation '{activation}'. Available: {list(ACTIVATIONS.keys())}")
      if recurrent_activation not in ACTIVATIONS:
         raise ValueError(f"Unknown recurrent_activation '{recurrent_activation}'. Available: {list(ACTIVATIONS.keys())}")
      if weight_init not in INITIALIZATIONS:
         raise ValueError(f"Unknown weight_init '{weight_init}'. Available: {list(INITIALIZATIONS.keys())}")
      if bias_init not in INITIALIZATIONS:
         raise ValueError(f"Unknown bias_init '{bias_init}'. Available: {list(INITIALIZATIONS.keys())}")
      if hidden_size <= 0:
         raise ValueError(f"hidden_size must be strictly positive, got {hidden_size}.")

      self.hidden_size = hidden_size

      self.activation = ACTIVATIONS[activation]
      self.recurrent_activation = ACTIVATIONS[recurrent_activation]

      self.l1_lambda = l1_lambda
      self.l2_lambda = l2_lambda

      self.rng = np.random.default_rng(seed)
      self.weight_initializer = INITIALIZATIONS[weight_init]
      self.bias_initializer = INITIALIZATIONS[bias_init]

   def forward(self, x_t, h_prev, c_prev):
      gates = x_t @ self.W_ih + h_prev @ self.W_hh + self.bias
      i_gate, f_gate, C_gate, o_gate = gates.split(4, axis=1)
      
      i_gate = self.recurrent_activation(i_gate)
      f_gate = self.recurrent_activation(f_gate)
      C_gate = self.activation(C_gate)
      o_gate = self.recurrent_activation(o_gate)

      c_t = f_gate * c_prev + i_gate * C_gate
      h_t = o_gate * self.activation(c_t)

      return h_t, c_t
   
   def build(self, input_shape):
      super().build(input_shape)
      self.input_size = input_shape[-1]
      
      self.W_ih = self.weight_initializer((self.input_size, 4 * self.hidden_size), rng=self.rng) 
      self.W_hh = self.weight_initializer((self.hidden_size, 4 * self.hidden_size), rng=self.rng) 
      self.bias = self.bias_initializer((1, 4 * self.hidden_size), rng=self.rng) 
      self.bias.data[0, self.hidden_size:2*self.hidden_size] = 1.0 # forget gate bias to 1 (unit_forget_bias in Keras)
   
   def regularization_loss(self):
      reg_loss = Tensor(0.0) 
      if self.l1_lambda > 0:
         reg_loss += self.l1_lambda * (abs(self.W_ih).sum() + abs(self.W_hh).sum())
      if self.l2_lambda > 0:
         reg_loss += self.l2_lambda * ((self.W_ih ** 2).sum() + (self.W_hh ** 2).sum())
      return reg_loss
   
class SimpleRNN(Layer):
   def __init__(self, hidden_size, return_sequences=False, return_state=False, cell_kwargs=None):
      super().__init__()
      if hidden_size <= 0:
         raise ValueError(f"hidden_size must be strictly positive, got {hidden_size}.")
      
      cell_kwargs = {} if cell_kwargs is None else cell_kwargs
      self.cell = SimpleRNNCell(hidden_size, **cell_kwargs) 
      self.return_sequences = return_sequences
      self.return_state = return_state

   def forward(self, inputs, initial_state=None):
      batch_size, seq_len, _ = inputs.data.shape
      h_t = Tensor(np.zeros((batch_size, self.cell.hidden_size))) if initial_state is None else initial_state

      outputs = []
      for t in range(seq_len):
         x_t = inputs[:, t:t+1, :].reshape(batch_size, -1)
         h_t = self.cell(x_t, h_t)
         outputs.append(h_t.reshape(batch_size, 1, self.cell.hidden_size))
      
      seq_outputs = Tensor.concatenate(outputs, axis=1)
      output = seq_outputs if self.return_sequences else h_t

      return (output, h_t) if self.return_state else output
   
   def build(self, input_shape):
      super().build(input_shape)
      self.cell.build(input_shape)
      self.output_shape = (input_shape[0], input_shape[1], self.cell.hidden_size) if self.return_sequences else (input_shape[0], self.cell.hidden_size)
   
   def regularization_loss(self):
      return self.cell.regularization_loss()
   
class LSTM(Layer):
   def __init__(self, hidden_size, return_sequences=False, return_state=False, cell_kwargs=None):
      super().__init__()
      if hidden_size <= 0:
         raise ValueError(f"hidden_size must be strictly positive, got {hidden_size}.")
      
      cell_kwargs = {} if cell_kwargs is None else cell_kwargs
      self.cell = LSTMCell(hidden_size, **cell_kwargs) 
      self.return_sequences = return_sequences
      self.return_state = return_state

   def forward(self, inputs, initial_state=None):
      batch_size, seq_len, _ = inputs.data.shape
      h_t = Tensor(np.zeros((batch_size, self.cell.hidden_size))) if initial_state is None else initial_state[0]
      c_t = Tensor(np.zeros((batch_size, self.cell.hidden_size))) if initial_state is None else initial_state[1]

      outputs = []
      for t in range(seq_len):
         x_t = inputs[:, t:t+1, :].reshape(batch_size, -1)
         h_t, c_t = self.cell(x_t, h_t, c_t)
         outputs.append(h_t.reshape(batch_size, 1, self.cell.hidden_size))
      
      seq_outputs = Tensor.concatenate(outputs, axis=1)
      output = seq_outputs if self.return_sequences else h_t

      return (output, (h_t, c_t)) if self.return_state else output
   
   def build(self, input_shape):
      super().build(input_shape)
      self.cell.build(input_shape)
      self.output_shape = (input_shape[0], input_shape[1], self.cell.hidden_size) if self.return_sequences else (input_shape[0], self.cell.hidden_size)

   def regularization_loss(self):
      return self.cell.regularization_loss()