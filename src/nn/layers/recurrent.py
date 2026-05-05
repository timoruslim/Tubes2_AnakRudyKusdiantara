from .layer import Layer
from ..engine import Tensor
from ..initializers import INITIALIZATIONS
from ..activations import ACTIVATIONS
import numpy as np

class SimpleRNNCell(Layer):
   def __init__(self, input_size, hidden_size, activation="tanh", weight_init='xavier', bias_init='zero', l1_lambda=0.0, l2_lambda=0.0, seed=None):
      super().__init__()
      
      self.input_size = input_size
      self.hidden_size = hidden_size
      self.activation = ACTIVATIONS[activation]

      self.l1_lambda = l1_lambda
      self.l2_lambda = l2_lambda

      rng = np.random.default_rng(seed)
      weight_initializer = INITIALIZATIONS[weight_init]
      bias_initializer = INITIALIZATIONS[bias_init]
      
      self.W_ih = weight_initializer((input_size, hidden_size), rng=rng) 
      self.W_hh = weight_initializer((hidden_size, hidden_size), rng=rng) 
      self.bias = bias_initializer((1, hidden_size), rng=rng) # single bias 

   def forward(self, x_t, h_prev):
      return self.activation(x_t @ self.W_ih + h_prev @ self.W_hh + self.bias)
   
   def regularization_loss(self):
      reg_loss = Tensor(0.0) 
      if self.l1_lambda > 0:
         reg_loss += self.l1_lambda * (abs(self.W_ih).sum() + abs(self.W_hh).sum())
      if self.l2_lambda > 0:
         reg_loss += self.l2_lambda * ((self.W_ih ** 2).sum() + (self.W_hh ** 2).sum())
      return reg_loss
   
class LSTMCell(Layer):
   def __init__(self, input_size, hidden_size, activation="tanh", recurrent_activation="sigmoid", weight_init='xavier', bias_init='zero', l1_lambda=0.0, l2_lambda=0.0, seed=None):
      super().__init__()

      self.input_size = input_size
      self.hidden_size = hidden_size

      self.activation = ACTIVATIONS[activation]
      self.recurrent_activation = ACTIVATIONS[recurrent_activation]

      self.l1_lambda = l1_lambda
      self.l2_lambda = l2_lambda

      rng = np.random.default_rng(seed)
      weight_initializer = INITIALIZATIONS[weight_init]
      bias_initializer = INITIALIZATIONS[bias_init]
      
      self.W_ih = weight_initializer((input_size, 4 * hidden_size), rng=rng) 
      self.W_hh = weight_initializer((hidden_size, 4 * hidden_size), rng=rng) 
      self.bias = bias_initializer((1, 4 * hidden_size), rng=rng) 
      self.bias.data[0, hidden_size:2*hidden_size] = 1.0 # forget gate bias to 1 (unit_forget_bias in Keras)

   def forward(self, x_t, h_prev, c_prev):
      gates = x_t @ self.W_ih + h_prev @ self.W_hh + self.bias
      i_gate, f_gate, o_gate, C_gate = gates.split(4, axis=1)
      
      i_gate = self.recurrent_activation(i_gate)
      f_gate = self.recurrent_activation(f_gate)
      o_gate = self.recurrent_activation(o_gate)
      C_gate = self.activation(C_gate)

      c_t = f_gate * c_prev + i_gate * C_gate
      h_t = o_gate * self.activation(c_t)

      return h_t, c_t
   
   def regularization_loss(self):
      reg_loss = Tensor(0.0) 
      if self.l1_lambda > 0:
         reg_loss += self.l1_lambda * (abs(self.W_ih).sum() + abs(self.W_hh).sum())
      if self.l2_lambda > 0:
         reg_loss += self.l2_lambda * ((self.W_ih ** 2).sum() + (self.W_hh ** 2).sum())
      return reg_loss
   
class SimpleRNN(Layer):
   def __init__(self, input_size, hidden_size, return_sequences=False, return_state=False, cell_kwargs=None):
      super().__init__()
      cell_kwargs = {} if cell_kwargs is None else cell_kwargs
      self.cell = SimpleRNNCell(input_size, hidden_size, **cell_kwargs) 
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
   
   def regularization_loss(self):
      return self.cell.regularization_loss()
   
class LSTM(Layer):
   def __init__(self, input_size, hidden_size, return_sequences=False, return_state=False, cell_kwargs=None):
      super().__init__()
      cell_kwargs = {} if cell_kwargs is None else cell_kwargs
      self.cell = LSTMCell(input_size, hidden_size, **cell_kwargs) 
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
   
   def regularization_loss(self):
      return self.cell.regularization_loss()