import re 
import json
import numpy as np

class TextTokenizer:

   def __init__(self):
      self.token_to_id = {
         '<pad>': 0,
         '<start>': 1,
         '<end>': 2,
         '<unk>': 3
      }
      self.id_to_token = {idx: token for idx, token in enumerate(self.token_to_id)}
      self.vocab_size = len(self.token_to_id);
   
   def clean_text(self, text):
      text = text.lower()
      text = re.sub(r'[^\w\s]', '', text)
      return text

   def build_vocab(self, texts):
      for text in texts:
         text = self.clean_text(text)
         tokens = text.split()
         for token in tokens:
            if token not in self.token_to_id:
               token_id = len(self.token_to_id)
               self.token_to_id[token] = token_id
               self.id_to_token[token_id] = token
               self.vocab_size += 1

   def pad_sequence(self, sequences, max_length=None):
      batch_size = len(sequences)
      if max_length is None:
         max_length = max(len(seq) for seq in sequences)
      padded_sequences = np.full((batch_size, max_length), self.token_to_id['<pad>'], dtype=np.int32)
      for i, seq in enumerate(sequences):
         padded_sequences[i, :min(len(seq), max_length)] = seq[:max_length]
      return padded_sequences

   def encode(self, text):
      text = self.clean_text(text)
      tokens = text.split()
      token_ids = [self.token_to_id.get(token, self.token_to_id['<unk>']) for token in tokens]
      token_ids = [self.token_to_id['<start>']] + token_ids + [self.token_to_id['<end>']]
      return token_ids

   def decode(self, token_ids):
      tokens = []
      for token_id in token_ids:
         token = self.id_to_token.get(token_id, '<unk>')
         if token in ['<start>', '<pad>']:
            continue
         elif token == '<end>':
            break
         tokens.append(token)
      return ' '.join(tokens)
   
   def save_vocab(self, file_path):
      with open(file_path, 'w') as f:
         json.dump(self.token_to_id, f)

   def load_vocab(self, file_path):
      with open(file_path, 'r') as f:
         self.token_to_id = json.load(f)
         self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
         self.vocab_size = len(self.token_to_id)