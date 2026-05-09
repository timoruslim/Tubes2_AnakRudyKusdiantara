import numpy as np

def load_keras_weights(custom_model, keras_model, verbose=False):
   try:
      import tensorflow as tf
   except ImportError:
      raise ImportError("TensorFlow is required to load Keras weights.")
   
   if not hasattr(custom_model, 'built') or not custom_model.built:
      raise RuntimeError("You must add Input layer or call build() before loading weights.")
   
   keras_model = tf.keras.models.load_model(keras_model) if isinstance(keras_model, str) else keras_model
   if not isinstance(keras_model, tf.keras.Model):
      raise ValueError(f"Expected keras_model to be a Keras Model instance or file path, got {type(keras_model).__name__}")
   
   keras_params = keras_model.get_weights()
   custom_params = custom_model.tensors()
   
   if len(keras_params) != len(custom_params):
      raise ValueError(
         f"Parameter count mismatch: Keras model {len(keras_params)} parameters vs Custom model {len(custom_params)} parameters."
      )
   
   print("\n" + "=" * 67)
   
   for i, (keras_param, custom_param) in enumerate(zip(keras_params, custom_params)):
      k_size = np.prod(keras_param.shape)
      c_size = np.prod(custom_param.data.shape)
      
      if k_size != c_size:
         raise ValueError(
               f"Element count mismatch at parameter index {i}: Keras shape {keras_param.shape} ({k_size} elements) vs Custom shape {custom_param.data.shape} ({c_size} elements)."
         )
      
      if verbose:
         print(f"  param_{i}: Keras {keras_param.shape} -> Custom {custom_param.data.shape}")
      
      custom_param.data = keras_param.reshape(custom_param.data.shape)

   print("=" * 67)
   
   print(f"Successfully loaded {len(keras_params)} parameters from Keras model.")

   print("=" * 67 + "\n")
