import numpy as np
import PIL.Image as Image

def load_image(file_path, target_size=None):

   try:
      with Image.open(file_path) as image:

         rgb_image = image.convert('RGB')
         if target_size is not None:
            rgb_image = rgb_image.resize(target_size, resample=Image.Resampling.LANCZOS)
         
         image_data = np.array(rgb_image, dtype=np.float32)
         image_data_normalized = image_data / 255.0

         return image_data_normalized
   
   except Exception as e:
      print(f"Error loading image from {file_path}: {e}")
      return None

def load_batch(file_paths, target_size=None):
   
   image_datas = []

   for file_path in file_paths:
      image_data = load_image(file_path, target_size)
      if image_data is not None:
         image_datas.append(image_data)
   
   return np.array(image_datas)