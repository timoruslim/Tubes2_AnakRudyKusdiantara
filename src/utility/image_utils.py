import numpy as np
import PIL.Image as Image

def load_image(file_path, target_size=None):
   image = Image.open(file_path)
   rgb_image = image.convert('RGB')
   if target_size is not None:
      rgb_image = rgb_image.resize(target_size)
   return np.array(rgb_image)