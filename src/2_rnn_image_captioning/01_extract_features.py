import os
import sys
import numpy as np 
from pathlib import Path
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

sys.path.append(str(SCRIPT_DIR.parent))
from utility.image_utils import load_image

def extract_and_output_features(image_dir, output_dir, target_size=(299, 299)):

   cnn_model = InceptionV3(weights='imagenet', pooling='avg', include_top=False)
   cnn_model.trainable = False

   for filename in tqdm(os.listdir(image_dir)):
      if filename.endswith(('.png', '.jpg', '.jpeg')):

         image_id = os.path.splitext(filename)[0]
         image_path = os.path.join(image_dir, filename)

         try:
            image = load_image(image_path, target_size)
            if image is None:
               continue
         except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

         image = preprocess_input(image * 255.0)

         feature = cnn_model.predict(image[np.newaxis, ...], verbose=0)
         feature = feature.squeeze()

         np.save(os.path.join(output_dir, f'{image_id}.npy'), feature)

if __name__ == "__main__":
   image_dir = PROJECT_ROOT / 'data' / '2_rnn_image_captioning' / 'images'
   output_dir = PROJECT_ROOT / 'data' / '2_rnn_image_captioning' / 'features'
   os.makedirs(output_dir, exist_ok=True)
   features = extract_and_output_features(image_dir, output_dir, target_size=(299, 299))