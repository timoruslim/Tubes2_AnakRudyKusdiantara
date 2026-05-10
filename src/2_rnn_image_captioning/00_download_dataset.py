import os
import subprocess
import shutil
import tempfile
from pathlib import Path

def download_and_extract():

   dataset_name = "adityajn105/flickr8k"
   script_dir = Path(__file__).parent
   project_root = script_dir.parent.parent
   output_dir = project_root / "data" / "2_rnn_image_captioning"
   
   os.makedirs(output_dir, exist_ok=True)
   images_target_dir = output_dir / "images"
   os.makedirs(images_target_dir, exist_ok=True)
   
   with tempfile.TemporaryDirectory() as temp_dir:

      print(f"\nDownloading {dataset_name} to temporary directory...")
      subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", temp_dir, "--unzip"], check=True)
      
      print("Rearranging files to match desired structure...")

      print(f"Moving captions.txt to {output_dir}")
      captions_src = os.path.join(temp_dir, "captions.txt")
      if os.path.exists(captions_src):
         target_captions = output_dir / "captions.txt"
         if target_captions.exists():
               os.remove(target_captions)
         shutil.move(captions_src, target_captions)

      print(f"Moving images to {images_target_dir}")
      images_src_dir = os.path.join(temp_dir, "Images")
      if os.path.exists(images_src_dir):
         for filename in os.listdir(images_src_dir):
               src_file = os.path.join(images_src_dir, filename)
               dst_file = images_target_dir / filename
               
               if not dst_file.exists(): 
                  shutil.move(src_file, dst_file)
         
      print(f"Done! Dataset is ready at {output_dir}")

if __name__ == "__main__":
   download_and_extract()
