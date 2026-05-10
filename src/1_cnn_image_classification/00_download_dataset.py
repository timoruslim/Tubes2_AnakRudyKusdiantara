import os
import subprocess
import shutil
import tempfile
from pathlib import Path

def download_and_extract():

   dataset_name = "puneet6060/intel-image-classification"
   script_dir = Path(__file__).parent
   project_root = script_dir.parent.parent
   output_dir = project_root / "data" / "1_cnn_image_classification"
   
   os.makedirs(output_dir, exist_ok=True)
   
   with tempfile.TemporaryDirectory() as temp_dir:

      print(f"\nDownloading {dataset_name} to temporary directory...")
      subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", temp_dir, "--unzip"], check=True)
      
      print("Rearranging files to match desired structure...")
      target_folders = ["seg_train", "seg_test", "seg_pred"]
      
      for root, dirs, _ in os.walk(temp_dir):
         for dir_name in dirs:
               if dir_name in target_folders:
                  source_dir = os.path.join(root, dir_name)
                  
                  inner_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
                  if len(inner_dirs) == 1 and inner_dirs[0] == dir_name:
                     continue
                  
                  target_path = output_dir / dir_name
                  
                  if target_path.exists():
                     shutil.rmtree(target_path)
                  
                  print(f"Moving {source_dir} to {target_path}...")
                  shutil.move(source_dir, target_path)

      print(f"Done! Dataset is ready at {output_dir}")

if __name__ == "__main__":
   download_and_extract()
