import os
import pandas as pd
from zipfile import ZipFile


csv_path = "D:\VR_Assignment2_HarshChandrakar_MT2024054-main\VR_Assignment2_HarshChandrakar_MT2024054-main\merging data\simplified_vqa_dataset.csv"
image_root = r"D:\VR_Assignment2_HarshChandrakar_MT2024054-main\VR_Assignment2_HarshChandrakar_MT2024054-main\abo-images-small\images\small"
zip_name = "collected_images.zip"  
df = pd.read_csv(csv_path)
image_paths = df['path'].unique()  # Get unique image paths (foldername/imagename)

# Create ZIP and add images
with ZipFile(zip_name, 'w') as zipf:
    for rel_path in image_paths:
        # Split into folder and image name
        parts = rel_path.split('/')
        if len(parts) != 2:
            print(f"Skipping invalid path format: {rel_path}")
            continue
        folder, image = parts
        abs_path = os.path.join(image_root, folder, image)
        if os.path.isfile(abs_path):
            # Store in zip as foldername/imagename
            zipf.write(abs_path, arcname=rel_path)
        else:
            print(f"Missing: {abs_path}")

print(f"Created {zip_name} with {len(image_paths)} images.")
