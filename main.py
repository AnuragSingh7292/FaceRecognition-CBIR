

import os
import re  # For extracting numeric parts of filenames

# Define the original and renamed folder paths
original_folder = 'C:/Users/anura/OneDrive/Documents/all In one/TCIA'
rename_folder = 'C:/Users/anura/OneDrive/Documents/all In one/TCIA_rename_correctly'

# Ensure the renamed folder exists
os.makedirs(rename_folder, exist_ok=True)

# Helper function to extract numbers from filenames
def extract_number(filename):
    match = re.search(r'\d+', filename)  # Find the first numeric part in the filename
    return int(match.group()) if match else float('inf')  # Return a large number if no number is found

# Get all image files in the original folder and sort them
image_files = [f for f in os.listdir(original_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
image_files.sort(key=extract_number)

# Rename and save the images in the new folder
for index, image in enumerate(image_files, start=1):
    src_path = os.path.join(original_folder, image)
    dst_path = os.path.join(rename_folder, f"{index}.tif")
    os.rename(src_path, dst_path)

print("Images renamed and ordered successfully!")
