import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io

# Load the .npy file containing the feature vectors
features_path = 'C:/Users/anura/OneDrive/Desktop/features.npy'
features_vector = np.load(features_path)
D = features_vector


# Change directory to the location of images
os.chdir('C:/Users/anura/OneDrive/Desktop/all In one/ORL all renamed tif')
# "C:/Users/anura/OneDrive/Desktop/all In one\ORL all images tif.zip"
# "C:/Users/anura/OneDrive/Desktop/all In one/ORL all renamed tif"

# Function to select the query image
root = tk.Tk()
root.withdraw()  # Hide the root window
file = filedialog.askopenfilename(title='Select any image')

# Extract file name and extension
path, file1 = os.path.split(file)
name, ext = os.path.splitext(file1)

# Query image features
Q = None
for i in range(1,401):
    if int(name) == i:  # Adding 1 since indexing starts at 0 in Python
        Q = D[i-1, :]
        break

if Q is None:
    raise ValueError("Query image not found in feature set.")

# Distance measurement
s2 = []
for i in range(400):

    # Convert D and Q to signed integers to avoid overflow for
    D_signed = D.astype(np.int32)
    Q_signed = Q.astype(np.int32)

    s1 = np.abs(D_signed[i,:] - Q_signed)
    s11 = np.abs(1 + D_signed[i,:] + Q_signed)
    p = np.divide(s1,s11)

    s2.append(np.sum(p))
# print(s2[0:10])

    # print("d(",i,")", D[i,0:3],"q = ", Q[0:3])
    # print("p = ",p,"s1 = ", s1 , "s11 = ", s11)
    

# Sort the distances and get the indices of the closest images
sorted_indices = np.argsort(s2)
# print(sorted_indices[0:10])
# print(sorted_indices)

# Ask the user how many images to retrieve
num = int(input('Enter the number of images you want to retrieve: '))

# Display the query image
plt.figure(figsize=(12,12))

# Load and display the query image
query_img = Image.open(os.path.join(path, file1))
plt.subplot((num // 5) + 2, 5, 3)
plt.imshow(query_img)
plt.title(name)
plt.axis('off') #  Turn off axis for the query image

# Display the retrieved images
for i in range(num):
    img_idx = sorted_indices[i]
    img_name = str(img_idx + 1) + ext
    img_path = os.path.join(path, img_name)

    try:
        retrieved_img = Image.open(img_path)
        plt.subplot((num // 5) + 2, 5, i + 6)
        plt.imshow(retrieved_img)
        # plt.title(img_name) # with extension will come the image 
        plt.title(img_name.replace('.tif', '')) # to remove extension form the image 
        plt.axis('off') # Turn off axis for the query image
    except FileNotFoundError:
        print(f"Image {img_name} not found.")

plt.tight_layout()
plt.show()
