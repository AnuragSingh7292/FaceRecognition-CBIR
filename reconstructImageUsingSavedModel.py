import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from google.colab import files

# Check the contents of the current directory
print("Files in current directory:", os.listdir('.'))

def se_block(input_tensor, ratio=16):
    # Define your squeeze-and-excitation block here
    pass  # Replace 'pass' with actual implementation


# Load the saved model (make sure the filename is correct)
try:
    autoencoder = load_model("perfect_autoencoder_model.h5", custom_objects={'se_block': se_block})
    print("Model loaded successfully.")
except FileNotFoundError as e:
    print("Error loading model:", e)

# Function to preprocess a single image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Function to reconstruct the image
def reconstruct_image(img_path):
    img_array = preprocess_image(img_path)
    reconstructed_img = autoencoder.predict(img_array)
    return img_array[0], reconstructed_img[0]  # Remove batch dimension for display

# Upload an image to test
uploaded = files.upload()
img_name = list(uploaded.keys())[0]

# Perform reconstruction
original_img, reconstructed_img = reconstruct_image(img_name)

# Display original and reconstructed images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(reconstructed_img)
plt.axis("off")
plt.show()
