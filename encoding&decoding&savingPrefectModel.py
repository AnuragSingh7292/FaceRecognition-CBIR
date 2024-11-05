import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape,
    LeakyReLU, BatchNormalization, GlobalAveragePooling2D, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from google.colab import files
from tensorflow.keras.callbacks import ModelCheckpoint  # Import for saving best model

# Parameters
image_shape = (128, 128, 3)
encoding_dim = 48
epochs = 100
batch_size = 16

# SENet Channel Attention Block
def se_block(input_tensor, reduction_ratio=16):
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(channels // reduction_ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    return Multiply()([input_tensor, se])

# Encoder
def build_encoder(input_shape, encoding_dim):
    inputs = Input(shape=input_shape)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    x = se_block(x)

    x = Flatten()(x)
    encoded = Dense(encoding_dim)(x)
    return Model(inputs, encoded, name="encoder")

encoderR = build_encoder(image_shape, encoding_dim)
encoderR.summary()

# Decoder
def build_decoder(output_shape, encoding_dim):
    decoder_input = Input(shape=(encoding_dim,))
    x = Dense(64 * 64 * 128)(decoder_input)
    x = Reshape((64, 64, 128))(x)

    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', activation='sigmoid')(x)
    return Model(decoder_input, x, name="decoder")

decoderR = build_decoder(image_shape, encoding_dim)
decoderR.summary()

# Autoencoder
autoencoder_input = Input(shape=image_shape)
encoded_img = encoderR(autoencoder_input)
reconstructed_img = decoderR(encoded_img)

autoencoder = Model(autoencoder_input, reconstructed_img)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Load and preprocess the images
uploaded = files.upload()
img_names = list(uploaded.keys())
images = []

for img_name in img_names:
    img = image.load_img(img_name, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    images.append(img_array)

images = np.array(images)

# Split data into train and test sets
x_train, x_test = train_test_split(images, test_size=0.2, random_state=42)

# Set up ModelCheckpoint to save the model with the lowest validation loss
checkpoint = ModelCheckpoint(
    "perfect_autoencoder_model.keras",  # Change made here
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

# Train the autoencoder
history = autoencoder.fit(
    x_train, x_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[checkpoint]  # Add the checkpoint callback here
)

# Reconstruct the images using the best saved model
reconstructed_images = autoencoder.predict(x_test)

# Display original and reconstructed images
plt.figure(figsize=(10, 4))
for i in range(min(5, len(x_test))):
    # Display original
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.axis('off')

    # Display reconstruction
    plt.subplot(2, 5, i + 6)
    plt.imshow(reconstructed_images[i])
    plt.axis('off')

plt.show()
