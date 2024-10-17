# autoencoder.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def z_score_normalizer(arr):
    return (arr - np.mean(arr)) / np.std(arr)


def add_varying_noise(images, low_noise_factor=0.1, high_noise_factor=0.5):
    noisy_images = []
    for img in images:
        # Randomly choose a noise factor between low and high
        noise_factor = np.random.uniform(low_noise_factor, high_noise_factor)
        noise = np.random.normal(loc=0.0, scale=1.0, size=img.shape)
        noisy_img = img + noise_factor * noise
        noisy_img = np.clip(noisy_img, -1., 1.)  # Ensure values stay within [-1, 1]
        noisy_images.append(noisy_img)
    
    return np.array(noisy_images)


def generate_augmented_data(X_class, num_images):
    augmented_images = []
    batch_size = 32
    augment_batches = (num_images // batch_size) + 1  # Calculate how many batches are needed

    for i in range(augment_batches):
        for X_batch in datagen.flow(X_class, batch_size=batch_size):
            augmented_images.append(X_batch)
            if len(augmented_images) * batch_size >= num_images:
                break

    # Flatten the list of augmented batches into a single array
    augmented_images = np.concatenate(augmented_images, axis=0)[:num_images]

    return augmented_images

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Define the autoencoder model
def build_autoencoder():
    
    input_img = layers.Input(shape=(48, 48, 1))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Latent space
    encoded = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='tanh', padding='same')(x)

    # Autoencoder model
    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss= ssim_loss)
    return autoencoder

# Train the model
def train_autoencoder(autoencoder,noisy_x_train, x_train, x_test, noisy_x_test):
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
    autoencoder.fit(noisy_x_train, x_train, epochs = 500 , batch_size = 512, validation_data=(noisy_x_test, x_test), callbacks=[early_stopping])
    
X = np.load ('Xtrain1_extra.npy')

image_size = 48 # Size of the images

num_images = 1000 # Number of images to generate

X = z_score_normalizer(X)

X = X.reshape((-1, image_size, image_size, 1))

# Set up the ImageDataGenerator for augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=5,         # rotate the image by up to 5 degrees in either direction
    horizontal_flip=True,     # flip the image horizontally
    vertical_flip=True,       # flip the image vertically
    zoom_range=(0.9, 1.0),    # zoom out by 10%
    brightness_range=(0.9, 1.1)  # darken or brighten by 10%
)

# Generate additional images using the ImageDataGenerator
augmented_images = []

augmented_images = generate_augmented_data(X, num_images)

# Combine the original and augmented data
X = np.concatenate([X, augmented_images], axis=0)

x_train, x_test = train_test_split(X, test_size=0.2,shuffle=True, random_state=42)

autoencoder = build_autoencoder()

noisy_x_train = add_varying_noise(x_train, low_noise_factor=0.1, high_noise_factor=0.5)
noisy_x_test = add_varying_noise(x_test, low_noise_factor=0.1, high_noise_factor=0.5)

train_autoencoder(autoencoder,noisy_x_train, x_train, x_test,noisy_x_test)

reconstructed_images = autoencoder.predict(x_test)

test_loss = autoencoder.evaluate(reconstructed_images, x_test)
print(f"Reconstruction Loss (Binary Cross-Entropy): {test_loss}")

# # Plot a few original and reconstructed images
# n = 10  # Number of samples to display
# plt.figure(figsize=(20, 4))

# for i in range(n):
#     # Display original images
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(48, 48), cmap='gray')
#     plt.title("Original")
#     plt.axis('off')

#     # Display reconstructed images
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(reconstructed_images[i].reshape(48, 48), cmap='gray')
#     plt.title("Reconstructed")
#     plt.axis('off')


# plt.show() 

autoencoder.save('autoencoder.h5')