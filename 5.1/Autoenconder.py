# autoencoder.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def z_score_normalizer(arr):
    return (arr - np.mean(arr)) / np.std(arr)

# Define the autoencoder model
def build_autoencoder():
    input_img = layers.Input(shape=(28, 28, 1))

    # Encoder
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Latent space
    encoded = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Autoencoder model
    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

# Train the model
def train_autoencoder(autoencoder, x_train, x_test):
    autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Main function
def main():
    
    X = np.load ('Xtrain1.npy')
    
    image_size = 48 # Size of the images

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

    augmented_images = datagen.flow(X, batch_size=512, shuffle=True)    
    augmented_images = np.array(augmented_images)
    
    # Combine the original and augmented data
    X = np.concatenate([X, augmented_images], axis=0)
    
    x_train, x_test = train_test_split(X, test_size=0.2,shuffle=True, random_state=42)
    autoencoder = build_autoencoder()
    train_autoencoder(autoencoder, x_train, x_test)
    
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

    #     plt.show()

    autoencoder.save('autoencoder.h5')

if __name__ == "__main__":
    main()