import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Load the data
X = np.load('Xtrain1.npy')
Y = np.load('Ytrain1.npy')

# Normalize the data using Z-score normalization
def z_score_normalizer(arr):
    return (arr - np.mean(arr)) / np.std(arr)

X = z_score_normalizer(X)

print("Original data shape:", X.shape)

# Reshape the data into 48x48 images (assuming grayscale)
image_size = 48
X = X.reshape(-1, image_size, image_size, 1)

# Initialize ImageDataGenerator with augmentation parameters
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,     # flip the image horizontally
    vertical_flip=True,       # flip the image vertically
    brightness_range=(0.90, 1.1) # darken or brighten by 10%
)

# Display images generated using ImageDataGenerator
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

# Pick 10 random images to augment
for i in range(5):
    img = X[i].reshape((1, image_size, image_size, 1))  # Reshape to 4D for datagen
    augmented_img = next(datagen.flow(img, batch_size=1))[0]  # Generate one augmented image
    
    # Display original image
    axes[2 * i].imshow(img.reshape(image_size, image_size), cmap='gray')
    axes[2 * i].axis('off')
    axes[2 * i].set_title('Original')
    
    # Display augmented image
    axes[2 * i + 1].imshow(augmented_img.reshape(image_size, image_size), cmap='gray')
    axes[2 * i + 1].axis('off')
    axes[2 * i + 1].set_title('Augmented')

plt.legend("Augmented images using ImageDataGenerator:")
plt.tight_layout()
plt.show()
