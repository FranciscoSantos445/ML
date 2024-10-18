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
    rotation_range= 90,
    horizontal_flip=True,
    zoom_range=(0.9, 1.0)
)

# Display images generated using ImageDataGenerator
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

# Pick 10 random images to augment
for i in range(10):
    img = X[i].reshape((1, image_size, image_size, 1))  # Reshape to 4D for datagen
    augmented_img = next(datagen.flow(img, batch_size=1))[0]  # Generate one augmented image
    axes[i].imshow(augmented_img.reshape(image_size, image_size), cmap='gray')
    axes[i].axis('off')

plt.legend("Augmented images using ImageDataGenerator:")
plt.tight_layout()

exit()

# Flatten images for SMOTE
X_flat = X.reshape(X.shape[0], -1)

# Apply SMOTE
smote = SMOTE()
X_smote, Y_smote = smote.fit_resample(X_flat, Y)

# Reshape SMOTE-generated data back to 48x48 images
X_smote = X_smote.reshape(-1, image_size, image_size, 1)

# Display images generated using SMOTE
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

# Pick 10 random SMOTE images
for i in range(10):
    axes[i].imshow(X_smote[i].reshape(image_size, image_size), cmap='gray')
    axes[i].axis('off')

plt.legend("Images after SMOTE augmentation:")
plt.tight_layout()
plt.show()
