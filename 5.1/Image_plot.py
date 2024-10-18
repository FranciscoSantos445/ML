import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Normalize the data using Z-score normalization
def z_score_normalizer(arr):
    return (arr - np.mean(arr)) / np.std(arr)

# Function to generate augmented data
def generate_augmented_data(X_class, Y_class, num_images):
    augmented_images = []
    augmented_labels = []
    batch_size = 32
    augment_batches = (num_images // batch_size) + 1  # Calculate how many batches are needed
    
    for i in range(augment_batches):
        for X_batch, y_batch in datagen.flow(X_class, Y_class, batch_size=batch_size):
            augmented_images.append(X_batch)
            augmented_labels.append(y_batch)
            if len(augmented_images) * batch_size >= num_images:
                break
    
    # Flatten the list of augmented batches into a single array
    augmented_images = np.concatenate(augmented_images, axis=0)[:num_images]
    augmented_labels = np.concatenate(augmented_labels, axis=0)[:num_images]
    
    return augmented_images, augmented_labels

# Load the .npy file
X = np.load('Xtrain1.npy')
Y = np.load('Ytrain1.npy')

# Assuming the original images are 48x48 RGB (3 channels)

X = z_score_normalizer(X)

# Reshape X to have the correct dimensions: (num_images, height, width, channels)
X = X.reshape((-1, 48, 48, 1))

# Separate positive and negative examples
X_positive = X[Y == 1]
X_negative = X[Y == 0]
Y_positive = Y[Y == 1]
Y_negative = Y[Y == 0]

# Find out the number of examples in each class
num_positives = len(Y_positive)
num_negatives = len(Y_negative)

balance_count = num_positives - num_negatives

# Number of additional images per class to generate (for both positive and negative)
num_additional_images = 1000  # Change this number to generate more or less data

print(f"Original dataset size: {len(Y)}")
print(f"Positives: {num_positives}, Negatives: {num_negatives}")
print(f"Generating {num_additional_images} augmented samples for each class.")

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
augmented_labels = []

augmented_images, augmented_labels = generate_augmented_data(X_negative, Y_negative, balance_count)

# Combine the original and augmented data
X = np.concatenate([X, augmented_images], axis=0)
Y = np.concatenate([Y, augmented_labels], axis=0)

print (f"New dataset size: {len(Y)} (Balanced dataset)")

# Generate additional data for both positive and negative classes
X_positive_augmented, Y_positive_augmented = generate_augmented_data(X_positive, Y_positive, num_additional_images)
X_negative_augmented, Y_negative_augmented = generate_augmented_data(X_negative, Y_negative, num_additional_images)

print(f"Positive augmented data: {len(Y_positive_augmented)}")
print(f"Negative augmented data: {len(Y_negative_augmented)}")

# Combine the original and augmented data
X = np.concatenate([X, X_positive_augmented, X_negative_augmented], axis=0)
Y = np.concatenate([Y, Y_positive_augmented, Y_negative_augmented], axis=0)

print(f"New dataset size: {len(Y)} (Balanced dataset)")

num_images = 10  # Number of images to display

# Reshape each row (1D array of 2304) into a 48x48 image
image_size = (48, 48)
images = X_positive_augmented[:num_images].reshape(-1, *image_size)

# Plot the images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()  # Flatten the 2D array of axes for easier indexing
for i in range(num_images):
    axes[i].imshow(images[i], cmap='gray')
    axes[i].set_title(f"Label: {Y_positive_augmented[i]}")
    axes[i].axis('off')  # Turn off axis for clarity
plt.show()
