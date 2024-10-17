import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

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

def z_score_normalizer(arr):
    return (arr - np.mean(arr)) / np.std(arr)

#code for extra data

# Load the trained model (if needed) or use the existing trained model
#model = tf.keras.models.load_model('Model_CNN.h5')
#model = joblib.load('best_knn_model.joblib')
model = joblib.load('best_rf_model.joblib')

# Load the new data
X = np.load('Xtrain1.npy')
Y = np.load('Ytrain1.npy')

# Normalize the new data (same as before)
X = z_score_normalizer(X)

# Reshape the new data into 48x48 images with 1 channel (grayscale)
image_size = 48
X = X.reshape(-1, image_size, image_size, 1)

# Number of additional images per class to generate (for both positive and negative)
num_additional_images = 1000  # Change this number to generate more or less data

# Set up the ImageDataGenerator for augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=5,         # rotate the image by up to 5 degrees in either direction
    horizontal_flip=True,     # flip the image horizontally
    vertical_flip=True,       # flip the image vertically
    zoom_range=(0.9, 1.0),    # zoom out by 10%
    brightness_range=(0.9, 1.1)  # darken or brighten by 10%
)

X = X.reshape((X.shape[0], -1))

# Make predictions on the new data
new_predictions = model.predict(X)