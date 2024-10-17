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

# Generate additional data for both positive and negative classes
X_positive_augmented, Y_positive_augmented = generate_augmented_data(X_positive, Y_positive, num_additional_images)
X_negative_augmented, Y_negative_augmented = generate_augmented_data(X_negative, Y_negative, num_additional_images)

# Combine the original and augmented data
X = np.concatenate([X, X_positive_augmented, X_negative_augmented], axis=0)
Y = np.concatenate([Y, Y_positive_augmented, Y_negative_augmented], axis=0)

X = X.reshape((X.shape[0], -1))

# Make predictions on the new data
new_predictions = model.predict(X)

# Convert the predicted probabilities to class labels (0 or 1)
new_predicted_classes = (new_predictions > 0.5).astype(int)

# Print the predictions for the first 10 images
print("Predicted labels for new data: ", new_predicted_classes[:].flatten())

# Compare the predicted classes with the actual labels
accuracy = np.mean(new_predicted_classes.flatten() == Y)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Optional: Visualize a few of the new data images with their predicted labels
fig, axes = plt.subplots(5, 10, figsize=(12, 6))
axes = axes.ravel()

for i in range(50):
    axes[i].imshow(X[i].reshape(48, 48), cmap='gray')
    axes[i].set_title(f'Pred: {new_predicted_classes[i]}, True: {Y[i]}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
