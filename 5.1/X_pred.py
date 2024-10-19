# Grupo 94

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def z_score_normalizer(arr):
    return (arr - np.mean(arr)) / np.std(arr)

# Load the trained model
model = tf.keras.models.load_model('Model_CNN.h5', compile=True)

# Load the new data
new_data = np.load('Xtest1.npy')

# Normalize the new data (same as before)
new_data = z_score_normalizer(new_data)

# Reshape the new data into 48x48 
image_size = 48
new_data = new_data.reshape(-1, image_size, image_size, 1)

# Make predictions on the new data
new_predictions = model.predict(new_data)

# Convert the predicted probabilities to class labels (0 or 1)
new_predicted_classes = (new_predictions > 0.5).astype(int)

# Flatten the array
new_predicted_classes = new_predicted_classes.flatten()

# Save the predicted classes to a file
np.save ('Ytest1.npy', new_predicted_classes)