import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

def z_score_normalizer(arr):
    return (arr - np.mean(arr)) / np.std(arr)

#code for extra data

# Load the trained model (if needed) or use the existing trained model
#model = tf.keras.models.load_model('Model_CNN.h5')
#model = joblib.load('best_knn_model.joblib')
model = joblib.load('best_rf_model.joblib')

# Load the new data
new_data = np.load('Xtrain1_extra.npy')

# Normalize the new data (same as before)
new_data = z_score_normalizer(new_data)

# Reshape the new data into 48x48 images with 1 channel (grayscale)
image_size = 48
new_data = new_data.reshape(-1, image_size, image_size, 1)

new_data = new_data.reshape((new_data.shape[0], -1))

# Make predictions on the new data
new_predictions = model.predict(new_data)

# Convert the predicted probabilities to class labels (0 or 1)
new_predicted_classes = (new_predictions > 0.5).astype(int)

# Print the predictions for the first 10 images
print("Predicted labels for new data: ", new_predicted_classes[:50].flatten())

# Optional: Visualize a few of the new data images with their predicted labels
fig, axes = plt.subplots(5, 10, figsize=(12, 6))
axes = axes.ravel()

for i in range(50):
    axes[i].imshow(new_data[i].reshape(48, 48), cmap='gray')
    axes[i].set_title(f'Pred: {new_predicted_classes[i]}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
