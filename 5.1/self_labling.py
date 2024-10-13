import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def z_score_normalizer(arr):
    return (arr - np.mean(arr)) / np.std(arr)

# Load the trained model
model = tf.keras.models.load_model('Model_CNN.h5')

# Load the new data
new_data = np.load('Xtrain1_extra.npy')

# Normalize the new data
new_data = z_score_normalizer(new_data)

# Reshape the new data into 48x48 images with 1 channel (grayscale)
image_size = 48
new_data = new_data.reshape(-1, image_size, image_size, 1)

# Make predictions on the new data
new_predictions = model.predict(new_data)

# Set a confidence threshold (e.g., 0.9) for pseudo-labeling
confidence_threshold = 0.9

# Convert the predicted probabilities to pseudo-labels if confidence is high enough
pseudo_labels = []
for pred in new_predictions:
    if pred > confidence_threshold:
        pseudo_labels.append(1)
    elif pred < (1 - confidence_threshold):
        pseudo_labels.append(0)

y_train = np.array([label for label in pseudo_labels if label is not None])
X_train = new_data[:len(y_train)]  # Keep only high-confidence samples

print (f'Number of pseudo-labeled samples: {len(y_train)}')
print (f'Number of high-confidence samples: {len(X_train)}')

# Retrain the model using the combined data
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the updated model
model.save('Model_CNN_updated.h5')

# Plot the images and their pseudo-labels
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
axes = axes.flatten()

for img, label, ax in zip(new_data[:25], pseudo_labels[:25], axes):
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')

plt.tight_layout()
plt.show()
