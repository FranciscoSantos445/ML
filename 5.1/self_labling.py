# Grupo 94

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def z_score_normalizer(arr):
    return (arr - np.mean(arr)) / np.std(arr)

# Load the trained model
model = tf.keras.models.load_model('Model_CNN.h5', compile=True)

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
confidence_threshold = 0.95

# Convert the predicted probabilities to pseudo-labels if confidence is high enough
pseudo_labels = []
for pred in new_predictions:
    if pred > confidence_threshold:
        pseudo_labels.append(1)
    elif pred < (1 - confidence_threshold):
        pseudo_labels.append(0)

y_train = np.array([label for label in pseudo_labels if label is not None])
X_train = new_data[:len(y_train)]  # Keep only high-confidence samples

print(f'Number of high-confidence samples: {len(y_train)}')

# Retrain the model using the combined data
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=2)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32,callbacks=[early_stopping])

X = np.load('Xtrain1.npy')
Y = np.load('Ytrain1.npy')

# Normalize the data
X = z_score_normalizer(X)

# Reshape the data into 48x48 images with 1 channel (grayscale)
X = X.reshape(-1, image_size, image_size, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Make predictions on the original data
predictions = model.predict(X_test)

# Compare predictions with Y
val_predicted_classes = (predictions > 0.5).astype(int).flatten()

f1 = f1_score(y_test, val_predicted_classes)

print (f"F1 score: {f1:.3f}")

# Save the updated model
model.save('Model_CNN_updated.h5')

# Plotting the history of the best model
plt.figure(figsize=(12, 6))

# Plot training loss and validation loss over the epochs from the best model's history
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training accuracy and validation accuracy over the epochs
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
