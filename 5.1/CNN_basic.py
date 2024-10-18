import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def z_score_normalizer(arr):
    return (arr - np.mean(arr)) / np.std(arr)

# Load the data
X = np.load('Xtrain1.npy')
Y = np.load('Ytrain1.npy')

# Normalize the data (optional but recommended for CNNs)
#X = X.astype('float32') / 255.0
X = z_score_normalizer(X)

# Reshape the data into 48x48 images
image_size = 48
X = X.reshape(-1, image_size, image_size, 1)  # Adding a channel dimension for grayscale

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (crater or no crater)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f'Test accuracy: {test_acc}')
# Get predictions from the model
predictions = model.predict(X_test)

model.save('model_CNN.h5')

# Convert probabilities to class labels (0 or 1)
predicted_classes = (predictions > 0.5).astype(int)

# Print the first 10 actual and predicted values
print("Actual labels: ", y_test[:10])
print("Predicted labels: ", predicted_classes[:10].flatten())

# Optional: Visualize a few test images with their predicted labels
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(10):
    axes[i].imshow(X_test[i].reshape(48, 48), cmap='gray')
    axes[i].set_title(f'Pred: {predicted_classes[i][0]}, Actual: {y_test[i]}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()