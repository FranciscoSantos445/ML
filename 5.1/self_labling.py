import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

num_images = 200 # Number of images to generate

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=5,         # rotate the image by up to 5 degrees in either direction
    horizontal_flip=True,     # flip the image horizontally
    vertical_flip=True,       # flip the image vertically
    zoom_range=(0.9, 1.0),    # zoom out by 10%
    brightness_range=(0.9, 1.1)  # darken or brighten by 10%
)

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

X_augmented , Y_augmented = generate_augmented_data(X_train, y_train, num_images)

X_train = np.concatenate([X_train, X_augmented], axis=0)
y_train = np.concatenate([y_train, Y_augmented], axis=0)

print (f'Number of pseudo-labeled samples: {len(y_train)}')
print (f'Number of high-confidence samples: {len(X_train)}')

# Retrain the model using the combined data
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32,callbacks=[early_stopping])

X = np.load('Xtrain1.npy')
Y = np.load('Ytrain1.npy')

# Normalize the data
X = z_score_normalizer(X)

# Reshape the data into 48x48 images with 1 channel (grayscale)
X = X.reshape(-1, image_size, image_size, 1)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, Y = X[indices], Y[indices]  # Shuffle the data

# Make predictions on the original data
predictions = model.predict(X)

# Compare predictions with Y
predicted_labels = (predictions > 0.5).astype(int).flatten()
accuracy = len(predicted_labels == Y) / len(Y)

print(f'Accuracy of the model on the original data: {accuracy * 100:.2f}%')

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
