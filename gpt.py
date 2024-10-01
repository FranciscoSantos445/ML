import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image

# Step 1: Load and preprocess the dataset
def load_images(image_folder):
    images = []
    labels = []
    
    for label_folder in ['class_0', 'class_1']:  # class_0 -> no crater, class_1 -> crater
        label = int(label_folder[-1])
        folder_path = os.path.join(image_folder, label_folder)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            img = Image.open(image_path).convert('RGB')  # For grayscale: .convert('L')
            img = img.resize((48, 48))  # Resize to a larger resolution to capture craters
            img = np.array(img) / 255.0  # Normalize pixel values
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

image_folder = 'path_to_your_image_dataset'
images, labels = load_images(image_folder)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 2: Add Data Augmentation
data_gen = ImageDataGenerator(
    rotation_range=15,       # Small rotations to simulate different angles of craters
    zoom_range=0.2,          # Random zoom to deal with different sizes of craters
    horizontal_flip=True,    # Flip horizontally for augmentation
    vertical_flip=True,      # Vertical flip might also be helpful depending on crater orientation
    brightness_range=[0.8, 1.2]  # Small changes in brightness
)

# Step 3: Define the CNN model
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))  # Adjusted for larger image size
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(48, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(48, (3, 3), activation='relu'))  # Added one more conv layer for better feature extraction
model.add(layers.MaxPooling2D((2, 2)))

# Flatten and fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(48, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model with augmentation
batch_size = 32
history = model.fit(
    data_gen.flow(X_train, y_train, batch_size=batch_size),  # Use the data generator for training
    epochs=15,  # Train for more epochs, if necessary
    validation_data=(X_test, y_test)
)

# Step 6: Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Predict function for new images
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')  # For grayscale images, use 'L'
    img = img.resize((48, 48))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Reshape for model input
    prediction = model.predict(img)
    return int(prediction[0] > 0.5)  # Return 1 for crater, 0 for no crater

# Example prediction
print(predict_image('path_to_your_new_image'))
