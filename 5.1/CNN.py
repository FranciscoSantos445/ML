import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image

# Step 1: Load and preprocess the dataset (this assumes you have images in two folders 'class_0' and 'class_1')
def load_images(image_folder):
    images = []
    labels = []
    
    for label_folder in ['class_0', 'class_1']:
        label = int(label_folder[-1])
        folder_path = os.path.join(image_folder, label_folder)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            img = Image.open(image_path).convert('RGB')
            img = img.resize((64, 64))  # Resize all images to 64x64
            img = np.array(img) / 255.0  # Normalize the image
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# To predict on a new image:
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((64, 64))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Reshape for model input
    prediction = model.predict(img)
    return int(prediction[0] > 0.5)  # Returns 1 or 0 based on the threshold

# Assume your dataset is structured like: dataset/class_0/ and dataset/class_1/
image_folder = 'path_to_your_image_dataset'
images, labels = load_images(image_folder)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 2: Define the CNN model
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten and fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid for binary classification (output 0 or 1)

# Step 3: Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 5: Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Example prediction
print(predict_image('path_to_your_new_image'))
