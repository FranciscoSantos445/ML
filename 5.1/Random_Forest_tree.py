##########################  RANDOM FOREST TREE ########################################################

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.metrics import f1_score,accuracy_score,make_scorer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import joblib

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

# Load the data
X = np.load('Xtrain1.npy')
Y = np.load('Ytrain1.npy')
num_images = 1000 # Number of images to generate
image_size = 48 # Size of the images

X = z_score_normalizer(X)

X = X.reshape((-1, image_size, image_size, 1))

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

param_grid = {
    'n_estimators': [100, 150, 200, 300],  # The number of trees in the forest
    'max_depth': [None, 10, 20, 30],           # The maximum depth of the trees
    'min_samples_split': [2, 5, 10],           # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],             # Minimum number of samples required to be at a leaf node
}

scorer = make_scorer(f1_score)

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

# Variable to keep track of the best F1 score and the best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_f1_score = 0
best_model = None

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,shuffle=True, random_state=42)

rf = RandomForestClassifier(random_state=42)

random_forest = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring=scorer, verbose=2, n_jobs=-1)

random_forest.fit(x_train, y_train)

model = random_forest.best_estimator_

y_pred = model.predict(X)

# Evaluate the model's performance
accuracy = accuracy_score(Y, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model,'best_rf_model.joblib')