import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
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

#code with KFoldq , SMOTE and l2 regularization with grid search

# Load the data
X = np.load('Xtrain1.npy')
Y = np.load('Ytrain1.npy')
image_size = 48 # Size of the images

# Number of additional images per class to generate (for both positive and negative)
num_additional_images = 1000  # Change this number to generate more or less data

# Set up the ImageDataGenerator for augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,     # flip the image horizontally
    vertical_flip=True,       # flip the image vertically
    brightness_range=(0.9, 1.1)  # darken o brighten by 10%
)

param_grid = {
    'n_neighbors': list(range(1, 31)),  # Try values of n_neighbors from 1 to 30
    'weights': ['distance'],  # Test both uniform and distance-based weights
    'metric': ['cosine']  # Test different distance metrics
}

X = z_score_normalizer(X)

X = X.reshape((-1, image_size, image_size, 1))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Separate positive and negative examples
X_positive = X_train[y_train == 1]
X_negative = X_train[y_train == 0]
Y_positive = y_train[y_train == 1]
Y_negative = y_train[y_train == 0]

# Find out the number of examples in each class
num_positives = len(Y_positive)
num_negatives = len(Y_negative)

balance_count = num_positives - num_negatives

# Generate additional images using the ImageDataGenerator
augmented_images = []
augmented_labels = []

augmented_images, augmented_labels = generate_augmented_data(X_negative, Y_negative, balance_count)

# Combine the original and augmented data
X_train = np.concatenate([X_train, augmented_images], axis=0)
y_train = np.concatenate([y_train, augmented_labels], axis=0)

# Generate additional data for both positive and negative classes
X_positive_augmented, Y_positive_augmented = generate_augmented_data(X_positive, Y_positive, num_additional_images)
X_negative_augmented, Y_negative_augmented = generate_augmented_data(X_negative, Y_negative, num_additional_images)

# Combine the original and augmented data
X_train = np.concatenate([X_train, X_positive_augmented, X_negative_augmented], axis=0)
y_train = np.concatenate([y_train, Y_positive_augmented, Y_negative_augmented], axis=0)

X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

knn = KNeighborsClassifier()

grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, scoring='f1', verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_

y_pred = model.predict(X_test)

val_predicted_classes = (y_pred > 0.5).astype(int)
    
# Compute F1 score for this fold
f1 = f1_score(y_test, val_predicted_classes)

# Evaluate the model's performance
print(f"F1 score: {f1:.3f}")

joblib.dump(model, 'best_knn_model.joblib')