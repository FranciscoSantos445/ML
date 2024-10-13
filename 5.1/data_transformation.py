import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from tensorflow.keras.regularizers import l2

def z_score_normalizer(arr):
    return (arr - np.mean(arr)) / np.std(arr)

#code with KFoldq , SMOTE and l2 regularization with grid search

# Load the data
X = np.load('Xtrain1.npy')
Y = np.load('Ytrain1.npy')

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range= 5,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=(0.9, 1.0),
    brightness_range=(0.9, 1.1)
)

# Generate additional images using the ImageDataGenerator
augmented_images = []
augmented_labels = []

batch_size = 32
augment_batches = 10  # How many batches of augmented data you want to generate

for i in range(augment_batches):
    for X_batch, y_batch in datagen.flow(X, Y, batch_size=batch_size):
        augmented_images.append(X_batch)
        augmented_labels.append(y_batch)
        if len(augmented_images) >= augment_batches * batch_size:
            break

# Concatenate the augmented images and labels with the original dataset
augmented_images = np.concatenate(augmented_images, axis=0)
augmented_labels = np.concatenate(augmented_labels, axis=0)

# Combine the original and augmented data
X = np.concatenate([X, augmented_images], axis=0)
Y = np.concatenate([Y, augmented_labels], axis=0)

# Variable to keep track of the best F1 score and the best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_f1_score = 0
best_model = None
l2_strengths = np.arange(0.001, 0.01, 0.001)

# Normalize the data using Z-score normalization
X = z_score_normalizer(X)

# Reshape the data into 48x48 images
image_size = 48
X = X.reshape(-1, image_size, image_size, 1)  # Adding a channel dimension for grayscale

# Split the data into training and testing sets
for train_index, val_index in kf.split(X):
    # Split the data into train and validation sets for this fold
    X_train, X_test = X[train_index], X[val_index]
    y_train, y_test = Y[train_index], Y[val_index]

    # X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten the images for SMOTE

    # smote = SMOTE()  approch to use SMOTE
    
    # X_train, y_train = smote.fit_resample(X_train_flat, y_train)
    
    # X_train = X_train.reshape(-1, image_size, image_size, 1)

    for l2_strength in l2_strengths:
        
        # Define the CNN model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(l2_strength), input_shape=(48, 48, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(l2_strength)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(l2_strength)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Flatten(),
            
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(l2_strength)),
            tf.keras.layers.Dropout(0.6),  # Dropout with 50% rate
            tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (crater or no crater)
        ])

        # Compile the model
        history = model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        ################################# alterar valores para testar accuracy ##########################################
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        
        model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])
        
        # Make predictions on the validation set of the current fold
        val_predictions = model.predict(X_test)
        val_predicted_classes = (val_predictions > 0.5).astype(int)

        # Compute F1 score for this fold
        f1 = f1_score(y_test, val_predicted_classes)
        
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model = model  # Keep the best model
            best_history = history
        
model = best_model # Use the best model

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

# Plot the loss function along the epochs of the best model
plt.figure()
plt.plot(best_history.history['loss'], label='Training Loss')
plt.title('Loss Function Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()