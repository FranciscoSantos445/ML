import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from tensorflow.keras.regularizers import l2

def z_score_normalizer(arr):
    return (arr - np.mean(arr)) / np.std(arr)

#code with KFoldq , SMOTE and l2 regularization

# Load the data
X = np.load('Xtrain1.npy')
Y = np.load('Ytrain1.npy')


# Variable to keep track of the best F1 score and the best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_f1_score = 0
best_model = None

# Normalize the data (optional but recommended for CNNs)
#X = X.astype('float32') / 255.0
X = z_score_normalizer(X)

# Reshape the data into 48x48 images
image_size = 48
X = X.reshape(-1, image_size, image_size, 1)  # Adding a channel dimension for grayscale

# Split the data into training and testing sets
for train_index, val_index in kf.split(X):
    # Split the data into train and validation sets for this fold
    X_train, X_test = X[train_index], X[val_index]
    y_train, y_test = Y[train_index], Y[val_index]

    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten the images for SMOTE

    smote = SMOTE()
    
    X_train, y_train = smote.fit_resample(X_train_flat, y_train)
    
    X_train = X_train.reshape(-1, image_size, image_size, 1)

    # Define the CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(48, 48, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.Dropout(0.5),  # Dropout with 50% rate
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (crater or no crater)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Make predictions on the validation set of the current fold
    val_predictions = model.predict(X_test)
    val_predicted_classes = (val_predictions > 0.5).astype(int)

    # Compute F1 score for this fold
    f1 = f1_score(y_test, val_predicted_classes)
    
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model = model  # Keep the best model
        
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

plt.tight_layout()
plt.show()