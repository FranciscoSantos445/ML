import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import f1_score,accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2

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

#code with KFoldq , data augmentation and l2 regularization with grid search

# Load the data
X = np.load('Xtrain1.npy')
Y = np.load('Ytrain1.npy')
num_images = 1000 # Number of images to generate
image_size = 48 # Size of the images

X = z_score_normalizer(X)

X = X.reshape((-1, image_size, image_size, 1))

# Number of additional images per class to generate (for both positive and negative)
num_additional_images = 1000  # Change this number to generate more or less data

# Set up the ImageDataGenerator for augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,     # flip the image horizontally
    vertical_flip=True,       # flip the image vertically
    brightness_range=(0.9, 1.1)  # darken o brighten by 10%
)

# Variable to keep track of the best F1 score and the best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_f1_score = 0
best_model = None
l2_strengths = np.arange(0.001, 0.01, 0.002)

# Split the data into training and testing sets
for train_index, val_index in kf.split(X):
    # Split the data into train and validation sets for this fold
    X_train, X_test = X[train_index], X[val_index]
    y_train, y_test = Y[train_index], Y[val_index]
    
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
            tf.keras.layers.Dropout(0.6),  # Reset 60% of the network for each iteration
            tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (crater or no crater)
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        ################################# alterar valores para testar accuracy ##########################################
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 4)
        
        history = model.fit(X_train,y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])
        
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

print("f1_score",best_f1_score) 

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X, Y)

model.save('model_CNN2.h5')

plt.figure(figsize=(12, 6))

# Plotting the history of the best model
plt.figure(figsize=(12, 6))

# Plot training loss and validation loss over the epochs from the best model's history
plt.subplot(1, 2, 1)
plt.plot(best_history.history['loss'], label='Training Loss')
plt.plot(best_history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training accuracy and validation accuracy over the epochs
plt.subplot(1, 2, 2)
plt.plot(best_history.history['accuracy'], label='Training Accuracy')
plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()