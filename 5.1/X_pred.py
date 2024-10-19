import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def z_score_normalizer(arr):
    return (arr - np.mean(arr)) / np.std(arr)

# Load the trained model (if needed) or use the existing trained model
# If you have saved the model, load it like this:
model = tf.keras.models.load_model('Model_CNN.h5')

model_trained = tf.keras.models.load_model('Model_CNN_updated.h5')

# Load the new data
new_data = np.load('Xtest1.npy')

# Normalize the new data (same as before)
new_data = z_score_normalizer(new_data)

# Reshape the new data into 48x48 images with 1 channel (grayscale)
image_size = 48
new_data = new_data.reshape(-1, image_size, image_size, 1)

# Make predictions on the new data
new_predictions = model.predict(new_data)

new_predictions_trained = model_trained.predict(new_data)

# Convert the predicted probabilities to class labels (0 or 1)
new_predicted_classes = (new_predictions > 0.5).astype(int)

new_predicted_classes_trained = (new_predictions_trained > 0.5).astype(int)

# Compare the new_predicted_classes with new_predicted_classes_trained
comparison = new_predicted_classes == new_predicted_classes_trained
comparison_result = np.all(comparison)

print("Are the predictions from both models identical? ", comparison_result)

if not comparison_result:
    differing_indices = np.where(comparison == False)[0]
    print("Indices where predictions differ: ", differing_indices)
    
    # Plot the images where predictions differ
    if not comparison_result:
        fig, axes = plt.subplots(4, 5, figsize=(12, 6))
        axes = axes.flatten()
        for idx, ax in zip(differing_indices[:20], axes):
            ax.imshow(new_data[idx].squeeze(), cmap='gray')
            ax.set_title(f'Pred: {new_predicted_classes[idx][0]}, Trained: {new_predicted_classes_trained[idx][0]}')
            ax.axis('off')
        plt.tight_layout()
        plt.show()

#np.save ('Ytest1.npy', new_predicted_classes)

zeros = new_predicted_classes[new_predicted_classes == 0]
ones = new_predicted_classes[new_predicted_classes == 1]

print("Destribuition of zeros: ", (len(zeros) / new_predicted_classes.shape[0])*100 )
print ("Destribuition of ones: ", (len(ones) / new_predicted_classes.shape[0])*100 )