import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split

def z_score_normalizer(arr):
    return (arr - np.mean(arr)) / np.std(arr)

# Pre-trained CNN model (assume it's already trained)
def load_pretrained_cnn():
    # Example CNN for classification
    cnn_model = tf.keras.models.load_model('Model_CNN.h5')
    return cnn_model

# Use the encoder part of the autoencoder
def build_encoder(autoencoder):
    encoder = models.Model(inputs=autoencoder.input, outputs=autoencoder.layers[-4].output)  # Extract encoder part
    return encoder

# Create the combined model (encoder + pre-trained CNN)
def build_combined_model(encoder, cnn_model):
    # Freeze CNN layers if you want to fine-tune only some layers
    for layer in cnn_model.layers:
        layer.trainable = True

    input_img = layers.Input(shape=(28, 28, 1))  # Example shape for MNIST-like data
    encoded_features = encoder(input_img)  # Extract features using encoder
    predictions = cnn_model(encoded_features)  # Classify using the pre-trained CNN
    
    combined_model = models.Model(inputs=input_img, outputs=predictions)
    combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return combined_model

# Train the combined model
def train_combined_model(combined_model, x_train, y_train, x_test, y_test):
    combined_model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

def main():
    # Load pre-trained CNN
    cnn_model = load_pretrained_cnn()
    
    # Load pre-trained autoencoder (assuming it is already trained and saved)
    autoencoder = tf.keras.models.load_model('autoencoder.h5')
    
    # Build the encoder from the autoencoder
    encoder = build_encoder(autoencoder)
    
    # Build the combined model
    combined_model = build_combined_model(encoder, cnn_model)
    
    # Load your dataset (use encoded data)
    X = np.load('Xtrain1.npy')
    Y = np.load('Ytrain1.npy')
    
    X = z_score_normalizer(X)
    
    X = X.reshape(-1, 28, 28, 1)
    
    x_train, y_train, x_test, y_test = train_test_split(X, Y, test_size=0.2,shuffle=True, random_state=42)
    
    # Train the combined model
    train_combined_model(combined_model, x_train, y_train, x_test, y_test)

    combined_model.save('combined_model.h5')

if __name__ == "__main__":
    main()