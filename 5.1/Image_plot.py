import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
file_path = 'Xtrain1.npy'  # Replace with your actual file path
data = np.load(file_path)

# Number of images to plot
num_images = 10

# Reshape each row (1D array of 2304) into a 48x48 image
image_size = (48, 48)
images = data[:num_images].reshape(-1, *image_size)

# Plot the images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()  # Flatten the 2D array of axes for easier indexing
for i in range(num_images):
    axes[i].imshow(images[i], cmap='gray')
    axes[i].axis('off')  # Turn off axis for clarity
plt.show()
