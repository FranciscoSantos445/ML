import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# Load data
data_x = np.load('X_train.npy')
data_y = np.load('y_train.npy')

# Normalize the target variable (y)
scaler_y = MinMaxScaler()
data_y = data_y.reshape(-1, 1)
y = scaler_y.fit_transform(data_y)

# Normalize the features (X)
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(data_x[:, :5])

# Define the RANSAC estimator
ransac = RANSACRegressor(stop_n_inliers= int( np.shape(X)[0] * 0.75)).fit(X, y)

# Get inliers and outliers from the training data
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

X = X[inlier_mask]
y = y[inlier_mask]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# Define the RANSAC estimator
ransac = RANSACRegressor().fit(X_train, y_train)

y_ransac = ransac.predict(X_test)

print(ransac.score(X, y))

# Invert the normalization for predictions and actual values
y_ransac = scaler_y.inverse_transform(y_ransac.reshape(-1, 1))
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Plot the test set results and predictions
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='True y')
plt.scatter(range(len(y_ransac)), y_ransac, color='red', label='Predicted y')
plt.xlabel('Sample index')
plt.ylabel('y value')
plt.legend()
plt.title('RANSAC Regression on Test Set')
plt.show()
