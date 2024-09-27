import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load your data and normalize it
data_x = np.load('X_train.npy')
data_y = np.load('y_train.npy')

scaler_y = MinMaxScaler()
data_y = data_y.reshape(-1, 1)
y = scaler_y.fit_transform(data_y)

scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(data_x[:, :5])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# Define the RANSAC estimator
ransac = RANSACRegressor()

residual = np.arange(0.1, 100, 0.1)

# Define the parameter grid for GridSearch
param_grid = {
    'residual_threshold': residual,  # Try different thresholds
    'max_trials': [5000],  # Different numbers of trials
    'stop_n_inliers': [int( np.shape(X_train)[0] * 0.75) ],  # Test without or with stopping conditions
}

# Set up the GridSearchCV
grid_search = GridSearchCV(ransac, param_grid, cv=5, scoring= 'r2', n_jobs=-1)

# Fit GridSearch to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and estimator
print("Best parameters found: ", grid_search.best_params_)
best_ransac = grid_search.best_estimator_

# Predict using the best RANSAC model
Y_ransac = best_ransac.predict(X_test)

# Print the best score achieved during cross-validation
print(best_ransac.score(X_test, y_test))
