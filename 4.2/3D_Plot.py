#Grupo 94
#Francisco Santos
#Antonio Quendera

# Code used to visualize the SSE for each combination of n, m, and d values in the ARX model, using a 3D plot
# There are values missing which represent the unstable models and the impossible represented by NaN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

def check_stability(model, n):
    """
    Check the stability of the ARX model by examining the poles (roots) of the characteristic equation.
    """
    # Extract AR coefficients from the model
    ar_coefficients = model.coef_[:n]
    
    # Form the polynomial
    characteristic_poly = np.concatenate(([1], ar_coefficients))
    
    # Find the roots
    poles = np.roots(characteristic_poly)
    
    # Check if any pole has an absolute value greater than 1
    if np.any(np.abs(poles) > 1):
        return False  # Unstable
    return True  # Stable

def sse(y_true, y_pred):
        
    return np.sum((y_true - y_pred) ** 2 )
    
def grid_search_arx_rigid(y, u):
    """
    Perform grid search over n, m, d values to find the best ARX model based on MSE.
    """
    
    best_MSE = float('inf')  # Initialize to a large value
    best_n, best_m, best_d = None, None, None
    best_model = None
    phi_best_test = None
    y_best_pred = None
    y_best_test = None 
    
    alphas_gen1 = np.arange(0.3, 0.6, 0.05)
    
    n_values = range(1, 10)  # Search over n from 1 to 9
    m_values = range(1, 10)  # Search over m from 1 to 9
    d_values = range(1, 10)  # Search over d from 1 to 9
    
    tscv = TimeSeriesSplit(n_splits=5)  # Need to use timeseries do to the nature of the data
    
    for n in n_values:
        for m in m_values:
            for d in d_values:
                # Build the regressor matrix for the current combination of n, m, d
                phi, y_out = build_regressor_matrix(y, u, n, m, d)
                
                if phi is not None:
                
                    # Perform cv using TimeSeriesSplit
                    for train_index, test_index in tscv.split(phi):
                        phi_train, phi_test = phi[train_index], phi[test_index]
                        y_train_out, y_test_out = y_out[train_index], y_out[test_index]
                        
                        model_rigid = RidgeCV(alphas=alphas_gen1, fit_intercept=False)
                        
                        model_rigid.fit(phi_train, y_train_out)
                        
                        if check_stability(model_rigid, n):
                        
                            # Predict on the test set
                            y_pred_rigid = model_rigid.predict(phi_test)
                            
                            # Calculate MSE
                            MSE_model = mean_squared_error(y_test_out, y_pred_rigid)
                            
                            # If this combination gives a better result, update the best parameters
                            if MSE_model < best_MSE:
                                best_MSE = MSE_model
                                best_n, best_m, best_d = n, m, d
                                y_best_pred = y_pred_rigid
                                best_model = model_rigid
                                y_best_test = y_test_out             
                                                
    return best_n, best_m, best_d, best_model, y_best_pred, y_best_test

def build_regressor_matrix(y, u, n, m, d):
    """
    Create the regressor matrix phi and output vector for the ARX model, so that a regression model can be applied.
    This is the step with Y = X*theta
    """
    # Number of samples
    N = len(y)
    
    # Determine the number of rows
    num_rows = N - max(n, m + d)
    
    if num_rows <= 0 or N <= max(n, m + d): # Impossible number of rows
        return None, None
    
    # Initializition
    phi = np.zeros((num_rows, n + m + 1))
    y_out = np.zeros(num_rows)
    
    for i in range(num_rows):
        
        if i + n > N or i + d + m + 1 > N: # Index out of bounds for its designated slices
            return None, None
        
        # Create phy slice of y values
        phi[i, :n] = -y[i:i + n][::-1]  # assign the y slice of phi
        
        # Create phi slice of u values
        phi[i, n:] = u[i + d:i + d + m + 1][::-1]  # assign the u slice of phi
        
        # y(k)
        y_out[i] = y[i + max(n, m + d)]
    
    return phi, y_out

# Example usage
y = np.load('output_train.npy')
u = np.load('u_train.npy')
u_test = np.load('u_test.npy')

# Range of m and d values to explore
m_values = range(1, 10)
d_values = range(1, 10)

fig = plt.figure(figsize=(18, 12)) 

n_values = [6, 7, 8, 9]
tscv = TimeSeriesSplit(n_splits=5)

for idx, n in enumerate(range(1,10)):
    
    sse_matrix = np.zeros((len(m_values), len(d_values)))  # Matrix to store SSE for each combination of m and d

    # Iterate through combinations of m and d
    for i, m in enumerate(m_values):
        for j, d in enumerate(d_values):
            try:
                # Build the regressor matrix for current n, m, d
                phi, y_out = build_regressor_matrix(y, u, n, m, d)
                
                # Perform TimeSeriesSplit and train the model
                for train_index, test_index in tscv.split(phi):
                    phi_train, phi_test = phi[train_index], phi[test_index]
                    y_train_out, y_test_out = y_out[train_index], y_out[test_index]
                    
                    # Train the model (you can switch between Linear and Ridge regression here)
                    model = LinearRegression(fit_intercept=False)
                    model.fit(phi_train, y_train_out)
                    
                    if check_stability(model, n) == True:
                        
                        # Predict the test set
                        y_pred = model.predict(phi_test)
                        
                        # Calculate SSE for this combination of m and d
                        sse_value = sse(y_test_out, y_pred)
                        
                        # Store SSE in the matrix
                        sse_matrix[i, j] = -sse_value
                        
                    else:
                        sse_matrix[i, j] = np.nan  # If the model is unstable, store NaN and continue
            
            except Exception as e:
                sse_matrix[i, j] = np.nan  # If there's an error, store NaN and continue

    # Create a meshgrid for m and d values
    M, D = np.meshgrid(m_values, d_values)
    
    # Add a subplot for the current n value
    ax = fig.add_subplot(3, 3, idx+1, projection='3d')
    
    # Plot the 3D surface for SSE vs m, d
    surf = ax.plot_surface(D, M, sse_matrix.T, cmap='viridis', edgecolor='none')

    # Add title and labels
    ax.set_title(f'SSE for n={n}')
    ax.set_xlabel('m values')
    ax.set_ylabel('d values')
    ax.set_zlabel('SSE')
    
    # Add a color bar for the surface plot
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Show the final plot with all 4 surfaces
plt.tight_layout()
plt.show()