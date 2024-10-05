import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

def sse(y_true, y_pred):
        
    return np.sum((y_true - y_pred) ** 2 )

def grid_search_arx_rigid(y, u):
   
    
    best_MSE = float('inf')  # Initialize to a large value
    best_n, best_m, best_d = None, None, None
    best_model = None
    y_best_pred = None
    y_best_test = None 
    
    alphas_gen1 = np.arange(0.3, 0.6, 0.05)
    
    n_values = range(1, 9)  # Search over n from 1 to 9
    m_values = range(1, 9)  # Search over m from 1 to 9
    d_values = range(1, 9)  # Search over d from 1 to 9
    
    tscv = TimeSeriesSplit(n_splits=5)  # Define 5 splits for time series
    
    for n in n_values:
        for m in m_values:
            for d in d_values:
                try:

                    # Build the regressor matrix for the current combination of n, m, d
                    phi, y_out = build_regressor_matrix(y, u, n, m, d)
                    
                    # Perform cross-validation using TimeSeriesSplit
                    for train_index, test_index in tscv.split(phi):
                        phi_train, phi_test = phi[train_index], phi[test_index]
                        y_train_out, y_test_out = y_out[train_index], y_out[test_index]
                        
                        model_rigid = RidgeCV(alphas=alphas_gen1, fit_intercept=False)
                        
                        # Train the model linear regression
                        model_rigid.fit(phi_train, y_train_out)
                        
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
                
                except Exception as e:
                    pass  # Skip this combination if it causes an error
    
    return best_n, best_m, best_d, best_model, y_best_test, y_best_pred

def grid_search_arx(y, u):
    """
    Perform grid search over n, m, d values to find the best ARX model based on MSE.

    Parameters:
    y (numpy array): Output sequence (time series data for y)
    u (numpy array): Input sequence (time series data for u)
    n_values (list): List of possible values for n (autoregressive order)
    m_values (list): List of possible values for m (exogenous input order)
    d_values (list): List of possible values for d (input time delay)

    Returns:
    best_n, best_m, best_d: The best values for n, m, and d based on MSE
    best_model: The model with the best parameters
    best_mse: The mean squared error of the best model
    y_best_pred: The predictions using the best model
    y_best_test: The test for the best model
    """
    
    best_MSE = float('inf')  # Initialize to a large value
    best_n, best_m, best_d = None, None, None
    best_model = None
    y_best_pred = None
    y_best_test = None 
        
    n_values = range(1, 9)  # Search over n from 1 to 9
    m_values = range(1, 9)  # Search over m from 1 to 9
    d_values = range(1, 9)  # Search over d from 1 to 9
    
    tscv = TimeSeriesSplit(n_splits=5)  # Define 5 splits for time series
    
    for n in n_values:
        for m in m_values:
            for d in d_values:
                try:

                    # Build the regressor matrix for the current combination of n, m, d
                    phi, y_out = build_regressor_matrix(y, u, n, m, d)
                    
                    # Perform cross-validation using TimeSeriesSplit
                    for train_index, test_index in tscv.split(phi):
                        phi_train, phi_test = phi[train_index], phi[test_index]
                        y_train_out, y_test_out = y_out[train_index], y_out[test_index]
                        
                        model = LinearRegression(fit_intercept=False)
                        
                        # Train the model linear regression
                        model.fit(phi_train, y_train_out)
                        
                        # Predict on the test set
                        y_pred = model.predict(phi_test)
                        
                        # Calculate MSE
                        MSE_model = mean_squared_error(y_test_out, y_pred)
                        
                        # If this combination gives a better result, update the best parameters
                        if MSE_model < best_MSE:
                            best_MSE = MSE_model
                            best_n, best_m, best_d = n, m, d
                            y_best_pred = y_pred
                            best_model = model
                            y_best_test = y_test_out
                
                except Exception as e:
                    pass  # Skip this combination if it causes an error
    
    return best_n, best_m, best_d, best_model, y_best_pred, y_best_test

def build_regressor_matrix(y, u, n, m, d):
    """
    Create the regressor matrix phi and output vector for the ARX model.

    Returns:
    phi (numpy array): phi(k)
    y_out (numpy array): y(k)
    """
    # Number of samples
    N = len(y)
    
    # Determine the number of rows
    num_rows = N - max(n, m + d)
    
    # Initializition
    phi = np.zeros((num_rows, n + m + 1))
    y_out = np.zeros(num_rows)
    
    for i in range(num_rows):
        # Create phy slice of y values
        phi[i, :n] = -y[i:i + n][::-1]  # assign the y slice of phi
        
        # Create phi slice of u values
        phi[i, n:] = u[i + d:i + d + m + 1][::-1]  # assign the u slice of phi
        
        # y(k)
        y_out[i] = y[i + max(n, m + d)]
    
    return phi, y_out

def generate_output_for_u_test(model, u_test, n, m, d):
    """
    Generate output for a given u_test input sequence where no output (y_test) is available.

    Parameters:
    model (sklearn model): Trained ARX model
    u_test (numpy array): Input sequence (test)
    n (int): Order of the autoregressive part (number of past y values)
    m (int): Order of the exogenous input part (number of past u values)
    d (int): Time delay for the input sequence
    y_initial (numpy array): Initial values of y for starting the prediction (optional)

    Returns:
    y_generated (numpy array): Predicted output for u_test
    """
    N = len(u_test)
    
    y_generated = np.zeros(N)
    
    # Iterate through the input data to predict the output step by step
    for k in range(max(n, m + d), N):
        # Build the regressor for the current step k
        phi_k = np.zeros(n + m + 1)
        
        # Add past y values
        phi_k[:n] = -y_generated[k-n:k][::-1]  # Autoregressive part
        
        # Corrected: Ensure we have the right slice for u_test, avoiding empty arrays
        u_slice_start = k - d - m
        u_slice_end = k - d
        
        if u_slice_start >= 0:
            # Use full slice if enough past values of u_test are available
            phi_k[n:] = u_test[u_slice_start:u_slice_end+1][::-1]
        else:
            # If not enough past values, use what is available and pad the rest with zeros
            available_u = u_test[0:u_slice_end+1][::-1]
            phi_k[n:n + len(available_u)] = available_u
        
        # Predict the next output using the model
        y_generated[k] = model.predict(phi_k.reshape(1, -1))
    
    return y_generated


# Example usage
# Generate some example data for y (output) and u (input)
# This is just an example. In practice, you would have actual time series data.
y = np.load('output_train.npy')
u = np.load('u_train.npy')
u_test = np.load('u_test.npy')

# Perform grid search to find the best ARX model
n,m,d,model,y_pred,y_test_out = grid_search_arx(y, u)

n2,m2,d2,model_rigid,y_pred_rigid,y_test_out2 = grid_search_arx_rigid(y, u)

if sse(y_test_out, y_pred) < sse(y_test_out2, y_pred_rigid):
    # Generate the output for u_test
    y_generated = generate_output_for_u_test(model, u_test, n, m, d)

    y_output = y_generated[-400:]
else:
    # Generate the output for u_test
    y_generated = generate_output_for_u_test(model_rigid, u_test, n, m, d)

    y_output = y_generated[-400:]


n_values = [6, 7, 8, 9]
m_values = [1,2,3,4,5,6, 7, 8, 9]
d_values = [1,2,3,4,5,6, 7, 8, 9]

fig = plt.figure(figsize=(18, 12))

for idx, n in enumerate(n_values):
    sse_matrix = np.zeros((len(m_values), len(d_values)))  # Matrix to store SSE for each combination of m and d

    # Iterate through combinations of m and d
    for i, m in enumerate(m_values):
        for j, d in enumerate(d_values):
            try:
                # Build the regressor matrix for current n, m, d
                phi, y_out = build_regressor_matrix(y, u, n, m, d)
                
                # Perform TimeSeriesSplit and train the model
                tscv = TimeSeriesSplit(n_splits=5)
                for train_index, test_index in tscv.split(phi):
                    phi_train, phi_test = phi[train_index], phi[test_index]
                    y_train_out, y_test_out = y_out[train_index], y_out[test_index]
                    
                    # Train the model (you can switch between Linear and Ridge regression here)
                    model = LinearRegression(fit_intercept=False)
                    model.fit(phi_train, y_train_out)
                    
                    # Predict the test set
                    y_pred = model.predict(phi_test)
                    
                    # Calculate SSE for this combination of m and d
                    sse_value = sse(y_test_out, y_pred)
                    
                    # Store SSE in the matrix
                    sse_matrix[i, j] = sse_value
            
            except Exception as e:
                sse_matrix[i, j] = np.nan  # If there's an error, store NaN and continue

    # Create a meshgrid for m and d values
    M, D = np.meshgrid(m_values, d_values)
    
    # Add a subplot for the current n value
    ax = fig.add_subplot(2, 2, idx+1, projection='3d')
    
    # Plot the 3D surface for SSE vs m, d
    surf = ax.plot_surface(M, D, sse_matrix.T, cmap='viridis', edgecolor='none')

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