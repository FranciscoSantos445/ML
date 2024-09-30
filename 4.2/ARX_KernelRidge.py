import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import itertools

def grid_search_arx(y, u):
    """
    Perform grid search over n, m, d values to find the best ARX model based on MSE.

    Parameters:
    y (numpy array): Output sequence (time series data for y)
    u (numpy array): Input sequence (time series data for u)
    n_values (list): List of possible values for n (autoregressive order)
    m_values (list): List of possible values for m (exogenous input order)
    d_values (list): List of possible values for d (input time delay)

<<<<<<< HEAD:4.2/ARX.py
    Returns:
    best_n, best_m, best_d: The best values for n, m, and d based on MSE
    best_model: The trained model with the best parameters
    best_mse: The mean squared error of the best model
    """
    best_mse = float('inf')  # Initialize to a large value
    best_n, best_m, best_d = None, None, None
    best_model = None
    
    n_values = range(1, 9)  # Search over n from 1 to 9
    m_values = range(1, 9)  # Search over m from 1 to 9
    d_values = range(1, 9)  # Search over d from 1 to 9
    for n in n_values:
        for m in m_values:
            for d in d_values:
                try:

                    # Build the regressor matrix for the current combination of n, m, d
                    phi, y_out = build_regressor_matrix(y, u, n, m, d)
                    
                    # Split into training and testing sets
                    phi_train, phi_test, y_train_out, y_test_out = train_test_split(phi, y_out, test_size=0.3, random_state=42)
                    
                    # Train the model
                    model = LinearRegression(fit_intercept=False)
                    model.fit(phi_train, y_train_out)
                    
                    # Predict on the test set
                    y_pred = model.predict(phi_test)
                    
                    # Calculate mean squared error
                    mse = mean_squared_error(y_test_out, y_pred)
                    
                    # If this combination gives a better result, update the best parameters
                    if mse < best_mse:
                        best_mse = mse
                        best_n, best_m, best_d = n, m, d
                        best_model = model
                        y_best_pred = y_pred
                        y_best_test = y_test_out
                        
                except Exception as e:
                    pass # Skip this combination if it causes an error
    
    return best_n, best_m, best_d, best_model, y_best_pred, y_best_test
=======

# def best_paramaters(u_train, y_train, u_test, y_test):
#     n_values = range(1, 10)  # Search over n from 1 to 9
#     m_values = range(1, 10)  # Search over m from 1 to 9
#     d_values = range(1, 10)  # Search over d from 1 to 9
#     best_sse = 500000
#     best_params=(0,0,0)
#     for n in n_values:
#         for m in m_values:
#             for d in d_values:
#                 try:
#                     # Evaluate the ARX model for this combination of n, m, and d using SSE
#                     sse = evaluate_arx_model_sse(n, m, d, u_train, y_train, u_test, y_test)
                    
#                     # Check if this is the best SSE we've found so far
#                     if sse < best_sse:
#                         best_sse = sse
#                         best_params = (n, m, d)
                        
#                     print(f"n={n}, m={m}, d={d}, SSE={sse}, Best_params={best_params}")
                    
#                 except Exception as e:
#                     # Handle cases where a combination doesn't work
#                     print(f"Error with n={n}, m={m}, d={d}: {e}")
    
#     print("Best params: ", best_params)
#     return best_params
>>>>>>> c0a4787bce8ea8835c277982f80c58c0db13e24f:4.2/ARX_KernelRidge.py

# Function to create the regressor matrix phi(k) and the corresponding output vector y(k)
def build_regressor_matrix(y, u, n, m, d):
    """
    Create the regressor matrix phi and output vector for the ARX model.

    Parameters:
    y (numpy array): Output sequence (time series data for y)
    u (numpy array): Input sequence (time series data for u)
    n (int): Order of the autoregressive part (number of past y values)
    m (int): Order of the exogenous input part (number of past u values)
    d (int): Time delay for the input sequence

    Returns:
    phi (numpy array): Regressor matrix
    y_out (numpy array): Output vector (corresponding to y(k))
    """
    # Number of samples
    N = len(y)
    
    # Determine the number of rows in the regressor matrix
    num_rows = N - max(n, m + d)
    
<<<<<<< HEAD:4.2/ARX.py
    # Initialize the regressor matrix and output vector
    phi = np.zeros((num_rows, n + m + 1))
    y_out = np.zeros(num_rows)
=======
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size= 0.3, random_state=None, shuffle=True)

    ##################################### Best paramaters #####################################

    n,m,d = 4,5,3
 
    ##################################### model prediction #####################################


     # Create ARX regressor matrices
    X_train, Y_train = create_arx_regressors(X_train, y_train, n, m, d)
    X_test, Y_test = create_arx_regressors(X_test, y_test, n, m, d)
    
    # Train Kernel Ridge Regression model with RBF kernel
    rbf_regressor = KernelRidge(kernel='rbf', gamma=0.1, alpha=1.0)
    rbf_regressor.fit(X_train, Y_train)
    
    print("Dimension X_test: ", np.shape(X_test))

    # Predict on the test set
    y_pred = rbf_regressor.predict(X_test)

    print("Dimension y_test: ", np.shape(Y_test))
    print("Dimension y_pred: ", np.shape(y_pred))

    # Predict on test data (iteratively)
    y_test_pred = []
    y_test_actual = np.zeros(len(y_test))
>>>>>>> c0a4787bce8ea8835c277982f80c58c0db13e24f:4.2/ARX_KernelRidge.py
    
    # Populate the regressor matrix and output vector
    for i in range(num_rows):
        # Create the autoregressive part (past y values)
        phi[i, :n] = -y[i:i + n][::-1]  # Reverse the order of y terms
        
        # Create the exogenous input part (past u values)
        phi[i, n:] = u[i + d:i + d + m + 1][::-1]  # Reverse the order of u terms
        
        # Output vector
        y_out[i] = y[i + max(n, m + d)]
    
    return phi, y_out

def generate_output_for_u_test(model, u_test, n, m, d, y_initial=None):
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
    
    # Initialize the output array with zeros or provided initial y values
    if y_initial is None:
        y_generated = np.zeros(N)
    else:
        y_generated = np.concatenate([y_initial, np.zeros(N - len(y_initial))])
    
    # Iterate through the input data to predict the output step by step
    for k in range(max(n, m + d), N):
        # Build the regressor for the current step k
        phi_k = np.zeros(n + m + 1)
        
<<<<<<< HEAD:4.2/ARX.py
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
=======
        y_test_pred.append(y_next[0])  # Append the prediction to the list

    # Take the last 400 samples of the predicted test output (as per the problem)
    y_test_pred_final = np.array(y_test_pred[-400:])

    # # Save the final prediction to submit
    np.save('y_test_pred.npy', y_test_pred_final)

    # For evaluation (if actual test output y_test is available):
    mse = mean_squared_error(Y_test, y_pred)

    print(f'Mean Squared Error on test set: {mse}')
>>>>>>> c0a4787bce8ea8835c277982f80c58c0db13e24f:4.2/ARX_KernelRidge.py
    
    return y_generated


# Example usage
# Generate some example data for y (output) and u (input)
# This is just an example. In practice, you would have actual time series data.
y = np.load('output_train.npy')
u = np.load('u_train.npy')
u_test = np.load('u_test.npy')

# Perform grid search to find the best ARX model
n,m,d,model,y_pred,y_test_out = grid_search_arx(y, u)

# # Retrieve the estimated parameters (theta)
# theta_estimated = model.coef_

# Optional: Use initial values of y for the first few steps (can use y_train_out[-n:] for example)
# y_initial = y_train_out[-n:]  # The last n values of y from training data

# Generate the output for u_test
y_generated = generate_output_for_u_test(model, u_test, n, m, d)

# Plot the actual and predicted output
plt.figure(figsize=(12, 6))

# Plot actual output
plt.subplot(2, 1, 1)
plt.plot(range(len(y_test_out)), y_test_out, label='Actual Output', color='blue')
plt.plot(range(len(y_pred)), y_pred, label='Predicted Output', linestyle='--', color='red')
plt.title('ARX Model: Actual vs Predicted Output')
plt.xlabel('Time step (k)')
plt.ylabel('Output y(k)')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(range(len(y_generated)), y_generated, label='Generated Output (y)', color='blue')
plt.title('Generated Output for u_test Using ARX Model')
plt.xlabel('Time step (k)')
plt.ylabel('Output y(k)')
plt.legend()
plt.grid()

plt.show()