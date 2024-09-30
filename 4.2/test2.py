import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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
    
    # Initialize the regressor matrix and output vector
    phi = np.zeros((num_rows, n + m + 1))
    y_out = np.zeros(num_rows)
    
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

# ARX model parameters
n = 2  # Order of autoregressive part (number of past y values)
m = 2  # Order of exogenous part (number of past u values)
d = 1  # Time delay for the input sequence

# Build the regressor matrix and output vector
phi, y_out = build_regressor_matrix(y, u, n, m, d)

phi_train, phi_test, y_train_out, y_test_out = train_test_split(phi, y_out, test_size=0.3)

# Fit the linear regression model to estimate theta (ARX parameters)
model = LinearRegression(fit_intercept=False)  # No intercept needed since it's handled in phi
model.fit(phi_train, y_train_out)

y_pred = model.predict(phi_test)

y_test = model.predict

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