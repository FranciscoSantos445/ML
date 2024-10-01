import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import itertools

def sse(y_true, y_pred):
        
    return np.sum((y_true - y_pred) ** 2 )

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
    
    best_sse = float('inf')  # Initialize to a large value
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
                    sse_model = sse(y_test_out, y_pred)
                    
                    # If this combination gives a better result, update the best parameters
                    if sse_model < best_sse:
                        best_sse = sse_model
                        best_n, best_m, best_d = n, m, d
                        best_model = model
                        y_best_pred = y_pred
                        y_best_test = y_test_out
                        
                except Exception as e:
                    pass # Skip this combination if it causes an error
    
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

# Perform grid search to find the best ARX model
n,m,d,model,y_pred,y_test_out = grid_search_arx(y, u)

y_initial = y[-n:]  # Use the last best_n values of y for initialization

# Generate the output for u_test
y_generated = generate_output_for_u_test(model, u_test, n, m, d, y_initial)

#print best model parameters
print( n, m, d, "\n")

print( "SSE: ", sse(y_test_out, y_pred), "\n")

# Plot the actual and predicted output
plt.figure(figsize=(12, 6))

# Plot actual output
plt.subplot(2, 1, 1)
plt.plot(range(len(y_test_out)), y_test_out, label='Actual Output', color='blue')
plt.plot(range(len(y_pred)), y_pred, label='Predicted Output',linestyle='--', color='red')
# plt.scatter(range(len(y_test_out)), y_test_out, label='Actual Output', color='blue')
# plt.scatter(range(len(y_pred)), y_pred, label='Predicted Output', color='red')
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

plt.tight_layout()  # Automatically adjust spacing
plt.show()