import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV,ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def sse(y_true, y_pred):
        
    return np.sum((y_true - y_pred) ** 2 )

def grid_search_arx(y, u, model):
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
                    
                    # Train the model linear regression
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
    
    return best_n, best_m, best_d, best_model, y_best_pred, y_best_test, best_sse

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

alphas_gen1 = np.arange(0.1, 100, 0.1)

alphas_gen2 = np.arange(0.0001, 0.01, 0.0005)

l1 = np.arange(0.1,1,0.1)


model_linear = LinearRegression(fit_intercept=False)

model_rigid = RidgeCV(alphas = alphas_gen1, fit_intercept=False)

model_lasso = LassoCV(alphas = alphas_gen2, fit_intercept=False)

model_Elastic = ElasticNetCV(alphas = None, l1_ratio = l1, fit_intercept=False, cv = 5, max_iter = 10000, tol=1e-4)

n = np.zeros(4)
m = np.zeros(4)
d = np.zeros(4)
y_pred_vectors = []
best_sse = np.zeros(4)
# Perform grid search to find the best ARX model
n[0],m[0],d[0],model_linear,y_pred,y_test_out, best_sse[0] = grid_search_arx(y, u, model_linear)
y_pred_vectors.append(y_pred)
print("PAST LINEAR")
n[1],m[1],d[1],model_rigid,y_pred,y_test_out, best_sse[1] = grid_search_arx(y, u,model_rigid)
y_pred_vectors.append(y_pred)
print("PAST RIGID")
n[2],m[2],d[2],model_lasso,y_pred,y_test_out, best_sse[2] = grid_search_arx(y, u,model_lasso)
y_pred_vectors.append(y_pred)
print("PAST LASSO")
n[3],m[3],d[3],model_Elastic,y_pred,y_test_out, best_sse[3] = grid_search_arx(y, u,model_Elastic)
y_pred_vectors.append(y_pred)

y_pred_vectors = np.array(y_pred_vectors)

print(" n , m, d linear : ",n[0],m[0],d[0], best_sse[0])
print(" n , m, d rigid : ",n[1],m[1],d[1],best_sse[1])
print(" n , m, d lasso : ",n[2],m[2],d[2],best_sse[2])
print(" n , m, d li : ",n[3],m[3],d[3],best_sse[3])
