#Grupo 94
#Francisco Santos
#Antonio Quendera

# Code used to filter best two model based on MSE for the ARX model 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV,LassoCV,ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

def sse(y_true, y_pred):
        
    return np.sum((y_true - y_pred) ** 2 )

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

def grid_search_arx(y, u, model):
    """
    Perform grid search over n, m, d values to find the best ARX model based on MSE.
    """
    
    best_MSE = float('inf')  # Initialize to a large value
    best_n, best_m, best_d = None, None, None
    best_model = None
    y_best_pred = None
    y_best_test = None 
        
    n_values = range(1, 10)  # Search over n from 1 to 9
    m_values = range(1, 10)  # Search over m from 1 to 9
    d_values = range(1, 10)  # Search over d from 1 to 9
    
    tscv = TimeSeriesSplit(n_splits=5)  # Define 5 splits for time series
    
    for n in n_values:
        for m in m_values:
            for d in d_values:
                # Build the regressor matrix for the current combination of n, m, d
                phi, y_out = build_regressor_matrix(y, u, n, m, d)
                
                if phi is not None:
                    
                    # Perform cross-validation using TimeSeriesSplit
                    for train_index, test_index in tscv.split(phi):
                        phi_train, phi_test = phi[train_index], phi[test_index]
                        y_train_out, y_test_out = y_out[train_index], y_out[test_index]
                        
                        # Train the model linear regression
                        model.fit(phi_train, y_train_out)
                        
                        if check_stability(model, n) == True:
                        
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

def generate_output_for_u_test(model, u_test, n, m, d):
    """
    Generate output for a given u_test input sequence where no output (y_test) is available.
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

# Load the data
y = np.load('output_train.npy')
u = np.load('u_train.npy')
u_test = np.load('u_test.npy')

alphas_gen1 = np.arange(0.3, 0.6, 0.05)

alphas_gen2 = np.arange(0.0001, 0.005, 0.0005)

l1 = np.arange(0.1,1,0.1)

model_linear = LinearRegression(fit_intercept=False)

model_rigid = RidgeCV(alphas = alphas_gen1, fit_intercept=False)

model_lasso = LassoCV(alphas = alphas_gen2, fit_intercept=False)

model_Elastic = ElasticNetCV(alphas = None, l1_ratio = l1, fit_intercept=False)

n = np.zeros(4)
m = np.zeros(4)
d = np.zeros(4)
y_pred_vectors = []
best_sse = np.zeros(4)

# Perform grid search to find the best ARX model
n[0],m[0],d[0],model_linear,y_pred,y_test_out, best_sse[0] = grid_search_arx(y, u, model_linear)
y_pred_vectors.append(y_pred)

print("PAST LINEAR \n")

n[1],m[1],d[1],model_rigid,y_pred,y_test_out, best_sse[1] = grid_search_arx(y, u,model_rigid)
y_pred_vectors.append(y_pred)

print("PAST RIGID \n")

n[2],m[2],d[2],model_lasso,y_pred,y_test_out, best_sse[2] = grid_search_arx(y, u,model_lasso)
y_pred_vectors.append(y_pred)

print("PAST LASSO \n")

n[3],m[3],d[3],model_Elastic,y_pred,y_test_out, best_sse[3] = grid_search_arx(y, u,model_Elastic)
y_pred_vectors.append(y_pred)

y_pred_vectors = np.array(y_pred_vectors)

print(" n , m, d linear : ",n[0],m[0],d[0], best_sse[0])
print(" n , m, d rigid : ",n[1],m[1],d[1],best_sse[1])
print(" n , m, d lasso : ",n[2],m[2],d[2],best_sse[2])
print(" n , m, d li : ",n[3],m[3],d[3],best_sse[3])