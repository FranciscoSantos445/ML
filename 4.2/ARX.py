import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def create_arx_regressors(u, y, n, m, d):
    N = len(y)
    p = max(n, d + m)
    X = []
    Y = []
    
    # Create the regressor matrix X and the output vector Y
    for k in range(p, N):
        # Lagged values of y
        y_lags = [y[k-i] for i in range(1, n+1)]
        # Lagged values of u
        u_lags = [u[k-d-i] for i in range(m+1)]
        # Combine lags into one feature vector
        phi_k = y_lags + u_lags
        X.append(phi_k)
        Y.append(y[k])
    
    return np.array(X), np.array(Y)

def evaluate_arx_model_sse(n, m, d, u_train, y_train, u_test, y_test):
    # Create ARX regressor matrices
    X_train, Y_train = create_arx_regressors(u_train, y_train, n, m, d)
    X_test, Y_test = create_arx_regressors(u_test, y_test, n, m, d)
    
    # Train Kernel Ridge Regression model with RBF kernel
    rbf_regressor = KernelRidge(kernel='rbf', gamma=0.1, alpha=1.0)
    rbf_regressor.fit(X_train, Y_train)
    
    # Predict on the test set
    y_pred = rbf_regressor.predict(X_test)
    
    # Compute the SSE
    sse = np.sum((Y_test - y_pred) ** 2)
    return sse


def best_paramaters(u_train, y_train, u_test, y_test):
    n_values = range(1, 10)  # Search over n from 1 to 9
    m_values = range(1, 10)  # Search over m from 1 to 9
    d_values = range(1, 10)  # Search over d from 1 to 9
    for n in n_values:
        for m in m_values:
            for d in d_values:
                try:
                    # Evaluate the ARX model for this combination of n, m, and d using SSE
                    sse = evaluate_arx_model_sse(n, m, d, u_train, y_train, u_test, y_test)
                    
                    # Check if this is the best SSE we've found so far
                    if sse < best_sse:
                        best_sse = sse
                        best_params = (n, m, d)
                        
                    print(f"n={n}, m={m}, d={d}, SSE={sse}")
                except Exception as e:
                    # Handle cases where a combination doesn't work
                    print(f"Error with n={n}, m={m}, d={d}: {e}")
    
    return best_params

def main():
    
    data_x = np.load('u_train.npy')
    data_y = np.load('output_train.npy')
    data_x_test = np.load('u_test.npy')

    ##################################### Splitting and best parameters #####################################   
    
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size= 0.3, random_state=None, shuffle=True)

    ##################################### Best paramaters #####################################

    best_params = best_paramaters(X_train, y_train, X_test, y_test)

    n,m,d = best_params
 
    ##################################### model prediction #####################################


    # Create regressor matrix for training data
    X_train, Y_train = create_arx_regressors(X_train, y_train, n, m, d)

    # Define the radial basis function (RBF) kernel ridge regression model
    rbf_regressor = KernelRidge(kernel='rbf', gamma=0.1, alpha=1.0)

    # Train the model
    rbf_regressor.fit(X_train, Y_train)

    # Predict on test data (iteratively)
    y_test_pred = []
    y_test_actual = np.zeros(len(y_test))

    # Initialize the first `n` values of y_test_pred with zero (or use provided initial conditions)
    y_test_pred = list(y_train[-n:])  # Start with the last n values from training set

    # Iteratively predict y_test
    for k in range(len(X_test)):
        # Create the regressor vector for the current step
        y_lags = y_test_pred[-n:]
        u_lags = [X_test[k - d - i] for i in range(m+1)]
        
        phi_k = np.array(y_lags + u_lags).reshape(1, -1)
        
        # Predict the next value
        y_next = rbf_regressor.predict(phi_k)
        
        y_test_pred.append(y_next[0])  # Append the prediction to the list

    # Take the last 400 samples of the predicted test output (as per the problem)
    y_test_pred_final = np.array(y_test_pred[-400:])

    # # Save the final prediction to submit
    np.save('y_test_pred.npy', y_test_pred_final)

    # For evaluation (if actual test output y_test is available):
    mse = mean_squared_error(y_test[-400:], y_test_pred_final)
    print(f'Mean Squared Error on test set: {mse}')
    
    plt.figure(figsize=(10, 6))
    
    # plt.scatter(range(len(y_test)),y_test, color='red', label='y')
    # plt.scatter(range(len(data_y)), data_y, color='blue', label='y')
    # plt.scatter(range(len(data_x_test)), data_x_test, color='yellow', label='X_teste_file')

    plt.title('Noggas')
    plt.xlabel('Index')
    plt.ylabel('Nuck figgers')

    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    
    main()