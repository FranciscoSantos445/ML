import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

# Function to create ARX regressors
def create_arx_regressors(u, y, n, m, d):
    N = len(y)
    p = max(n, d + m)
    X = []
    Y = []
    
    for k in range(p, N):
        # Lagged values of y
        y_lags = [y[k - i] for i in range(1, n + 1)]
        # Lagged values of u
        u_lags = [u[k - d - i] for i in range(m + 1)]
        # Combine lags into one feature vector
        phi_k = y_lags + u_lags
        X.append(phi_k)
        Y.append(y[k])
    
    return np.array(X), np.array(Y)

def main():
    # Load training data
    data_x = np.load('u_train.npy')  # Input data
    data_x_test = np.load('u_test.npy')  # Input data
    data_y = np.load('output_train.npy')  # Output data
    
    ##################################### Plot Results #####################################
    plt.figure(figsize=(10, 6))
    
    plt.scatter(range(len(data_x)), data_x, color='red', label='Data_x')
    plt.scatter(range(len(data_y)), data_y, color='blue', label='Data_y')
    start_index = 2049
    plt.scatter(range(start_index, start_index + len(data_x_test)), data_x_test, color='green', label='Data_x_test')

    
    plt.title('Predictions vs Actuals')
    plt.xlabel('Index')
    plt.ylabel('Output Value')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
