import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def outliers(data):
    # Compute quartiles
    Q1 = np.percentile(data, 25)
    Q2 = np.median(data)  # or np.percentile(data, 50)
    Q3 = np.percentile(data, 75)

    # Compute interquartile range (IQR)
    IQR = Q3 - Q1

    # Compute whiskers
    lower_whisker = np.min(data[data >= Q1 - 1.5 * IQR])
    upper_whisker = np.max(data[data <= Q3 + 1.5 * IQR])

    # Find outliers
    outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
    
    return outliers

def Coefficient_of_Determination(y,y_pred):

    # Square the residuals
    squared_residuals = (y - y_pred) ** 2
    
    y_mean = np.mean(y)

    # Sum the squared residuals to get the sum of squared errors (SSE)
    SSEres = np.sum(squared_residuals)
    
    # Compute the total sum of squares
    SStotal = np.sum((y - y_mean) ** 2)
    
    r2 = 1 - (SSEres/SStotal)
    
    return r2

def linear_regression(X, y):
    """
    Performs linear regression on the input data X and target y.
    """
    # Linear regression formula: (X^T X)^(-1) X^T y
    beta = np.dot( np.dot(np.linalg.inv( np.dot(np.transpose(X),X) ), np.transpose(X)), y)  # (XT * X)** -1 * XT * Y

    beta = beta.flatten()  # Converts beta to shape (5,)
    
    return beta
    

def ridge_regression(X, y, lambda_param):
    """
    Performs ridge regression on the input data X and target y with regularization parameter lambda_param.
    """
    # Get the number of features
    n_features = X.shape[1]
    
    # Identity matrix for regularization (excluding the bias term if needed)
    I = np.eye(n_features)
    
    # Ridge regression formula: (X^T X + λI)^(-1) X^T y
    ridge_term = lambda_param * I
    beta = np.dot( np.dot(np.linalg.inv( np.add( np.dot(np.transpose(X),X), ridge_term) ), np.transpose(X)), y)
    
    beta = beta.flatten()  # Converts beta to shape (5,)
    
    return beta

def main():
    
    data = np.load('X_train.npy')
    y = np.load('y_train.npy')

    # Initializations 
    
    scaler = MinMaxScaler()
    Y = np.zeros((200,1))
    Y_rigid = np.zeros((200,1))
    y = y.reshape(-1, 1)
    lambda_param = 0.2
    
    # Normalize the entire dataset (all 5 features)
    
    normalized_data = scaler.fit_transform(data)
    y_normalised = scaler.fit_transform(y)
    X = data[:, :5]  # Extracts the first 5 columns from data as a 2D array

    beta = linear_regression(X, y_normalised)
    
    beta_ridge = ridge_regression(X, y_normalised, lambda_param)

    for j in range(X.shape[0]):
    # Calculate y_hat^j
        Y[j] = np.dot(beta, X[j, :])
        
    for j in range(X.shape[0]):
    # Calculate y_hat^j
        Y_rigid[j] = np.dot(beta_ridge, X[j, :])

    Y_normalised = scaler.fit_transform(Y)
    
    Y_rigid_normalised = scaler.fit_transform(Y_rigid)

    r2 = Coefficient_of_Determination(y_normalised,Y_normalised)
    
    #r2_rigid = Coefficient_of_Determination(y_normalised,Y_rigid_normalised)
    
    print("r2 :", r2)
    #print("r2_rigid :", r2_rigid)

    # Get the current date and time
    current_time = datetime.now()

    # Print the date and time
    print("Current Date and Time: ", current_time.strftime("%Y-%m-%d %H:%M:%S"))
    
    #plot the data
    
    plt.figure(figsize=(10, 6))

    # plt.scatter(range(len(feature_1)), feature_1, color='red', label='Feature 1')
    # plt.scatter(range(len(feature_2)), feature_2, color='blue', label='Feature 2')
    # plt.scatter(range(len(feature_3)), feature_3, color='green', label='Feature 3')
    # plt.scatter(range(len(feature_4)), feature_4, color='orange', label='Feature 4')
    # plt.scatter(range(len(feature_5)), feature_5, color='purple', label='Feature 5')
    
    plt.scatter(range(len(y_normalised)), y_normalised, color='red', label='y')
    plt.scatter(range(len(Y_normalised)), Y_normalised, color='blue', label='ypred')
    #plt.scatter(range(len(Y_rigid_normalised)), Y_normalised, color='green', label='yrigid')

    plt.title('Normalized Features')
    plt.xlabel('Index')
    plt.ylabel('Normalized Value')

    plt.legend()

    plt.grid(True)
    plt.show()
    

if __name__ == "__main__":
    main()