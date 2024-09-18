import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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
    
    dif = y - y_pred

    # Square the residuals
    squared_residuals = dif ** 2
    
    y_mean = np.mean(y)

    # Sum the squared residuals to get the sum of squared errors (SSE)
    SSEres = np.sum(squared_residuals)
    
    # Compute the total sum of squares
    SStotal = np.sum((y - y_mean) ** 2)
    
    r2 = 1 - (SSEres/SStotal)
    
    return r2
    

def main():
    
    data = np.load('X_train.npy')
    y = np.load('y_train.npy')

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    Y  = np.zeros((200,1))
    y = y.reshape(-1, 1)
    # Normalize the entire dataset (all 5 features)
    normalized_data = scaler.fit_transform(data)
    y_normalised = scaler.fit_transform(y)
    #X = [normalized_data[:, 0] , normalized_data[:, 1], normalized_data[:, 2], normalized_data[:, 3], normalized_data[:, 4]]
    X = data[:, :5]  # Extracts the first 5 columns from data as a 2D array

    
    #y = beta + np.sum(beta[1:i] * data[:,i]) 
    
    beta = np.dot( np.dot(np.linalg.inv( np.dot(np.transpose(X),X) ), np.transpose(X)), y_normalised)  # (XT * X)** -1 * XT * Y

    beta = beta.flatten()  # Converts beta to shape (5,)

    for j in range(200):
    # Calculate y_hat^j
        Y[j] = np.dot(beta, X[j, :])

    Y_normalised = scaler.fit_transform(Y)

    r2 = Coefficient_of_Determination(y_normalised,Y_normalised)


    # Create the plot
    plt.figure(figsize=(10, 6))

    # # Plot each feature with a different color
    # # plt.scatter(range(len(feature_1)), feature_1, color='red', label='Feature 1')
    # # plt.scatter(range(len(feature_2)), feature_2, color='blue', label='Feature 2')
    # plt.scatter(range(len(feature_3)), feature_3, color='green', label='Feature 3')
    # plt.scatter(range(len(feature_4)), feature_4, color='orange', label='Feature 4')
    # plt.scatter(range(len(feature_5)), feature_5, color='purple', label='Feature 5')
    print("r2 :", r2)
    plt.scatter(range(len(y_normalised)), y_normalised, color='purple', label='y')
    plt.scatter(range(len(Y_normalised)), Y_normalised, color='blue', label='ypred')

    # Add title and labels
    plt.title('Normalized Features')
    plt.xlabel('Index')
    plt.ylabel('Normalized Value')

    # Show legend
    plt.legend()

    # Show grid and plot
    plt.grid(True)
    plt.show()
    

if __name__ == "__main__":
    main()