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

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    
    #term1 = np.linalg.inv( np.dot(np.transpose(X),X) ) # (XT * X)** -1
    
    #X = [normalized_data[:, 0] , normalized_data[:, 1], normalized_data[:, 2], normalized_data[:, 3], normalized_data[:, 4]]
    X = [data[:, 0] , data[:, 1], data[:, 2], data[:, 3], data[:, 4]]
    
    #y = beta + np.sum(beta[1:i] * data[:,i]) 
    
    beta = np.dot( np.dot(np.linalg.inv( np.dot(np.transpose(X),X) ), np.transpose(X)), y)  # (XT * X)** -1 * XT * Y


    # Normalize the entire dataset (all 5 features)
    normalized_data = scaler.fit_transform(data)

    # Extract normalized features
    feature_1 = normalized_data[:, 0]
    feature_2 = normalized_data[:, 1]
    feature_3 = normalized_data[:, 2]
    feature_4 = normalized_data[:, 3]
    feature_5 = normalized_data[:, 4]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each feature with a different color
    # plt.scatter(range(len(feature_1)), feature_1, color='red', label='Feature 1')
    # plt.scatter(range(len(feature_2)), feature_2, color='blue', label='Feature 2')
    plt.scatter(range(len(feature_3)), feature_3, color='green', label='Feature 3')
    plt.scatter(range(len(feature_4)), feature_4, color='orange', label='Feature 4')
    plt.scatter(range(len(feature_5)), feature_5, color='purple', label='Feature 5')

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