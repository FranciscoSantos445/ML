import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from datetime import datetime

def outliers(data):
    
    # Compute quartiles
    Q1 = np.percentile(data, 25)
    # Q2 = np.median(data)  # or np.percentile(data, 50)
    Q3 = np.percentile(data, 75)

    # Compute interquartile range (IQR)
    IQR = Q3 - Q1

    # Compute whiskers
    lower_whisker = np.min(data[data >= Q1 - 1.5 * IQR])
    upper_whisker = np.max(data[data <= Q3 + 1.5 * IQR])

    # Find outliers
    #outliers = np.where(data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)])
    outliers = np.where((data < lower_whisker) | (data > upper_whisker))[0]
    
    return outliers

def main():
    

    data = np.load('X_train.npy')
    y = np.load('y_train.npy')

    #############################  INITIALIZATIONS  ############################# 
    
    scaler = MinMaxScaler()
    lambda_param = 10
    y = y.reshape(-1, 1)
     
    # Normalize the entire dataset (all 5 features)
    
    normalized_data = scaler.fit_transform(data)
    y_normalised = scaler.fit_transform(y)
    X = normalized_data[:, :5]  # Extracts the first 5 columns from data as a 2D array
    
    outliers_y = outliers(y_normalised)

    y_normalised = np.delete(y_normalised,outliers_y,axis=0)  
        
    X = np.delete(X,outliers_y,axis=0)
    
    Y = np.zeros((np.shape(X)[0],1))
    Y_rigid = np.zeros((np.shape(X)[0],1))
    


    #############################  REGRESSIONS  ############################# 


    # beta = linear_regression(X, y_normalised)
    
    beta = LinearRegression().fit(X, y_normalised)
    
    # beta_ridge = ridge_regression(X, y_normalised, lambda_param)
    
    
    Y = Ridge(alpha=1.0)
    Y.fit(X, y)
    

    # for j in range(np.shape(X)[0]):
    # # Calculate y_pred
    #     Y[j] = np.matmul(beta, X[j, :])
        
    # print(np.shape(Y),"\n")
        
    # for j in range((np.shape(X)[0])):
    # # Calculate y_pred
    #     Y_rigid[j] = np.matmul(beta_ridge, X[j, :])

    
    

    # r2 = Coefficient_of_Determination(y_normalised,Y)
    
    reg.score(X, Y)
    
    # r2_rigid = Coefficient_of_Determination(y_normalised,Y_rigid)
    
    reg.score(X, Y_rigid)
    
    
    
    
    #############################  PLOTS  #############################



    # Get the current date and time
    current_time = datetime.now()

    # Print the date and time
    print("Current Date and Time: ", current_time.strftime("%Y-%m-%d %H:%M:%S"))
    
    #plot the data
    
    plt.figure(figsize=(10, 6))

    # # plt.scatter(range(len(feature_1)), feature_1, color='red', label='Feature 1')
    # # plt.scatter(range(len(feature_2)), feature_2, color='blue', label='Feature 2')
    # # plt.scatter(range(len(feature_3)), feature_3, color='green', label='Feature 3')
    # # plt.scatter(range(len(feature_4)), feature_4, color='orange', label='Feature 4')
    # # plt.scatter(range(len(feature_5)), feature_5, color='purple', label='Feature 5')
    
    plt.scatter(range(len(y_normalised)), y_normalised, color='red', label='y')
    plt.scatter(range(len(Y)), Y, color='blue', label='ypred')
    plt.scatter(range(len(Y_rigid)), Y_rigid, color='green', label='yrigid', s = 10)

    plt.title('Normalized Features')
    plt.xlabel('Index')
    plt.ylabel('Normalized Value')

    plt.legend()

    plt.grid(True)
    plt.show()
    

if __name__ == "__main__":
    main()