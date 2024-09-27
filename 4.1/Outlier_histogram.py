import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, RANSACRegressor, ElasticNetCV
from sklearn.model_selection import train_test_split
from datetime import datetime


def take_out_bad_value(y_true,y_pred):
    dif_array = np.zeros(len(y_true))
    for i in range(len(y_true)):
        dif_array[i] = abs(y_true[i].item() - y_pred[i].item())

    index_max_value = np.argmax(dif_array)

    return index_max_value

def sse(y_true, y_pred):
    
    return np.sum((y_true - y_pred) ** 2)


def main():
    

    data_x = np.load('X_train.npy')
    data_y = np.load('y_train.npy')
    data_x_test = np.load('x_test.npy')

    #############################  INITIALIZATIONS  ############################# 
    
    scaler_y = MinMaxScaler()
    data_y = data_y.reshape(-1, 1)  
    y = scaler_y.fit_transform(data_y)
    best_r2 = 0
    r2_Array = np.zeros(200)
    # Normalize the entire dataset (all 5 features)
    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(data_x[:, :5])
    X_test_file = scaler_X.fit_transform(data_x_test[:, :5])
    
    # Initialize the linear regression model
    model_linear = LinearRegression()

    # Fit the model on the normalized data
    model_linear.fit(X, y)

    # Predict the target values (optional)
    Y = model_linear.predict(X)

    for i in range(200):
        r2_Array[i] = abs(y[i]-Y[i])
        

    Y = np.zeros((np.shape(X)[0],1))
    Y_rigid = np.zeros((np.shape(X)[0],1))
     
    #############################  PLOTS  #############################
    
    plt.figure(figsize=(10, 6))
    
    # plt.scatter(partition_array,r_mean_array, color='red', label='y')
    
    plt.xlabel('Partition')
    plt.ylabel('R2 Values')

    plt.legend()
    plt.grid(True)
    plt.show()
    
    

if __name__ == "__main__":
    
    main()