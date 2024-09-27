import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, RANSACRegressor, ElasticNetCV
from sklearn.model_selection import train_test_split


def best_partition(data_x, y):
    
    r_Array = np.zeros(46)
    r_mean_array = np.zeros(46)
    partition_array = np.zeros(46)
    
    for index in range(46):

        for index2 in range(46):
            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(data_x, y, test_size= (0.05 + (index-1)*0.01), random_state=None, shuffle=True)
            
            #############################  LINEAR REGRESSION  #############################
            
            partition = 0.05 + (index-1)*0.01
            
            y_train.ravel()
            
            # Initialize the linear regression model
            model_linear = LinearRegression()

            # Fit the model on the data
            model_linear.fit(X_train, y_train)

            # Predict the target values (optional)
            Y = model_linear.predict(X_test)
            
            # Calculates R2 Coeficient
            r2 = model_linear.score(X_test, y_test)
            
            r_Array[index2] = r2
        
        partition_array[index] = partition 
        
        r_mean_array[index] = np.mean(r_Array)
    
    best_partition = partition_array[np.argmax(r_mean_array)]
        
    print("Best partition: ", best_partition)
    
    return best_partition

def sse(y_true, y_pred):
    
    return np.sum((y_true - y_pred) ** 2)

def take_out_bad_value(y_true,y_pred):
    
    dif_array = np.zeros(len(y_true))
    
    for i in range(len(y_true)):
        
        y = y_true[i]
    
        y_calc_pred = y_pred[i]
        
        dif_array[i] = abs(y - y_calc_pred)

    index_max_value = np.argmax(dif_array)

    return index_max_value

def main():
    
    data_x = np.load('X_train.npy')
    data_y = np.load('y_train.npy')
    data_x_test = np.load('x_test.npy')
    
    alphas_gen1 = np.arange(0.01, 10, 0.01)
    number_outliers = int(data_x.shape[0] * 0.25) # 25% of the data is outliers
    
    #############################  Normalization  ############################# 
        
    scaler_y = MinMaxScaler()
    data_y = data_y.reshape(-1, 1)  
    y = scaler_y.fit_transform(data_y)
    
    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(data_x[:, :5])
    X_test_file = scaler_X.fit_transform(data_x_test[:, :5])
    
    ##################################### outlier removal #####################################
    
    
    for i in range(number_outliers):

        # Initialize the linear regression model
        model_linear = LinearRegression()

        # Fit the model on the normalized data
        model_linear.fit(data_x, data_y)

        # Predict the target values (optional)
        Y = model_linear.predict(data_x)

        bad_index = take_out_bad_value(data_y,Y)

        data_y = np.delete(data_y, bad_index, axis=0)
        data_x = np.delete(data_x, bad_index, axis=0)
        
        
    ##################################### model prediction #####################################

    # Initialize the rigid regression model
    model_rigid = RidgeCV(alphas = alphas_gen1).fit(data_x,data_y)
    
    # Predict 
    Y_rigid = model_rigid.predict(data_x_test)
    
    Y_rigid = scaler_y.inverse_transform(Y_rigid.reshape(-1, 1)) # Invert the normalization
    
    np.save('Ytest_Regression', Y_rigid)
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(range(len(Y_rigid)),Y_rigid, color='red', label='y')
    
    plt.show()
    
if __name__ == "__main__":
    
    main()