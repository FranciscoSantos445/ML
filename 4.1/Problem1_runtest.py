import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, RANSACRegressor, ElasticNetCV
from sklearn.model_selection import train_test_split

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


    ##################################### outlier removal #####################################
    
    
    for i in range(number_outliers):

        # Initialize the linear regression model
        model_linear = LinearRegression()

        # Fit the model on the normalized data
        model_linear.fit(data_x, data_y)

        # Predict the target values (optional)
        Y = model_linear.predict(data_x)

        bad_index = take_out_bad_value(data_y,Y) # deletes worse value

        data_y = np.delete(data_y, bad_index, axis=0)
        data_x = np.delete(data_x, bad_index, axis=0)
        
        
    ##################################### model prediction #####################################

    # Initialize the rigid regression model
    model_rigid = RidgeCV(alphas = alphas_gen1).fit(data_x,data_y)
    
    # Predict 
    Y_rigid = model_rigid.predict(data_x_test)
  
    np.save('y_test', Y_rigid)

if __name__ == "__main__":
    
    main()