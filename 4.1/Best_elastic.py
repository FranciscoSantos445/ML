import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split
from datetime import datetime

def take_out_shit_value(y_true,y_pred):
    
    dif_array = np.zeros(len(y_true))
    
    for i in range(len(y_true)):
        
        y = y_true[i][0]
    
        y_calc_pred = y_pred[i][0]
        
        dif_array[i] = abs(y - y_calc_pred)

    index_max_value = np.argmax(dif_array)

    return index_max_value

def main():
    
    data_x = np.load('X_train.npy')
    data_y = np.load('y_train.npy')

    #############################  INITIALIZATIONS  ############################# 
    
    scaler_y = MinMaxScaler()
    data_y = data_y.reshape(-1, 1)  
    y = scaler_y.fit_transform(data_y)
    
    # Normalize the entire dataset (all 5 features)
    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(data_x[:, :5])
    
    Y_rigid = np.zeros((np.shape(X)[0],1))
    
    k = 17

    for i in range(50):
    
        # Initialize the linear regression model
        model_linear = LinearRegression()

        # Fit the model on the normalized data
        model_linear.fit(X, y)

        # Predict the target values (optional)
        Y = model_linear.predict(X)

        shit_index = take_out_shit_value(y,Y)

        y = np.delete(y, shit_index, axis=0)
        X = np.delete(X, shit_index, axis=0)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.38, random_state=None)

    #############################  ElasticNet REGRESSION  ############################
    
    model_elastic = ElasticNetCV(
        l1_ratio=[0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8, 0.9], # L1 to L2 mixing (Lasso to Ridge)
        alphas=None,               # Use default alpha values if not provided
        cv=5,                      # 5-fold cross-validation
        max_iter=10000,            # Max iterations (increase if needed)
        tol=1e-4,                  # Convergence tolerance
    )
    
    model_elastic.fit(X_train, y_train.ravel())

    Y_elastic = model_elastic.predict(X_test)

    r2_elastic = model_elastic.score(X_test,y_test)
    
    #############################  Denormalization ##########################
    
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    Y_elastic = scaler_y.inverse_transform(Y_elastic.reshape(-1, 1))
     
    ############################# Prints  #############################

    print("R2 Elastic Regression: ", r2_elastic)
    
    print(f"Best α_elastic = {model_elastic.alpha_}")
    
    print(f"Best L1 ratio = {model_elastic.l1_ratio_}")
    
    #############################  PLOTS  #############################
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(range(len(y_test)),y_test, color='red', label='y')
    plt.scatter(range(len(Y_elastic)), Y_elastic, color='blue', label='y_elastic')

    plt.title('Normalized Features')
    plt.xlabel('Index')
    plt.ylabel('Normalized Value')

    plt.legend()
    plt.grid(True)
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(Y_elastic, y_test, color='blue', label='Predictions vs Actual')
    
    plt.title('Y validation vs Y predicted')
    plt.xlabel('Y predicted')
    plt.ylabel('Y validation')

    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()