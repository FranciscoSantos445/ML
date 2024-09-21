import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
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
    

    data_x = np.load('X_train.npy')
    data_y = np.load('y_train.npy')
    data_test_x = np.load('x_test.npy')

    #############################  INITIALIZATIONS  ############################# 
    
    scaler_y = MinMaxScaler()
    data_y = data_y.reshape(-1, 1)  
    y = scaler_y.fit_transform(data_y)
    best_r2 = 0
    
    # Normalize the entire dataset (all 5 features)
    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(data_x[:, :5])
    
    outliers_y = outliers(y)

    y = np.delete(y,outliers_y,axis=0)  
        
    X = np.delete(X,outliers_y,axis=0)
    
    Y = np.zeros((np.shape(X)[0],1))
    Y_rigid = np.zeros((np.shape(X)[0],1))
    
    alphas_gen1 = np.arange(0.01, 10, 0.01)
    alphas_gen2 = np.arange(0.00001, 0.001, 0.00005)
    
    for index in range(50):

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True)
        
        #############################  LINEAR REGRESSION  #############################
        
        y_train.ravel()
        
        # Initialize the linear regression model
        model_linear = LinearRegression()

        # Fit the model on the data
        model_linear.fit(X_train, y_train)

        # Predict the target values (optional)
        Y = model_linear.predict(X_test)
        
        # Calculates R2 Coeficient
        r2 = model_linear.score(X_test, y_test)

        # Store the best Linear model if the R2 is better than the previous best
        if r2 > best_r2:
            best_r2 = r2
            best_model_linear = model_linear
            best_X_train, best_y_train = X_train, y_train

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)     
    
    #############################  linear REGRESSION  #############################

    Y = best_model_linear.predict(X_test)
    
    # Calculates R2 Coeficient
    r2 = model_linear.score(X_test, y_test)

    #############################  Rigid REGRESSION  #############################
    
    # Initialize the rigid regression model
    model_rigid = RidgeCV(alphas = alphas_gen1).fit(best_X_train, best_y_train)

    # Fit the model on the normalized data
    model_rigid.fit(best_X_train, best_y_train)

    # Predict 
    Y_rigid = model_rigid.predict(X_test)

    # Calculates R2 Coeficient
    r2_rigid = model_rigid.score(X_test, y_test)
    
    #############################  Lasso REGRESSION  ############################
    
    # Initialize the rigid regression model
    model_lasso = LassoCV(alphas=alphas_gen2).fit(best_X_train, best_y_train)
    
    # Fit the model on the normalized data             é preciso?
    model_lasso.fit(best_X_train, best_y_train)

    # Predict
    Y_lasso = model_lasso.predict(X_test)

    # Calculates R2 Coeficient
    r2_lasso = model_lasso.score(X_test, y_test)
    
    ################################## X_test.npy ###########################
    
    # Y_lasso = model_lasso.predict(X_test)
    
    
    #############################  Denormalization ##########################
        
    Y = scaler_y.inverse_transform(Y.reshape(-1, 1))
    Y_lasso = scaler_y.inverse_transform(Y_lasso.reshape(-1, 1))
    Y_rigid = scaler_y.inverse_transform(Y_rigid.reshape(-1, 1))
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))
     
     
    ############################# Prints  #############################

    print("R2 Linear Regression: ", r2)
    print("R2 Rigid Regression: ", r2_rigid)
    print("R2 Lasso Regression: ", r2_lasso)
    
    print(f"Best α_rigid = {model_rigid.alpha_}")
    print(f"Best α_lasso = {model_lasso.alpha_}")
    
    #############################  PLOTS  #############################

    # # plt.scatter(range(len(feature_1)), feature_1, color='red', label='Feature 1')
    # # plt.scatter(range(len(feature_2)), feature_2, color='blue', label='Feature 2')
    # # plt.scatter(range(len(feature_3)), feature_3, color='green', label='Feature 3')
    # # plt.scatter(range(len(feature_4)), feature_4, color='orange', label='Feature 4')
    # # plt.scatter(range(len(feature_5)), feature_5, color='purple', label='Feature 5')
    
    # plt.figure(figsize=(10, 6))
    
    # plt.scatter(range(len(y_test)),y_test, color='red', label='y')
    # plt.scatter(range(len(Y)), Y, color='blue', label='y_linear')
    # plt.scatter(range(len(Y_rigid)), Y_rigid, color='green', label='y_rigid')
    # plt.scatter(range(len(Y_lasso)), Y_lasso, color='purple', label='y_lasso')

    # plt.title('Normalized Features')
    # plt.xlabel('Index')
    # plt.ylabel('Normalized Value')

    # plt.legend()
    # plt.grid(True)
    
    # plt.figure(figsize=(10, 6))
    
    # plt.scatter(Y, y_test, color='blue', label='Predictions vs Actual')
    
    # plt.title('Y validation vs Y predicted')
    # plt.xlabel('Y predicted')
    # plt.ylabel('Y validation')

    # plt.legend()
    # plt.grid(True)
    
    # plt.figure(figsize=(10, 6))
    
    # plt.scatter(Y, y_test, color='blue', label='Predictions vs Actual')
    
    # plt.title('Y validation vs Y predicted')
    # plt.xlabel('Y predicted')
    # plt.ylabel('Y validation')

    # plt.legend()
    # plt.grid(True)
    
    # plt.show()

if __name__ == "__main__":
    main()