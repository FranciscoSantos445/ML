import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from datetime import datetime


def outliers(y, y_pred, X_train):
    
    distances = abs(y - y_pred)
    
    Q3 = np.percentile(distances, 75)
    
    print(Q3,"\n")
    
    outliers = np.where(distances >= Q3)
    
    y = np.delete(y,outliers,axis=0)
    
    y_pred = np.delete(y_pred,outliers,axis=0)
    
    X_train = np.delete(X_train,outliers,axis=0)
    
    return y , y_pred, X_train
    

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    
    alphas_gen1 = np.arange(0.01, 10, 0.01)
    alphas_gen2 = np.arange(0.00001, 0.001, 0.00005)
    
    #############################  LINEAR REGRESSION  #############################
    
    y_train.ravel()
    
    # Initialize the linear regression model
    model_linear = LinearRegression()

    # Fit the model on the normalized data
    model_linear.fit(X_train, y_train)

    # Predict the target values (optional)
    Y = model_linear.predict(X_train)
    
    y_train, Y, X_train = outliers(y_train, Y, X_train)
    
    model_linear.fit(X_train, y_train)
    
    Y = model_linear.predict(X_test) 
       
    # Calculates R2 Coeficient
    r2 = model_linear.score(X_test, y_test)
    
    #############################  Rigid REGRESSION  #############################
    
    # Initialize the rigid regression model
    model_rigid = RidgeCV(alphas = alphas_gen1).fit(X_train, y_train)

    # Fit the model on the normalized data
    model_rigid.fit(X_train, y_train)

    # Predict the target values (optional)
    Y_rigid = model_rigid.predict(X_test)

    # Calculates R2 Coeficient
    r2_rigid = model_rigid.score(X_test, y_test)
    
    #############################  Lasso REGRESSION  ############################
    
    # Initialize the rigid regression model
    model_lasso = LassoCV(alphas=alphas_gen2).fit(X_train, y_train)
    
    # Fit the model on the normalized data
    model_lasso.fit(X_train, y_train)

    # Predict the target values (optional)
    Y_lasso = model_lasso.predict(X_test)

    # Calculates R2 Coeficient
    r2_lasso = model_lasso.score(X_test, y_test)
       
    
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
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(range(len(y_test)),y_test, color='red', label='y')
    plt.scatter(range(len(Y)), Y, color='blue', label='y_linear')
    plt.scatter(range(len(Y_rigid)), Y_rigid, color='green', label='y_rigid')
    plt.scatter(range(len(Y_lasso)), Y_lasso, color='purple', label='y_lasso')

    plt.title('Normalized Features')
    plt.xlabel('Index')
    plt.ylabel('Normalized Value')

    plt.legend()
    plt.grid(True)
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(Y, y_test, color='blue', label='Predictions vs Actual')
    
    plt.title('Y validation vs Y predicted')
    plt.xlabel('Y predicted')
    plt.ylabel('Y validation')

    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()