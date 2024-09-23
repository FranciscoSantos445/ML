import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, RANSACRegressor
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
    
def best_partition(X, y):
    
    r_Array = np.zeros(50)
    r_mean_array = np.zeros(50)
    partition_array = np.zeros(50)
    
    for index in range(50):

        for index2 in range(50):
            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= (0.05 + (index-1)*0.01633), random_state=None, shuffle=True)
            
            #############################  LINEAR REGRESSION  #############################
            
            partition = 0.1 + (index-1)*0.01633
            
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
        
        partition_array[index] = partition = 0.1 + (index-1)*0.01633
        
        r_mean_array[index] = np.mean(r_Array)
    
    best_partition = partition_array[np.argmax(r_mean_array)]
        
    print("Best partition: ", best_partition)
    
    return best_partition

def main():
    

    data_x = np.load('X_train.npy')
    data_y = np.load('y_train.npy')
    data_x_test = np.load('x_test.npy')

    #############################  INITIALIZATIONS  ############################# 
    
    scaler_y = MinMaxScaler()
    data_y = data_y.reshape(-1, 1)  
    y = scaler_y.fit_transform(data_y)
    best_r2 = 0
    
    # Normalize the entire dataset (all 5 features)
    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(data_x[:, :5])
    X_test_file = scaler_X.fit_transform(data_x_test[:, :5])
    
    outliers_y = outliers(y)

    y = np.delete(y,outliers_y,axis=0)  
        
    X = np.delete(X,outliers_y,axis=0)
    
    Y = np.zeros((np.shape(X)[0],1))
    Y_rigid = np.zeros((np.shape(X)[0],1))
    
    alphas_gen1 = np.arange(0.01, 10, 0.01)
    alphas_gen2 = np.arange(0.00001, 0.001, 0.00005)
    
    partiotion = best_partition(X, y)
    
    ######################## train multiple times to get the best model ###################3
    
    for index in range(50):

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = partiotion, random_state=None)
        
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= None, random_state=None, shuffle=True) 
        
    
    #############################  linear REGRESSION  #############################

    Y = best_model_linear.predict(X_test)
    
    # Calculates R2 Coeficient
    r2 = model_linear.score(X_test, y_test)
    
    #############################  RANSAC REGRESSION  ############################
    
    ransac = RANSACRegressor(random_state=0).fit(best_X_train,best_y_train )
    
    Y_ransac = ransac.predict(X_test)
    
    r2_ransac = ransac.score(X_test, y_test)
    
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
    
    # Fit the model on the normalized data
    model_lasso.fit(best_X_train, best_y_train)

    # Predict
    Y_lasso = model_lasso.predict(X_test)

    # Calculates R2 Coeficient
    r2_lasso = model_lasso.score(X_test, y_test)
    
    ################################## X_test.npy ###########################
    
    Y_file = best_model_linear.predict(X_test_file)
    
    Y_ransac_file = ransac.predict(X_test_file)
    
    Y_rigid_file = model_rigid.predict(X_test_file)
    
    Y_lasso_file = model_lasso.predict(X_test_file)
    
    
    #############################  Denormalization ##########################
        
    Y = scaler_y.inverse_transform(Y.reshape(-1, 1))
    Y_ransac = scaler_y.inverse_transform(Y_ransac.reshape(-1, 1))
    Y_lasso = scaler_y.inverse_transform(Y_lasso.reshape(-1, 1))
    Y_rigid = scaler_y.inverse_transform(Y_rigid.reshape(-1, 1))
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))
     
     
    ############################# Prints  #############################

    print("R2 Linear Regression: ", r2)
    print("R2 ransac Regression: ", r2_ransac)
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
    plt.scatter(range(len(Y_ransac)), Y_ransac, color='black', label='y_ransac')

    plt.title('Y validation vs Y predicted')
    plt.xlabel('Index')
    plt.ylabel('Y Values')

    plt.legend()
    plt.grid(True)
    
    plt.figure(figsize=(10, 6))
    
    if r2 > r2_rigid and r2 > r2_lasso and r2 > r2_ransac:
        plt.scatter(Y,y_test, color='blue', label='y_linear V y_test')
    
    elif r2_ransac > r2_rigid and r2_ransac > r2 and r2_ransac > r2_lasso:
        plt.scatter(Y_ransac,y_test, color='black', label='y_ransac V y_test')
    
    elif r2_rigid > r2 and r2_rigid > r2_lasso and r2_rigid > r2_ransac:
        plt.scatter(Y_rigid,y_test, color='green', label='y_rigid V y_test')
        
    elif r2_lasso > r2_rigid and r2_lasso > r2 and r2_lasso > r2_ransac:
            plt.scatter(Y_lasso,y_test, color='purple', label='y_lasso V y_test')
    
    plt.title('Y validation vs Y predicted')
    plt.xlabel('Y predicted')
    plt.ylabel('Y validation')

    plt.legend()
    plt.grid(True)
    
    ########################## plot do test set ##########################
    
    plt.figure(figsize=(10, 6))
    
    if r2 > r2_rigid and r2 > r2_lasso and r2 > r2_ransac:
        Y_file = scaler_y.inverse_transform(Y_file.reshape(-1, 1))
        plt.scatter(range(len(Y_file)),Y_file, color='blue', label='y_linear')
        
    elif r2_ransac > r2_rigid and r2_ransac > r2 and r2_ransac > r2_lasso:
        Y_ransac_file = scaler_y.inverse_transform(Y_ransac_file.reshape(-1, 1))
        plt.scatter(range(len(Y_ransac_file)),Y_ransac_file, color='black', label='y_ransac V y_test')

        
    elif r2_rigid > r2 and r2_rigid > r2_lasso and r2_rigid > r2_ransac:
        Y_rigid_file = scaler_y.inverse_transform(Y_rigid_file.reshape(-1, 1))
        plt.scatter(range(len(Y_rigid_file)),Y_rigid_file, color='green', label='y_rigid')
        
    elif r2_lasso > r2_rigid and r2_lasso > r2 and r2_lasso > r2_ransac:
        Y_lasso_file = scaler_y.inverse_transform(Y_lasso_file.reshape(-1, 1))  
        plt.scatter(range(len(Y_lasso_file)),Y_lasso_file, color='purple', label='y_lasso')
        
    
                
    plt.title('Y test models ')
    plt.xlabel('Indexes')
    plt.ylabel('Y values')

    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    
    main()