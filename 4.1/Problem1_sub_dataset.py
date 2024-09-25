import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, RANSACRegressor, ElasticNetCV
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

def sse(y_true, y_pred):
    
    return np.sum((y_true - y_pred) ** 2)

def take_out_bad_value(y_true,y_pred):
    dif_array = np.zeros(len(y_true))
    for i in range(len(y_true)):
        dif_array[i] = abs(y_true[i]-y_pred[i])

    index_max_value = np.argmax(dif_array)

    return index_max_value
   
def best_partition(X, y):
    
    r_Array = np.zeros(46)
    r_mean_array = np.zeros(46)
    partition_array = np.zeros(46)
    
    for index in range(46):

        for index2 in range(46):
            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= (0.05 + (index-1)*0.01), random_state=None, shuffle=True)
            
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
    
    for i in range(50):
    
        # Initialize the linear regression model
        model_linear = LinearRegression()

        # Fit the model on the normalized data
        model_linear.fit(X, y)

        # Predict the target values (optional)
        Y = model_linear.predict(X)

        bad_index = take_out_bad_value(y,Y)

        y = np.delete(y, bad_index, axis=0)
        X = np.delete(X, bad_index, axis=0)
    
    Y = np.zeros((np.shape(X)[0],1))
    Y_rigid = np.zeros((np.shape(X)[0],1))
    alphas = np.zeros(3)
    estimated_coef = np.zeros((5, 3)) 
    alphas_gen1 = np.arange(0.01, 10, 0.01)
    alphas_gen2 = np.arange(0.00001, 0.001, 0.00005)
    
    
    partiotion = best_partition(X, y)
    
    ######################## train multiple times to get the best model ###################3
    
    for index in range(200):

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

        r2 = round(r2, 2)

        if r2 > 0:
            r2_Array[index] = r2 
        
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

    sse_linear = sse(y_test,Y)
    
    #############################  RANSAC REGRESSION  ############################
    
    ransac = RANSACRegressor(random_state=0).fit(best_X_train,best_y_train )
    
    Y_ransac = ransac.predict(X_test)
    
    r2_ransac = ransac.score(X_test, y_test)

    sse_ransac = sse(y_test,Y_ransac)
    
    #############################  Rigid REGRESSION  #############################
    
    # Initialize the rigid regression model
    model_rigid = RidgeCV(alphas = alphas_gen1).fit(best_X_train, best_y_train)

    # Predict 
    Y_rigid = model_rigid.predict(X_test)

    # Calculates R2 Coeficient
    r2_rigid = model_rigid.score(X_test, y_test)

    sse_rigid = sse(y_test,Y_rigid)
    
    #############################  Lasso REGRESSION  ############################
    
    # Initialize the rigid regression model
    model_lasso = LassoCV(alphas=alphas_gen2).fit(best_X_train, best_y_train)

    # Predict
    Y_lasso = model_lasso.predict(X_test)

    # Calculates R2 Coeficient
    r2_lasso = model_lasso.score(X_test, y_test)

    sse_lasso = sse(y_test,Y_lasso)
    
    #############################  ElasticNet REGRESSION  ############################

    model_elastic = ElasticNetCV(cv=5, random_state=0)

    model_elastic.fit(best_X_train,best_y_train)

    Y_elastic = model_elastic.predict(X_test)

    r2_elastic = model_elastic.score(X_test,y_test)

    sse_elastic = sse(y_test,Y_elastic)


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
    Y_elastic = scaler_y.inverse_transform(Y_elastic.reshape(-1, 1))
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))
     
     
    ############################# Prints  #############################

    print("R2 Linear Regression: ", r2)
    print("R2 Ransac Regression: ", r2_ransac)
    print("R2 Rigid Regression: ", r2_rigid)
    print("R2 Lasso Regression: ", r2_lasso)
    print("R2 Elastic Regression: ", r2_elastic)

    print("\nSSE Linear Regression: ", sse_linear)
    print("SSE Ransac Regression: ", sse_ransac)
    print("SSE Rigid Regression: ", sse_rigid)
    print("SSE Lasso Regression: ", sse_lasso)
    print("SSE Elastic Regression: ", sse_elastic)
    
    # print(f"Best α_rigid = {model_rigid.alpha_}")
    # print(f"Best α_lasso = {model_lasso.alpha_}")


    # print("alpah\n", model_elastic.alpha_)
    # print("Intercept ", model_elastic.intercept_)
    
    #############################  PLOTS  #############################
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(range(len(y_test)),y_test, color='red', label='y')
    plt.scatter(range(len(Y)), Y, color='blue', label='y_linear')
    plt.scatter(range(len(Y_rigid)), Y_rigid, color='green', label='y_rigid')
    plt.scatter(range(len(Y_lasso)), Y_lasso, color='purple', label='y_lasso')
    plt.scatter(range(len(Y_ransac)), Y_ransac, color='black', label='y_ransac')
    plt.scatter(range(len(Y_elastic)), Y_elastic, color='yellow', label='y_elastic')

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
    plt.show()
    
    ########################## plot do test set ##########################
    
    # plt.figure(figsize=(10, 6))
    
    # if r2 > r2_rigid and r2 > r2_lasso and r2 > r2_ransac:
    #     Y_file = scaler_y.inverse_transform(Y_file.reshape(-1, 1))
    #     plt.scatter(range(len(Y_file)),Y_file, color='blue', label='y_linear')
        
    # elif r2_ransac > r2_rigid and r2_ransac > r2 and r2_ransac > r2_lasso:
    #     Y_ransac_file = scaler_y.inverse_transform(Y_ransac_file.reshape(-1, 1))
    #     plt.scatter(range(len(Y_ransac_file)),Y_ransac_file, color='black', label='y_ransac V y_test')

        
    # elif r2_rigid > r2 and r2_rigid > r2_lasso and r2_rigid > r2_ransac:
    #     Y_rigid_file = scaler_y.inverse_transform(Y_rigid_file.reshape(-1, 1))
    #     plt.scatter(range(len(Y_rigid_file)),Y_rigid_file, color='green', label='y_rigid')
        
    # elif r2_lasso > r2_rigid and r2_lasso > r2 and r2_lasso > r2_ransac:
    #     Y_lasso_file = scaler_y.inverse_transform(Y_lasso_file.reshape(-1, 1))  
    #     plt.scatter(range(len(Y_lasso_file)),Y_lasso_file, color='purple', label='y_lasso')
        
    
                
    # plt.title('Y test models ')
    # plt.xlabel('Indexes')
    # plt.ylabel('Y values')

    # plt.legend()
    # plt.grid(True)
    
    # plt.show()

    ########################## plot do histograma ##########################

    # plt.figure(figsize=(10, 6))
    # # Create a histogram
    # plt.hist(r2_Array, bins=100, edgecolor='black')  # You can adjust 'bins' for more or fewer bars

    # # Add labels and title
    # plt.xlabel('R2')
    # plt.ylabel('Frequency')
    # plt.title('R2 along various tests')

    # # Show the plot
    # plt.legend()
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    
    main()