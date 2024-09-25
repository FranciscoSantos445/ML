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
    print("Partition Array: ", partition_array)

    return best_partition

def main():
    

    data_x = np.load('X_train.npy')
    data_y = np.load('y_train.npy')
    data_x_test = np.load('x_test.npy')

    #############################  INITIALIZATIONS  ############################# 
    
    scaler_y = MinMaxScaler()
    data_y = data_y.reshape(-1, 1)  
    y = scaler_y.fit_transform(data_y)
    # Normalize the entire dataset (all 5 features)
    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(data_x[:, :5])
    
    
    r2_elastic = np.zeros(400)
    r2_lasso = np.zeros(400)
    r2_rigid = np.zeros(400)
    r2_ransac = np.zeros(400)
    r2_linear = np.zeros(400)
    
    estimated_coef = np.zeros((5, 3)) 
    
    alphas_gen1 = np.arange(0.01, 10, 0.01)
    alphas_gen2 = np.arange(0.00001, 0.001, 0.00005)
    k = 17

    points_linear = 0
    points_ransac = 0
    points_rigid = 0
    points_lasso = 0
    points_elastic = 0
    
    partiotion = best_partition(X, y)
    
    ######################## train multiple times to get the best model ###################3
    
    for index in range(400):

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = partiotion, random_state=None)
        
        y_train.ravel()

        #############################  linear REGRESSION  #############################

        # Initialize the linear regression model
        model_linear = LinearRegression().fit(X_train, y_train)
        
        # Calculates R2 Coeficient
        r2_linear[index]  = model_linear.score(X_test, y_test)
        
        #############################  RANSAC REGRESSION  ############################
        
        model_ransac = RANSACRegressor(random_state=0).fit(X_train,y_train )
        
        r2_ransac[index]  = model_ransac.score(X_test, y_test)
        
        #############################  Rigid REGRESSION  #############################
        
        # Initialize the rigid regression model
        model_rigid = RidgeCV(alphas = alphas_gen1).fit(X_train, y_train)

        # Calculates R2 Coeficient
        r2_rigid[index]  = model_rigid.score(X_test, y_test)
        
        #############################  Lasso REGRESSION  ############################
        
        # Initialize the rigid regression model
        model_lasso = LassoCV(alphas=alphas_gen2).fit(X_train, y_train)

        # Calculates R2 Coeficient
        r2_lasso[index]  = model_lasso.score(X_test, y_test)
        
        #############################  ElasticNet REGRESSION  ############################

        model_elastic = ElasticNetCV(cv=5, random_state=0).fit(X_train,y_train)

        r2_elastic[index] = model_elastic.score(X_test,y_test)


        r2_scores = {
            'linear': r2_linear[index],
            'ransac': r2_ransac[index],
            'rigid': r2_rigid[index],
            'lasso': r2_lasso[index],
            'elastic': r2_elastic[index]
        }

        sorted_r2 = sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)

         # Assign points based on ranking
        sorted_models = [model for model, r2 in sorted_r2]  # This extracts the models in order of R2

        # Award points (5 for highest, 4 for second, etc.)
        if sorted_models[0] == 'linear':
            points_linear += 5
        elif sorted_models[0] == 'ransac':
            points_ransac += 5
        elif sorted_models[0] == 'rigid':
            points_rigid += 5
        elif sorted_models[0] == 'lasso':
            points_lasso += 5
        elif sorted_models[0] == 'elastic':
            points_elastic += 5
        
        if sorted_models[1] == 'linear':
            points_linear += 4
        elif sorted_models[1] == 'ransac':
            points_ransac += 4
        elif sorted_models[1] == 'rigid':
            points_rigid += 4
        elif sorted_models[1] == 'lasso':
            points_lasso += 4
        elif sorted_models[1] == 'elastic':
            points_elastic += 4
        
        if sorted_models[2] == 'linear':
            points_linear += 3
        elif sorted_models[2] == 'ransac':
            points_ransac += 3
        elif sorted_models[2] == 'rigid':
            points_rigid += 3
        elif sorted_models[2] == 'lasso':
            points_lasso += 3
        elif sorted_models[2] == 'elastic':
            points_elastic += 3
        
        if sorted_models[3] == 'linear':
            points_linear += 2
        elif sorted_models[3] == 'ransac':
            points_ransac += 2
        elif sorted_models[3] == 'rigid':
            points_rigid += 2
        elif sorted_models[3] == 'lasso':
            points_lasso += 2
        elif sorted_models[3] == 'elastic':
            points_elastic += 2
        
        if sorted_models[4] == 'linear':
            points_linear += 1
        elif sorted_models[4] == 'ransac':
            points_ransac += 1
        elif sorted_models[4] == 'rigid':
            points_rigid += 1
        elif sorted_models[4] == 'lasso':
            points_lasso += 1
        elif sorted_models[4] == 'elastic':
            points_elastic += 1


    ############################# Prints  #############################

    # After the loop, print the total points for each model
    print(f"Total Points - Linear: {points_linear}, RANSAC: {points_ransac}, Ridge: {points_rigid}, Lasso: {points_lasso}, ElasticNet: {points_elastic}")

    # Determine the model with the most points
    points = {
        'linear': points_linear,
        'ransac': points_ransac,
        'rigid': points_rigid,
        'lasso': points_lasso,
        'elastic': points_elastic
    }

    best_model = max(points, key=points.get)
    print(f"The model with the highest points is: {best_model}")

    

if __name__ == "__main__":
    
    main()