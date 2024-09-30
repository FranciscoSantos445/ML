import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, RANSACRegressor, ElasticNetCV
from sklearn.model_selection import train_test_split

def take_out_bad_value(y_true,y_pred):
    
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
    data_x_test = np.load('x_test.npy')

    #############################  INITIALIZATIONS  ############################# 
    
    scaler_y = MinMaxScaler()
    data_y = data_y.reshape(-1, 1)  
    y = scaler_y.fit_transform(data_y)
    # Normalize the entire dataset (all 5 features)
    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(data_x[:, :5])
    
    
    r2_elastic = np.zeros(100)
    r2_lasso = np.zeros(100)
    r2_rigid = np.zeros(100)
    r2_linear = np.zeros(100)
    
    r2_elastic_points = np.zeros(100)
    r2_lasso_points = np.zeros(100)
    r2_rigid_points = np.zeros(100)
    r2_linear_points = np.zeros(100)
    
    estimated_coef = np.zeros((5, 3)) 
    
    alphas_gen1 = np.arange(0.1, 100, 0.1)
    alphas_gen2 = np.arange(0.00001, 0.0001, 0.0005)

    points_linear = 0
    points_rigid = 0
    points_lasso = 0
    points_elastic = 0
    
    number_outliers = int(data_x.shape[0] * 0.25)
    
    for i in range(number_outliers):

        # Initialize the linear regression model
        model_linear = LinearRegression()

        # Fit the model on the normalized data
        model_linear.fit(X, y)

        # Predict the target values (optional)
        Y = model_linear.predict(X)

        bad_index = take_out_bad_value(y,Y)

        data_y = np.delete(y, bad_index, axis=0)
        data_x = np.delete(X, bad_index, axis=0)
    
    
    ######################## train multiple times to get the best model ###################3
    
    for index in range(100):

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=None)
        
        y_train.ravel()

        #############################  linear REGRESSION  #############################

        # Initialize the linear regression model
        model_linear = LinearRegression().fit(X_train, y_train)
        
        # Calculates R2 Coeficient
        r2_linear[index]  = model_linear.score(X_test, y_test)
        
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

        model_elastic = ElasticNetCV(
            l1_ratio=[0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8, 0.9], # L1 to L2 mixing (Lasso to Ridge)
            alphas=None,               # Use default alpha values if not provided
            cv=5,                      # 5-fold cross-validation
            max_iter=10000,            # Max iterations (increase if needed)
            tol=1e-4,                  # Convergence tolerance
        ).fit(X_train, y_train.ravel())

        r2_elastic[index] = model_elastic.score(X_test,y_test)


        r2_scores = {
            'linear': r2_linear[index],
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
        elif sorted_models[0] == 'rigid':
            points_rigid += 5
        elif sorted_models[0] == 'lasso':
            points_lasso += 5
        elif sorted_models[0] == 'elastic':
            points_elastic += 5
        
        if sorted_models[1] == 'linear':
            points_linear += 4
        elif sorted_models[1] == 'rigid':
            points_rigid += 4
        elif sorted_models[1] == 'lasso':
            points_lasso += 4
        elif sorted_models[1] == 'elastic':
            points_elastic += 4
        
        if sorted_models[2] == 'linear':
            points_linear += 3
        elif sorted_models[2] == 'rigid':
            points_rigid += 3
        elif sorted_models[2] == 'lasso':
            points_lasso += 3
        elif sorted_models[2] == 'elastic':
            points_elastic += 3
        
        if sorted_models[3] == 'linear':
            points_linear += 2
        elif sorted_models[3] == 'rigid':
            points_rigid += 2
        elif sorted_models[3] == 'lasso':
            points_lasso += 2
        elif sorted_models[3] == 'elastic':
            points_elastic += 2

        r2_elastic_points[index] = points_elastic
        r2_linear_points[index] = points_linear
        r2_lasso_points[index] = points_lasso
        r2_rigid_points[index] = points_rigid

    ############################# Prints  #############################

    # After the loop, print the total points for each model
    print(f"Total Points - Linear: {points_linear}, Ridge: {points_rigid}, Lasso: {points_lasso}, ElasticNet: {points_elastic}")

    # Determine the model with the most points
    points = {
        'linear': points_linear,
        'rigid': points_rigid,
        'lasso': points_lasso,
        'elastic': points_elastic
    }

    best_model = max(points, key=points.get)
    print(f"The model with the highest points is: {best_model}")
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(range(len(r2_linear_points)),r2_linear_points, color='red', label='y_linear')
    plt.plot(range(len(r2_rigid_points)),r2_rigid_points, color='blue', label='y_rigid')
    plt.plot(range(len(r2_lasso_points)),r2_lasso_points, color='green', label='y_lasso')
    plt.plot(range(len(r2_elastic_points)),r2_elastic_points, color='black', label='y_elastic')    
    
    plt.title('Models Scores')
    plt.xlabel('Interactions')
    plt.ylabel('Score values')

    plt.legend()
    plt.grid(True)
    
    plt.show()

    

if __name__ == "__main__":
    
    main()