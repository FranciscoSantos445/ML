    # beta = linear_regression(X, y_normalised)
    
    # beta_ridge = ridge_regression(X, y_normalised, lambda_param)

    # for j in range(X.shape[0]):
    # # Calculate y_pred
    #     Y[j] = np.matmul(beta, X[j, :])
        
    # for j in range(X.shape[0]):
    # # Calculate y_pred
    #     Y_rigid[j] = np.matmul(beta_ridge, X[j, :])

    # r2 = Coefficient_of_Determination(y_normalised,Y)
    
    # r2_rigid = Coefficient_of_Determination(y_normalised,Y_rigid)
    
    # print("r2 :", r2)
    # print("r2_rigid :", r2_rigid)

    # # Get the current date and time
    # current_time = datetime.now()

    # # Print the date and time
    # print("Current Date and Time: ", current_time.strftime("%Y-%m-%d %H:%M:%S"))
    
    # #plot the data
    
    # plt.figure(figsize=(10, 6))

    # # plt.scatter(range(len(feature_1)), feature_1, color='red', label='Feature 1')
    # # plt.scatter(range(len(feature_2)), feature_2, color='blue', label='Feature 2')
    # # plt.scatter(range(len(feature_3)), feature_3, color='green', label='Feature 3')
    # # plt.scatter(range(len(feature_4)), feature_4, color='orange', label='Feature 4')
    # # plt.scatter(range(len(feature_5)), feature_5, color='purple', label='Feature 5')
    
    # plt.scatter(range(len(y_normalised)), y_normalised, color='red', label='y')
    # plt.scatter(range(len(Y)), Y, color='blue', label='ypred')
    # plt.scatter(range(len(Y_rigid)), Y_rigid, color='green', label='yrigid', s = 10)

    # plt.title('Normalized Features')
    # plt.xlabel('Index')
    # plt.ylabel('Normalized Value')

    # plt.legend()

    # plt.grid(True)
    # plt.show()