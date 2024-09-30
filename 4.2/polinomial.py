import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score


def main():
    # Sample data (replace this with your actual data)
    data_x = np.load('u_train.npy')
    data_y = np.load('output_train.npy')
    data_x_test = np.load('u_test.npy')
    
    data_x = data_x.reshape(-1, 1)

    # Degree of the polynomial (e.g., 2 for quadratic, 3 for cubic)
    degrees = range(9, 50)
    
        # Store MSE values for each degree
    mse_values = []

    # Loop over each degree and evaluate performance using cross-validation
    for degree in degrees:
        # Transform the input data to polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(data_x)

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_poly, data_y)

        # Use cross-validation to evaluate the model
        mse = -cross_val_score(model, X_poly, data_y, cv=5, scoring='neg_mean_squared_error').mean()
        
        # Store the MSE for this degree
        mse_values.append(mse)
        print(f'Degree {degree}: MSE = {mse}')
    
    best_degree = degrees[np.argmin(mse_values)]

    # Transform the input features to polynomial features
    poly = PolynomialFeatures(degree=best_degree)
    X_poly = poly.fit_transform(data_x)

    # Create and fit the polynomial regression model
    model = LinearRegression()
    model.fit(X_poly, data_y)

    # Predict the output using the trained model
    y_pred = model.predict(X_poly)

    # Calculate and print the Mean Squared Error (MSE)
    mse = mean_squared_error(data_y, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Plot the original data and the polynomial regression fit
    plt.scatter(range(len(data_y)), data_y, color='blue', label='Original data')
    plt.plot(range(len(y_pred)), y_pred, color='red', label=f'Polynomial fit (degree={best_degree})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Polynomial Regression (degree={best_degree})')
    plt.show()

if __name__ == "__main__":
    
    main()