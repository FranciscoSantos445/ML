import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def main():

    data_x = np.load('u_train.npy')  # Input data
    data_x_test = np.load('u_test.npy')  # Input data
    data_y = np.load('output_train.npy')  # Output data
    
    data_x = data_x.reshape(-1,1)
        
    # Degree of the polynomial (e.g., 2 for quadratic, 3 for cubic)
    degree = 2

    # Transform the input features to polynomial features
    poly = PolynomialFeatures(degree=degree)
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
    plt.scatter(data_x, data_y, color='blue', label='Original data')
    plt.scatter(data_x , y_pred, color='red', label=f'Polynomial fit (degree={degree})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Polynomial Regression (degree={degree})')
plt.show()
    
if __name__ == "__main__":
    main()