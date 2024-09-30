import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def main():

    data_x = np.load('u_train.npy')  # Input data
    data_x_test = np.load('u_test.npy')  # Input data
    data_y = np.load('output_train.npy')  # Output data
    
    data_x = data_x.reshape(-1,1)
        
    model=LinearRegression().fit(data_x,data_y)
    y_pred=model.predict(data_x)
    
    plt.figure()
    
    plt.scatter(range(len(y_pred)), y_pred, color='black', label='y_pred')
    plt.scatter(range(len(data_y)), data_y, color='blue', label='Data_y')
    plt.scatter(range(len(data_x)), data_x, color='red', label='Data_x')

    
    plt.show()
    
if __name__ == "__main__":
    main()