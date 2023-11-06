import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
import sklearn

# load the training data
arr = np.loadtxt("dataset_train.csv", delimiter=",", dtype=float)
inputs, targets = arr[:, 0], arr[:, 1]

x = inputs
y_train = targets
# Cast the input data into the X matrix format as previously defined:
x_train = np.array([np.ones_like(x), x, x**2, np.sin(2*np.pi*x)], dtype='float').transpose()

#Model
lr = LinearRegression()

#Fit model
lr.fit(x_train, y_train)

#Ridge Regression Model
ridgeReg = Ridge(alpha=10)

ridgeReg.fit(x_train, y_train)

# load the test data
arr = np.loadtxt("dataset_test.csv", delimiter=",", dtype=float)
inputs_test, targets_test = arr[:, 0], arr[:, 1]

x = inputs_test
x_test = np.array([np.ones_like(x), x, x**2, np.sin(2*np.pi*x)], dtype='float').transpose()

#predict

prediction1 = lr.predict(x_test)
print("MSE 1 test = ", sklearn.metrics.mean_squared_error(targets_test, prediction1))
prediction2 = ridgeReg.predict(x_test)
print("MSE 2 test = ", sklearn.metrics.mean_squared_error(targets_test, prediction2))
