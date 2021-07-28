import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, 1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)

#Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
Y_pred = regressor.predict(X_test)


#Visualising Training set
plt.scatter(X_train, y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.xlabel("Years of Experience")
plt.ylabel("Salary of Employees")
plt.show()

#Visualising Test Set
plt.scatter(X_test,y_test,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.xlabel("Years of Experience")
plt.ylabel("Salary of Employees")
plt.show()