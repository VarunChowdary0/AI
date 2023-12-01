'''
    Basic Execution of Linear Regression Algorithem,
      and Data visualization using matplotlib
'''


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

myDataSet = pd.read_csv('./DataSets/SalaryData.csv')
# print("---------------- Data Set  --------------- \n",myDataSet)
# print("----------------   Shape   --------------- \n",myDataSet.shape)
# print("--------------- Description--------------- \n",myDataSet.describe())

myDataSet.plot(x='YearsExperience',y='Salary')
plt.title("Experience Vs Salary")
plt.xlabel('Experience')
plt.ylabel('Salary')
# plt.show()

# data splicing;

X = myDataSet['YearsExperience'].values.reshape(-1,1)
Y = myDataSet['Salary'].values.reshape(-1,1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.2,random_state=2)

'''
    first parameter : independent variables
    second parameter : dependent
    third parameter : training split percentage
    4th para,meter : how much random the test values should be ?
'''

regressor = LinearRegression()
regressor.fit(X_train,Y_train)  #training using linear Regression.

print(regressor.intercept_) # intercept
print(regressor.coef_) # slope

Y_pred=regressor.predict(X_test)

df = pd.DataFrame({'Actual':Y_test.flatten(),'Predicted':Y_pred.flatten()})
print(df)

df.plot(kind='bar')
plt.show()

plt.scatter(X_test,Y_test, color= 'green')
plt.plot(X_test,Y_pred,color='red')
plt.show()

print("Mean Abs Error: ",metrics.mean_absolute_error(Y_test,Y_pred))
print("Mean Sqr Error: ",metrics.mean_squared_error(Y_test,Y_pred))
print("RMS Error: ",np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))

breaker = False
while(not breaker):
    YourExp = float(input(">> "))
    if(int(YourExp) == -1):
        break
    print(regressor.predict([[YourExp]]))