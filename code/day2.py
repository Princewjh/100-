#简单线性回归 Simple Linear Regression

#1.数据预处理 Data Preprocessing
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

dataset = pd.read_csv('../dataset/studentscores.csv')
X = dataset.iloc[ : , : -1].values
Y = dataset.iloc[ : , 1].values



from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y ,test_size = 0.25, random_state = 0)

#2在训练集上拟合简单线性回归模型
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

#3 预测结果
Y_pred = regressor.predict(X_test)

#4 可视化
#可视化训练数据
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.show()

#可视化测试数据集
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, Y_pred, color = 'blue')
plt.show()


