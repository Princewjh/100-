#导入库文件
import numpy as np 
import pandas as pd 

#导入数据集
dataset = pd.read_csv('../dataset/Data.csv')
X = dataset.iloc[:,:-1].values  #iloc方法，根据行号来索引，前三列数据为X
Y = dataset.iloc[:,3].values    #最后一列为Y

#处理缺失数据
from sklearn.preprocessing import Imputer
#axis=0表示按列进行，strategy = mean, median, most_frequent
imputer = Imputer(missing_values= 'NaN', strategy="mean", axis=0)
#补全X中2,3列的缺失值
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])


#编码分类特征
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
#构造哑变量
#categorical_features指明列索引，需要对第一列进行编码
onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print(X)
print(Y)

#将数据集分为训练集和测试集
from sklearn.cross_validation import train_test_split
#random_state：是随机数的种子
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 0)

#特征缩放
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print(X_train)
print(X_test)
