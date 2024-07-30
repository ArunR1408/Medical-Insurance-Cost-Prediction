#Medical Insurance Cost Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

d = pd.read_csv('insurance.csv')
print(d)
print('\n')
print(d.shape)
print('\n')
print(d.info)
print('\n')
print(d.isnull().sum())
print('\n')
print(d.describe)
print('\n')

sns.set()
plt.figure(figsize=(6,6))
sns.distplot(d['age'])
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(x="sex",data=d)
plt.title("Sex Distribution")
plt.show()

sns.distplot(d['bmi'])
plt.show()

d['region'].value_counts()
d.replace({'sex':{'male':0,'female':1}},inplace=True)
d.replace({'smoker':{'yes':0,'no':1}},inplace=True)
d.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)
x = d.drop(columns='charges',axis=1)
y = d['charges']
print('\n')
print(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print('\n')
print(x_train.shape)
print('\n')
print(x_test.shape)

reg = LinearRegression()
reg.fit(x_train,y_train)
training_data_prediction = reg.predict(x_train)
r2_train = metrics.r2_score(y_train,training_data_prediction)
print('\n')
print(r2_train)
test_data_prediction = reg.predict(x_test)
print(metrics.r2_score(y_test,test_data_prediction))

sample_input_data = (30,1,27,0,1,0)
input_data_as_numpy_array = np.array(sample_input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = reg.predict(input_data_reshaped)
print('\n')
print("The Insurance Cost is ",prediction)
