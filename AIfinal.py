#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.svm import SVR
import sklearn.svm as svm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import preprocessing
from sklearn import utils
from scipy import stats


# Read dataset from csv
column_names = [
    'Fran Datum Tid (UTC)', 'till', 'day', 'temperature', 'Kvalitet', 'Tidsutsnitt:', 'Unnamed: 5']
column_names_used = [
    'Fran Datum Tid (UTC)', 'till', 'day']


def make_numeric_values(arr, title):
    new_arr = []
    for date in arr[title]:
        new_date = make_date(date)
        new_arr.append(new_date)
    arr[title] = new_arr

def fix_array(arr):
    for name in column_names_used:
        make_numeric_values(arr, name)

def make_date(date):
    new_date = date.split(' ')
    new_date = new_date[0]
    new_date = new_date.split('-')
    new_number = ''
    first = True
    for number in new_date:
        if first:
            first = False
        else:
            new_number = new_number + number
    return new_number

def convert_date_to_string(plus_days):
    date = datetime.datetime.today() + timedelta(days=plus_days)
    date = date.strftime("%Y-%m-%d %H:%M:%S") 
    date = date.split(' ')
    date = date[0]
    date = date.split('-')
    date = date[1]+date[2]
    return date


data1 = pd.read_csv(r"C:\Users\akhil\Downloads\Archive_data.csv", sep=';', skiprows=3607, names=column_names)
    #print data1
    #sys.exit()
data2 = pd.read_csv(r"C:\Users\akhil\Downloads\Latest_data.csv", sep=';', skiprows=15, names=column_names)
data1 = data2.append(data1)
data1 = data1.drop('Tidsutsnitt:', axis=1)
X = data1.drop(["temperature"], axis=1)
X = X.drop(['Kvalitet'], axis = 1)
X = X.drop(['Unnamed: 5'], axis = 1)
fix_array(X)

y = data1['temperature']
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=4)


# In[5]:


train_X.shape


# In[6]:



train_y.shape


# In[7]:



train_y.head()


# In[21]:


data1.isnull().sum()


# In[29]:


df_new = data1[(np.abs(stats.zscore(data1.temperature)) < 3)]


# In[26]:


data1.head()


# In[27]:


data1.describe()


# In[ ]:





# In[ ]:





# In[ ]:


#multiple linear regression


# In[17]:


model=LinearRegression()
model.fit(train_X,train_y)
prediction = model.predict(test_X)
     
     


# In[ ]:





# In[18]:



#calculating error
np.mean(np.absolute(prediction-test_y))


# In[19]:


print('Variance score: %.2f' % model.score(test_X, test_y))


# In[20]:


for i in range(len(prediction)):
  prediction[i]=round(prediction[i],2)
pd.DataFrame({'Actual':test_y,'Prediction':prediction,'diff':(test_y-prediction)})


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:



# Print samples after running train_test_split
print("X_train: {}, Y_train: {}".format(len(X_train), len(X_test)))
print("X_train: {}, Y_train: {}".format(len(y_train), len(y_test)))

print("\n")

# Decision Tree Classifier Model setup after parameter tuning
# create a regressor object
regressor = DecisionTreeRegressor(random_state = 0) 
  
# fit the regressor with X and Y data
regressor.fit(X, y)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)
     



#regressor = DecisionTreeRegressor(random_state = 0) 
  
# fit the regressor with X and Y data
#regressor.fit(X, y)
# Print results to evaluate model
#print("Showing Performance Metrics for Decision Tree Classifier\n")

#print ("Training Accuracy: {}".format(model.score(X_train, y_train)))
predicted = regressor.predict(X_test)
#print ("Testing Accuracy: {}".format(accuracy_score(y_test, predicted)))

#print("\n")
  


# In[ ]:



print("Cross Validation Accuracy: \n")
cv_accuracy = cross_val_score(estimator=regressor, X=X_train, y=y_train, cv=10)
print("Accuracy using 10 folds: ")
print(cv_accuracy)

print("\n")


# In[ ]:


print("Mean accuracy: {}".format(cv_accuracy.mean()))
print("Standard Deviation: {}".format(cv_accuracy.std()))

print("\n")


# In[ ]:


prediction2=regressor.predict(X_test)
np.mean(np.absolute(prediction2-y_test))


# In[ ]:


print('Variance score: %.2f' % regressor.score(X_test, y_test))


# In[ ]:


for i in range(len(prediction2)):
  prediction2[i]=round(prediction2[i],2)
pd.DataFrame({'Actual':y_test,'Prediction':prediction2,'diff':(y_test-prediction2)})


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




