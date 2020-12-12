#!/usr/bin/env python
# coding: utf-8

# Importing libraries

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Reading Data

# In[3]:


df = pd.read_excel('Data_Train.xlsx')


# Exploring Data

# In[4]:


df.head()


# In[5]:


df.describe()


# Let's Check for NULL values

# In[6]:


df.isnull().sum()


# In[7]:


df.dropna(axis=0, inplace = True)


# In[8]:


df.isnull().sum()


# Data Cleaning

# In[9]:


df.drop(['Route', 'Additional_Info'], axis = 1, inplace = True)


# In[10]:


df.head()


# In[11]:


df['Total_Stops'].value_counts()


# In[12]:


df['Airline'].value_counts()


# In[13]:


df['Airline'].value_counts()


# In[14]:


df.info()


# In[15]:


plt.figure(figsize=(30,5))
sns.barplot(x="Airline",y="Price",data=df)


# In[16]:


df['Total_Stops'] = df['Total_Stops'].replace(['non-stop','1 stop', '2 stops', '3 stops', '4 stops'], [0, 1, 2, 3, 4])


# In[17]:


df.head()


# In[18]:


airline = pd.get_dummies(df['Airline'], drop_first = True)
destination = pd.get_dummies(df['Destination'], drop_first = True)


# In[19]:


df = pd.concat([df, airline, destination], axis = 1)


# In[20]:


df.head()


# In[21]:


df.rename(columns = {'Kolkata':'Kolkata_dest', 'Delhi':'Delhi_dest'}, inplace = True)


# In[22]:


source = pd.get_dummies(df['Source'], drop_first = True)
df = pd.concat([df, source], axis = 1)
df.rename(columns = {'Kolkata':'Kolkata_source', 'Delhi':'Delhi_source'}, inplace = True)


# In[23]:


df.drop(['Airline', 'Source', 'Destination'], axis = 1, inplace = True)


# In[24]:


df.head()


# In[25]:


y = df['Price']


# In[26]:


df.drop(['Price'], axis = 1, inplace = True)


# In[27]:


df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'])
df.head()


# In[28]:


df['Day_of_Journey'] = df['Date_of_Journey'].dt.dayofweek


# In[29]:


df.head()


# In[30]:


df['month']=df['Date_of_Journey'].dt.month
df['date']=df['Date_of_Journey'].dt.day


# In[31]:


df['dep_time'] = pd.to_datetime(df['Dep_Time'], format = '%H:%M')
df['Hour_dep'] = df['dep_time'].dt.hour 
df['minute_dep'] = df['dep_time'].dt.minute 


# In[32]:


df.drop(['Dep_Time', 'dep_time', 'Date_of_Journey'], axis = 1, inplace = True)


# In[66]:


df.head()


# In[67]:


df.columns


# In[68]:


duration = list(df['Duration'])
for i in range(len(duration)) :
    if len(duration[i].split()) != 2:
        if 'h' in duration[i] :
            duration[i] = duration[i].strip() + ' 0m'
        elif 'm' in duration[i] :
            duration[i] = '0h {}'.format(duration[i].strip())
dur_hours = []
dur_minutes = []  
 
for i in range(len(duration)) :
    dur_hours.append(int(duration[i].split()[0][:-1]))
    dur_minutes.append(int(duration[i].split()[1][:-1]))
     
 
df['Duration_hours'] = dur_hours
df['Duration_minutes'] =dur_minutes
df.loc[:,'Duration_hours'] *= 60
df['Duration_Total_mins']= df['Duration_hours']+df['Duration_minutes']


# In[69]:


df.drop(['Duration', 'Duration_hours', 'Duration_minutes'], axis = 1, inplace = True)


# In[70]:


df['arr_time'] = pd.to_datetime(df['Arrival_Time'])
df['Hour_arr'] = df['arr_time'].dt.hour 
df['minute_arr'] = df['arr_time'].dt.minute


# In[71]:


df.drop(['Arrival_Time', 'arr_time'], axis = 1, inplace = True)


# In[72]:


df.head()


# In[73]:


X = df[:]


# In[74]:


X.head()


# In[75]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[44]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


# In[45]:


rf_random.best_params_


# In[76]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators= 1000,
 min_samples_split= 10,
 min_samples_leaf= 1,
 max_features= 'sqrt',
 max_depth= 80,
 bootstrap= False)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[77]:


rf.score(X_test, y_test)


# In[78]:


from sklearn.metrics import accuracy_score,r2_score
rs=r2_score(y_test,y_pred)


# In[79]:


print(rs)


# In[80]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(rmse)


# In[82]:


print(1 - np.sqrt(np.square(np.log10(y_pred +1) - np.log10(y_test +1)).mean()))

