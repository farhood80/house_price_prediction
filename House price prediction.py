#!/usr/bin/env python
# coding: utf-8

# <b>project name : 
#     Boston House price prediction <b>
#     

# import dependencies

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error


# <b>Importing the Dataset<b>

# In[67]:


house_price_dataset = sklearn.datasets.load_boston()


# In[68]:


print(house_price_dataset)


# In[69]:


# loading  the dataset as pandas Dataframe
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)


# In[70]:


# print first five rows of the the dataset
house_price_dataframe.head()


# In[71]:


#add the target column to DataFrame
house_price_dataframe ['price'] = house_price_dataset.target


# In[72]:


house_price_dataframe.head()


# In[73]:


#checking the number of columns and rows in the dataframe
house_price_dataframe.shape


# In[74]:


#check for missing values
house_price_dataframe.isnull().sum() # or house_price_dataframe.isnull()


# In[75]:


#statical measures of the dataset
house_price_dataframe.describe()


# <b> the correlation between various features in the  dataset
#     
#   <B>  trying to find positive correlation between all values

# In[76]:


correlation = house_price_dataframe.corr()


# In[77]:


#construction a heatmap to understand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar = True, square = True, fmt ='.1f', annot = True, annot_kws={'size':8}, cmap = 'Greens')


# <B> spiltin the data and target

# In[78]:


x = house_price_dataframe.drop(['price'], axis =1)
y = house_price_dataframe['price']


# In[79]:


print(x)
print(y)


# <B> spiltin th data into training and test data

# In[80]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 2)


# In[81]:


print(x.shape,x_train.shape, x_test.shape)


# <b> Model training
#     
# <b> XGBRegressor

# In[82]:


#loading the model
model = XGBRegressor()


# In[83]:


#training model with x_train
model.fit(x_train, y_train)


# <b> prediction on training data

# In[84]:


# accuracy for the prediction on training data
training_data_prediction = model.predict(x_train)


# In[85]:


print(training_data_prediction)


# In[86]:


# R squared error
score_1 = metrics.r2_score(y_train, training_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(y_train, training_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)


# <b>Visualizing the actual Prices and predicted prices

# In[87]:


plt.scatter(y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Preicted Price")
plt.show()


# <b> prediction on test data

# In[88]:


test_data_prediction = model.predict(x_test)


# In[89]:


# R squared error
score_1 = metrics.r2_score(y_test, test_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(y_test, test_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)


# In[ ]:




