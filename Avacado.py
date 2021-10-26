#!/usr/bin/env python
# coding: utf-8

# In[120]:


import pandas as pd
#import pandas.plotting._core.PlotAccessor
import pandas.plotting as  pl

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[121]:


df=pd.read_csv("A.csv")
print(df)

We call data and store in the dataframe is df
# In[122]:


df.columns # by this we are able to find out the columns


# In[ ]:





# In[123]:


df.head() # by calling head we can see top data fro data set.


# In[124]:


df.drop("Unnamed: 0", axis = 1, inplace = True) # we are droping one attribute from the data set. 


# In[125]:


df.head()

BY using head method we are able to see the most upper data.
by which we can analize to the data.
# In[126]:


df.info()

By .info we get to know
1.how many rows and coluns are present.
2.Data type of values.
3.Null value has been present or not easy to know.

# In[127]:


df.shape# by shape we can able to see the shape of the date how many row and columns are available


# In[128]:


df.columns


# In[129]:


df.dtypes # by using type we are able to find out the type of the data.


# In[130]:


df.drop("Date", axis = 1, inplace = True)# we are droping unused data from data set


# In[131]:


df.drop("region", axis = 1, inplace = True)


# In[132]:


df.drop("type", axis = 1, inplace = True)


# In[133]:


df.drop("year", axis = 1, inplace = True)


# In[134]:


df.head()


# In[135]:


df.shape


# # Summary Statistics

# In[136]:


df.describe() # by this method we are able to see mean std deviation min valuemax value

mean and standard deviation diffrence is not that much high so skwness is less or neglagiable.
but the difrence between max and 75 % difrecncess is there in some entity so need 
# In[137]:


df.AveragePrice.unique()


# In[138]:


df.AveragePrice.value_counts()# we are able to count our data, how many time it has been come in our data set


# # Data Visualization

# In[139]:


sns.heatmap(df.isnull())

Null value is not there in this data set. Heatmap is showing no value has been missed.
# # To check correlation

# In[140]:


dfcor = df.corr() # by which we are able to find out correlation value through which we are able to identify the correlation between data
dfcor


# In[141]:


sns.heatmap(dfcor)


# In[142]:


plt.figure(figsize=(16,14))
sns.heatmap(dfcor,cmap='Blues',annot=True)


# In[143]:


df.skew() # we can see the skewnwws in the data

Skewness is not that much so need to make data
# # Plotting Outlier

# In[144]:


df.columns


# In[145]:


df['AveragePrice'].plot.box() # by this we can easly identify the outiler is present or not
                               # like in this box plot outlier is present


# In[146]:


df['4046'].plot.box()


# In[147]:


df['4225'].plot.box()


# In[148]:


df['Total Bags'].plot.box()


# In[149]:


df['Small Bags'].plot.box()


# In[150]:


df['Large Bags'].plot.box()


# In[151]:


df.columns


# In[152]:


df["AveragePrice"].plot(kind = 'box')


# In[153]:


df["Total Volume"].plot(kind = 'box')


# In[154]:


df["4046"].plot(kind = 'box')


# In[155]:


df["4225"].plot(kind = 'box')


# In[156]:


df["4770"].plot(kind = 'box')


# In[157]:


df["Total Bags"].plot(kind = 'box')


# In[158]:


df["Large Bags"].plot(kind = 'box')


# In[159]:


df["Small Bags"].plot(kind = 'box')


# In[ ]:


df.columns

#BY sns distplot we are able to see the skwness is there or not
# In[212]:


sns.distplot(df["AveragePrice"]) 


# In[213]:


sns.distplot(df["Total Volume"])


# In[214]:


sns.distplot(df["4046"])


# In[215]:


sns.distplot(df["4225"])


# In[216]:


sns.distplot(df["4770"])


# In[217]:


sns.distplot(df["Total Bags"])


# In[218]:


sns.distplot(df["Small Bags"])


# In[219]:


sns.distplot(df["Large Bags"])


# In[220]:


sns.distplot(df["XLarge Bags"])


# In[170]:


#Multivert


# In[171]:


df.plot(kind="kde",subplots=True,layout=(21,6), figsize=(40,50))


# # Remove Outlier

# In[172]:


from scipy.stats import zscore 
z=np.abs(zscore(df))
z


# In[173]:


threshold=3
print(np.where(z>3))


# In[174]:


df_new = df[(z<3).all(axis=1)]


# In[175]:


df_new.head()


# In[176]:


df.shape


# In[177]:


df_new.shape


# In[178]:


1517-1436


# In[179]:


81/1517

We are lossing only less than 1 % so we are going with this Z score.
if we are lossing data more than 9 % then we must to try IQR method.
if we are lossing data more than 15 % in that case we are going to remove outlier.
# In[180]:


y=df_new.iloc[:,: 1]
#y=df_new[1:,:]


# In[181]:


y


# In[182]:


x=df_new.iloc[: , 1:]
x.head()

For find out the target 
# In[183]:


y.shape


# In[184]:


x.shape


# In[185]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split


# In[186]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.33,random_state=42)

split the data from x and y train and test 
33 percet is test.
70 percent is train
random state = 42
# In[187]:


x_train.shape # y train rows and columns


# In[188]:


y_train.shape  # y train rows and columns


# In[189]:


x_test.shape  # y test rows and columns


# In[190]:


x_train.shape


# In[191]:


lm = LinearRegression() # linearRegression is stored in the lm


# In[192]:


lm.fit(x_train,y_train) 


# In[193]:


lm.coef_


# In[194]:


lm.intercept_ # we checked the intercept line on it.


# In[195]:


lm.score(x_train,y_train) # we are checking the score of the data set 


# # For pridect the Win 

# In[196]:


pred=lm.predict(x_test)
print ("predict the result Win:",pred)
print("actual Win",y_test)


# In[197]:


print('error')
print('mean absoulte error', mean_absolute_error(y_test,pred))
print("mean squared error",mean_squared_error(y_test,pred))

print("root mean squared error",np.sqrt(mean_squared_error(y_test,pred)))


# In[198]:


from sklearn.metrics import r2_score
print(r2_score(y_test,pred))


# In[199]:


t = np.array([40,50,10,12,18,16,17,19])


# In[200]:


t.shape


# In[201]:


t=t.reshape(1,-1)
t.shape


# In[202]:


lm.predict(t)


# # Decision Tree Regressor

# In[203]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x_train, y_train)


# In[204]:


y_pred = regressor.predict(x_test)


# In[205]:


print(r2_score(y_test,pred))

only 21 percent accuracy score is not good score.
# # Support vector regression

# In[221]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train, y_train)


# In[222]:


y_pred = regressor.predict(x_test)


# In[223]:


print(r2_score(y_test,pred))

only 21 percent accuracy score is not good score.
# # Classification
Classification method we are using
data set we can do by two method

from my opinon this data is continous data so, we cant't use classification method
# In[209]:


from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[211]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
knn.score(x_train,y_train)
predknn=knn.predict(x_test)
print(accuracy_score(y_test,predknn))
print(confusion_matrix(y_test,predknn))
print(classification_report(y_test,predknn))


# In[ ]:





# In[ ]:




