#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


df=pd.read_csv("Power.csv")
print(df)


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().values.any()


# In[7]:


df.isnull().sum()


# In[8]:


df.dtypes


# In[9]:


df.drop("country",axis=1,inplace=True)


# In[10]:


df.drop("country_long",axis=1,inplace=True)


# In[11]:


df.drop("name",axis=1,inplace=True)


# In[12]:


df.drop("gppd_idnr",axis=1,inplace=True)


# In[13]:


df.drop("primary_fuel",axis=1,inplace=True)


# In[14]:


df.drop("other_fuel1",axis=1,inplace=True)


# In[15]:


df.drop("other_fuel2",axis=1,inplace=True)


# In[16]:


df.drop("owner",axis=1,inplace=True)


# In[17]:


df.drop("url",axis=1,inplace=True)


# In[18]:


df.drop("source",axis=1,inplace=True)


# In[19]:


df.drop("geolocation_source",axis=1,inplace=True)


# In[20]:


df.drop("generation_data_source",axis=1,inplace=True)


# In[21]:


df.head()


# In[22]:


df.isnull().sum()


# In[23]:


#We drop the column from the data frame those are not in useless data.

df.drop("other_fuel3",axis=1,inplace=True)


# In[24]:


df.drop("wepp_id",axis=1,inplace=True)


# In[25]:


df.drop("estimated_generation_gwh",axis=1,inplace=True)


# In[26]:


df.head()


# In[27]:


df = df.replace(np.nan,df.mean())

#We replace the nan value with mean value in whole data frame.
#By using this replace method


# In[28]:


df.head()


# In[29]:


df.isnull().sum()


# In[30]:


df.describe()

capacity_mw:-
mean value  321.046378
std dev. of age is 580.221767
diffrence is soo high hence skwness will be high.
75 % and max value has high difrence therefore outlier is present.

latitude:-
mean value 21.19618
std dev. of age is 6.088110
diffrence is soo high hence skwness will be high.
75 % and max value has not high difrence therefore outlier is present but we can neglate

longitude:-
mean value 77.447848
std dev. of age is 4.024165	
diffrence is not that much hence skwness is not there.
75 % and max value has not diffrence therefore outlier is present but we can neglate

# In[31]:


# label encoding 
# we are using label encoding for string value into int.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == np.number:
        continue
    df[column] = LabelEncoder().fit_transform(df[column])


# In[32]:


df.head()


# # Univerate

# In[33]:


df.columns


# In[34]:


#How many time that capacity_mw count in the data frame.
df['capacity_mw'].value_counts()


# In[35]:


df['latitude'].value_counts()


# In[36]:


df['longitude'].value_counts()


# In[37]:


df['commissioning_year'].value_counts()


# In[38]:


df['year_of_capacity_data'].value_counts()


# In[39]:


df['generation_gwh_2013'].value_counts()


# In[40]:


df['generation_gwh_2014'].value_counts()


# In[41]:


df['generation_gwh_2015'].value_counts()


# In[42]:


df['generation_gwh_2016'].value_counts()


# In[43]:


df['generation_gwh_2017'].value_counts()


# In[44]:


df.skew()


# In[ ]:


#Data vicualization
sns.countplot(df["latitude"],palette="coolwarm",hue="capacity_mw",data=df)


# In[ ]:


sns.countplot(df["longitude"],palette="coolwarm",hue="capacity_mw",data=df)


# In[ ]:


sns.countplot(df["commissioning_year"],palette="coolwarm",hue="capacity_mw",data=df)


# In[ ]:


sns.countplot(df["year_of_capacity_data"],palette="coolwarm",hue="capacity_mw",data=df)


# In[ ]:


sns.countplot(df["generation_gwh_2013"],palette="coolwarm",hue="capacity_mw",data=df)


# In[ ]:


sns.countplot(df["generation_gwh_2014"],palette="coolwarm",hue="capacity_mw",data=df)


# In[ ]:


sns.countplot(df["generation_gwh_2015"],palette="coolwarm",hue="capacity_mw",data=df)


# In[ ]:


sns.countplot(df["generation_gwh_2016"],palette="coolwarm",hue="capacity_mw",data=df)


# In[ ]:


sns.countplot(df["generation_gwh_2017"],palette="coolwarm",hue="capacity_mw",data=df)


# In[ ]:


#Corretion of the numberic data


# In[ ]:


sns.heatmap(df.corr(),annot=True);


# In[ ]:


df.hist(figsize=(30,30),layout=(22,3),sharex=False)


# # Boxplot

# In[46]:


df.plot(kind="box",figsize=(30,30),layout=(12,3),sharex=False,subplots=True)

Oulier is present in the data set
# In[47]:


sns.distplot(df["longitude"]) 


# In[48]:


sns.distplot(df["commissioning_year"]) 


# In[49]:


sns.distplot(df["year_of_capacity_data"]) 


# In[50]:


sns.distplot(df["generation_gwh_2013"]) 


# In[51]:


sns.distplot(df["generation_gwh_2014"]) 


# In[52]:


sns.distplot(df["generation_gwh_2015"]) 


# In[53]:


sns.distplot(df["generation_gwh_2016"]) 


# In[54]:


sns.distplot(df["generation_gwh_2017"]) 


# In[55]:


sns.distplot(df["longitude"]) 


# In[56]:


sns.distplot(df["longitude"]) 


# In[57]:


sns.distplot(df["longitude"]) 


# In[58]:


sns.distplot(df["longitude"]) 


# In[59]:


sns.distplot(df["longitude"]) 


# # Remove outlier

# In[60]:


from scipy.stats import zscore 
z=np.abs(zscore(df))
z


# In[61]:


threshold=3
print(np.where(z>3))


# In[62]:


df_new = df[(z<3).all(axis=1)]


# In[63]:


df_new.shape


# In[64]:


df.shape


# In[65]:


df.head()


# In[66]:


y=df.iloc[:,: 1]
#y=df_new[1:,:]


# In[67]:


y.head()


# In[68]:


x=df.iloc[: , 1:]
x.head()


# In[98]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split


# In[99]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.33,random_state=42)

split the data from x and y train and test 
33 percet is test.
70 percent is train
random state = 42
# In[100]:


x_train.shape # y train rows and columns


# In[101]:


y_train.shape  # y train rows and columns


# In[102]:


lm = LinearRegression() # linearRegression is stored in the lm


# In[103]:


lm.fit(x_train,y_train) 


# In[104]:


lm.coef_


# In[105]:


lm.intercept_ # we checked the intercept line on it.


# In[106]:


lm.score(x_train,y_train) # we are checking the score of the data set 


# # Decision Tree Regressor

# In[107]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

regressor = DecisionTreeRegressor()
regressor.fit(x_train, y_train)


# In[108]:


pred=lm.predict(x_test)
print(r2_score(y_test,pred))
y_pred = regressor.predict(x_test)


# In[109]:


print(r2_score(y_test,pred))


# # Support vector regression

# In[116]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train, y_train)


# In[117]:


y_pred = regressor.predict(x_test)


# In[118]:


print(r2_score(y_test,pred))


# In[ ]:





# In[119]:


from sklearn.ensemble import RandomForestRegressor
regressor1 = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor1.fit(x_train, y_train)


# In[120]:


y_pred = regressor1.predict(x_test)


# In[121]:


print(r2_score(y_test,pred))


# In[ ]:





# In[ ]:




