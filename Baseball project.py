#!/usr/bin/env python
# coding: utf-8

# In[133]:


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


# In[134]:


df=pd.read_csv("Baseball.csv")
print(df)


# In[135]:


df.head()

BY using head method we are able to see the most upper data.
by which we can analize to the data.
# In[136]:


df.shape

30 rows 17 columns as per analize
# In[137]:


df.columns

we are able to identifying the colunms are available in given data
# In[138]:


df.dtypes

To see the data type in the given data.
# In[139]:


df.info()

By .info we get to know
1.how many rows and coluns are present.
2.Data type of values.
3.Null value has been present or not easy to know.

# # Summary Statistics

# In[140]:


df.describe()

mean and standard deviation diffrence is so high that means data has been spread to high
# In[141]:


df.W.unique()


# In[142]:


df.W.value_counts()


# # Data Visualization

# In[143]:


sns.heatmap(df.isnull())


# # To check correlation

# In[144]:


dfcor = df.corr()
dfcor


# In[145]:


sns.heatmap(dfcor)


# In[146]:


plt.figure(figsize=(16,14))
sns.heatmap(dfcor,cmap='Blues',annot=True)


# In[147]:


df.skew()

not that much skew is present in the given data.
# # Plotting Outlier

# In[148]:


df.columns


# In[149]:


df['W'].plot.box()


# In[150]:


df['R'].plot.box()


# In[151]:


df['AB'].plot.box()


# In[152]:


df["H"].plot.box()


# In[153]:



#df['2B'].plot.box
df["2B"].plot(kind = 'box')


# In[154]:



df["3B"].plot(kind = 'box')


# In[155]:


df["HR"].plot(kind = 'box')


# In[156]:


df["BB"].plot(kind = 'box')


# In[157]:


df["SO"].plot(kind = 'box')


# In[158]:


df["SB"].plot(kind = 'box')


# In[159]:


df["RA"].plot(kind = 'box')


# In[160]:


df["ER"].plot(kind = 'box')


# In[161]:


df["ERA"].plot(kind = 'box')


# In[162]:


df["CG"].plot(kind = 'box')


# In[163]:


df["SHO"].plot(kind = 'box')


# In[164]:


df["E"].plot(kind = 'box')


# In[165]:


df.columns


# In[223]:


df.skew()


# In[166]:


sns.distplot(df["W"])


# In[167]:


sns.distplot(df['R'])


# In[168]:


sns.distplot(df['AB'])


# In[169]:


sns.distplot(df['H'])


# In[170]:


sns.distplot(df['2B'])


# In[171]:


sns.distplot(df['3B'])


# In[172]:


sns.distplot(df['HR'])


# In[173]:


sns.distplot(df['BB'])


# In[174]:


sns.distplot(df['SO'])


# In[175]:


sns.distplot(df['SB'])


# In[176]:


sns.distplot(df['SB'])


# In[177]:


sns.distplot(df['RA'])


# In[178]:


sns.distplot(df['ER'])


# In[179]:


sns.distplot(df['ERA'])


# In[180]:


sns.distplot(df['CG'])


# In[181]:


sns.distplot(df['SHO'])


# In[182]:


sns.distplot(df['SV'])


# In[183]:


sns.distplot(df['E'])


# In[184]:


df.plot(kind="kde",subplots=True,layout=(21,6), figsize=(40,50))


# # bivariate analysis

# In[185]:


plt.scatter(df['R'],df['W'])


# In[186]:


plt.plot.box(df['R'],df['W'])
#df['R'].plot.box()


# In[ ]:


sns.boxplot(x='W',y='R', data=df)
plt.show()


# In[ ]:


sns.violinplot(x="W",y="AB", data=df, size=100)


# In[ ]:


sns.violinplot(x="W",y="H", data=df, size=100)








# In[ ]:


sns.violinplot(x="W",y="2B", data=df, size=100)


# In[ ]:


sns.violinplot(x="W",y="3B", data=df, size=100)


# In[ ]:


sns.violinplot(x="W",y="HR", data=df, size=100)


# In[ ]:


sns.violinplot(x="W",y="SO", data=df, size=100)


# In[ ]:


sns.violinplot(x="W",y="SB", data=df, size=100)


# In[ ]:


sns.violinplot(x="W",y="RA", data=df, size=100)


# In[ ]:


sns.violinplot(x="W",y="ER", data=df, size=100)


# In[ ]:


sns.violinplot(x="W",y="ERA", data=df, size=100)


# In[ ]:


sns.pairplot(df)


# # Removing Outlier

# In[ ]:


df.drop("E",axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


from scipy.stats import zscore
z=np.abs(zscore(df))
z


# In[ ]:


threshold=3
print(np.where(z>3))


# In[ ]:


df_new = df[(z<3).all(axis=1)]


# In[ ]:


df_new


# In[187]:


df.shape


# In[188]:


df_new.shape


# In[189]:


1/30

We are lossing only less than 1 % so we are going with this Z score.
if we are lossing data more than 9 % then we must to try IQR method.
if we are lossing data more than 15 % in that case we are going to remove outlier.
# In[261]:


y=y=df_new.iloc[:,: 1]
y.head()
#y.shape


# In[234]:


#x=df.iloc[0: -1]
#x.head()

x=df_new.iloc[: , 1:]
x.head()

For find out the target 
# In[235]:


y.shape


# In[236]:


x.shape


# In[237]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split


# In[238]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.33,random_state=42)

test size .33 means we are taking test size 30 percentage of the data.
random state 42 means we are taking 40 percentage data randamly
# In[239]:


x_train.shape


# In[240]:


y_train.shape


# In[241]:


x_test.shape


# In[242]:


x_train.shape


# In[243]:


lm = LinearRegression()


# In[244]:


lm.fit(x_train,y_train)


# In[245]:


lm.coef_


# In[246]:


lm.intercept_


# In[247]:


lm.score(x_train,y_train)


# # For pridect the Win 

# In[248]:


pred=lm.predict(x_test)
print ("predict the result Win:",pred)
print("actual Win",y_test)


# In[249]:


print('error')
print('mean absoulte error', mean_absolute_error(y_test,pred))
print("mean squared error",mean_squared_error(y_test,pred))

print("root mean squared error",np.sqrt(mean_squared_error(y_test,pred)))


# In[250]:


from sklearn.metrics import r2_score
print(r2_score(y_test,pred))


# In[256]:


t = np.array([40,50,10,12,18,16,17,19,10,14,11,12,13,14,15])


# In[257]:


t.shape


# In[258]:


t=t.reshape(1,-1)
t.shape


# In[259]:


lm.predict(t)


# In[ ]:





# In[ ]:




