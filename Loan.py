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
#import sklearn.preprocessing.LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# In[2]:



df=pd.read_csv("Loan.csv")
print(df)


# In[3]:


df.head(10)


# In[4]:


#Number of rows and columns
df.shape


# In[5]:


#statistical measeures
df.describe()


# In[6]:


#number of missing value in each column
df.isnull().sum()

we can identifying by this null metthod how many missing value.
not that muach missing value is there we have to drop the missing value
# In[7]:


# wereplace the nan value with mean
df = df.replace(np.nan,df.mean())


# In[8]:


df.isnull().sum()


# In[9]:


df= df.dropna()


# In[10]:


df.isnull().sum()


# In[11]:


# label encoding 
# we are using label encoding for string value into int.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == np.number:
        continue
    df[column] = LabelEncoder().fit_transform(df[column])


# In[12]:


df


# In[13]:


df['Dependents'].value_counts()


# In[14]:



#Data vicualization
sns.countplot(x="Education", hue="Loan_Status",data=df)


# In[15]:


sns.countplot(x="Gender", hue="Loan_Status",data=df)


# In[16]:


sns.countplot(x="Married", hue="Loan_Status",data=df)


# In[17]:


sns.countplot(x="Dependents", hue="Loan_Status",data=df)


# In[18]:


sns.countplot(x="Self_Employed", hue="Loan_Status",data=df)


# In[19]:


sns.countplot(x="ApplicantIncome", hue="Loan_Status",data=df)


# In[20]:


sns.countplot(x="LoanAmount", hue="Loan_Status",data=df)


# In[21]:


sns.countplot(x="Loan_Amount_Term", hue="Loan_Status",data=df)


# In[22]:


sns.countplot(x="Credit_History", hue="Loan_Status",data=df)


# In[23]:


sns.countplot(x="Property_Area", hue="Loan_Status",data=df)


# In[24]:


#Droping unused data from the daataset.

df.drop("Loan_ID",axis=1,inplace=True)


# In[25]:


df


# # Multivert

# In[26]:


df.plot(kind="kde",subplots=True,layout=(21,6), figsize=(40,50))


# In[27]:


df.skew()

Skewness is not that much so need to make data
# # Plotting Outlier

# In[28]:


df.columns


# In[29]:


df['Gender'].plot.box() 


# In[30]:


df['Married'].plot.box()


# In[31]:


df['Dependents'].plot.box() 


# In[32]:


df['Education'].plot.box() 


# In[33]:


df['CoapplicantIncome'].plot.box() 


# In[34]:


df['LoanAmount'].plot.box()


# In[35]:


df['Loan_Amount_Term'].plot.box() 


# In[36]:


df['Credit_History'].plot.box() 


# In[37]:


df['Property_Area'].plot.box()


# # Remove Outlier

# In[38]:


from scipy.stats import zscore 
z=np.abs(zscore(df))
z


# In[39]:


threshold=3
print(np.where(z>3))


# In[40]:


df_new = df[(z<3).all(axis=1)]


# In[41]:


df_new.head()


# In[42]:


x=df.iloc[:,: -1]


# In[44]:


x.head()


# In[45]:


y=df.iloc[:,-1]


# In[46]:


y.head()


# In[47]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=30, random_state= 0)


# In[48]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators= 10, criterion = 'entropy', random_state = 0)
forest.fit(x_train, y_train)


# In[49]:


forest.score(x_train,y_train)


# In[50]:


from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[51]:


mnb = MultinomialNB()
mnb.fit(x_train,y_train)
predmnb = mnb.predict(x_test)
print(accuracy_score(y_test,predmnb))
print(confusion_matrix(y_test,predmnb))
print(classification_report(y_test,predmnb))


# In[52]:


svc = SVC(kernel='rbf') # kernel = 'rbf is -----by defult'
svc.fit(x_train,y_train)
svc.score(x_train,y_train)
presvc = svc.predict(x_test)
print(accuracy_score(y_test,presvc))
print(confusion_matrix(y_test,presvc))
print(classification_report(y_test,presvc))


# In[53]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
knn.score(x_train,y_train)
predknn=knn.predict(x_test)
print(accuracy_score(y_test,predknn))
print(confusion_matrix(y_test,predknn))
print(classification_report(y_test,predknn))

conclusion that RandomForestClassifier has accuracy is more then 90 percent.
Random Forest classifier has good accuracy then DecisionTreeClassifier,SVC,KNN
# In[ ]:




