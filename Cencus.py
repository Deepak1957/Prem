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


df=pd.read_csv("Cencus.csv")
print(df)


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.nunique()

Ther we are able to see the unique value of evey attribute
# In[7]:


df.describe().T


# #Value count Function

# In[8]:


df.columns


# In[9]:


df['ge'].value_counts()


# In[10]:


df['Workclass'].value_counts()


# In[11]:


df['Fnlwgt'].value_counts()


# In[12]:


df['Education'].value_counts()


# In[13]:


df['Marital_status'].value_counts()


# In[14]:


df['Occupation'].value_counts()


# In[15]:


df['Relationship'].value_counts()


# In[16]:


df['Race'].value_counts()


# In[17]:


df['Capital_gain'].value_counts()


# In[18]:


df['Hours_per_week'].value_counts()


# In[19]:


df['Native_country'].value_counts()


# In[20]:


df.columns


# In[21]:


#Data vicualization
sns.countplot(df["Income"],palette="coolwarm",hue="Sex",data=df)


# In[22]:


sns.countplot(df["Income"],palette="coolwarm",hue="Native_country",data=df)


# In[23]:


sns.countplot(df["Income"],palette="coolwarm",hue="Hours_per_week",data=df)


# In[24]:


sns.countplot(df["Income"],palette="coolwarm",hue="Capital_loss",data=df)


# In[25]:


sns.countplot(df["Income"],palette="coolwarm",hue="Capital_gain",data=df)


# In[26]:


sns.countplot(df["Income"],palette="coolwarm",hue="Relationship",data=df)


# In[27]:


sns.countplot(df["Income"],palette="coolwarm",hue="Occupation",data=df)


# In[28]:


sns.countplot(df["Income"],palette="coolwarm",hue="Marital_status",data=df)


# In[ ]:





# # Filling ? Values

# In[30]:


#replace the values with mode.
#Most frequently coming string is replace to the ?


# In[31]:


df['Workclass']= df['Workclass'].replace("?","Private")
df['Occupation']= df['Occupation'].replace("?","proof.specialty")
df['Native_country']= df['Native_country'].replace("?","United.States")


# In[32]:


df.head()


# In[33]:


df["Marital_status"].value_counts()


# In[34]:


df["Education"].value_counts()


# In[35]:


#Label encoder use for convert the string value into numberi value.
#it made the data ready for m/l

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == np.number:
        continue
    df[column] = LabelEncoder().fit_transform(df[column])


# In[36]:


df.head(9)


# In[37]:


#Corretion of the numberic data


# In[38]:


df.corr()


# In[39]:


sns.heatmap(df.corr(),annot=True);


# #Histogram

# In[40]:


df.hist(figsize=(30,30),layout=(12,3),sharex=False)


# # Boxplot

# In[41]:


df.plot(kind="box",figsize=(30,30),layout=(12,3),sharex=False,subplots=True)


# In[42]:


sns.countplot(df['Race'],hue='Income',data=df,palette='Blues')


# In[43]:


sns.countplot(df['Workclass'],hue='Income',data=df,palette='Blues')


# In[ ]:


sns.countplot(df['Fnlwgt'],hue='Income',data=df,palette='Blues')


# In[ ]:


sns.countplot(df['Education'],hue='Income',data=df,palette='Blues')


# In[ ]:


sns.countplot(df['Education_num '],hue='Income',data=df,palette='Blues')


# In[ ]:


sns.countplot(df['Marital_status'],hue='Income',data=df,palette='Blues')


# In[ ]:


sns.countplot(df['Occupation'],hue='Income',data=df,palette='Blues')


# In[ ]:


sns.countplot(df['Relationship '],hue='Income',data=df,palette='Blues')


# In[ ]:


sns.countplot(df['Sex '],hue='Income',data=df,palette='Blues')


# In[ ]:


sns.countplot(df['Capital_gain'],hue='Income',data=df,palette='Blues')


# In[ ]:


sns.countplot(df['Capital_loss'],hue='Income',data=df,palette='Blues')


# In[ ]:


sns.countplot(df['Hours_per_week'],hue='Income',data=df,palette='Blues')


# In[ ]:


sns.countplot(df['Native_country'],hue='Income',data=df,palette='Blues')


# In[48]:


sns.countplot(df['Income'],hue='Income',data=df,palette='Blues')


# In[49]:


df.head()


# In[52]:


x=df.iloc[:,: -1]


# In[53]:


x.head()


# In[54]:


y=df.iloc[:,-1]


# In[55]:


y.head()

Spilt the data into x and y taining and tesing.
making ready for machine learning
# In[58]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=30, random_state= 0)


# # Logistic Regression

# In[59]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression()

model = lr.fit(x_train,y_train)
prediction = model.predict(x_test)



# In[60]:


score = lr.score(x_test, y_test)
print(score)


# # Random forest classifier

# In[61]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators= 10, criterion = 'entropy', random_state = 0)
forest.fit(x_train, y_train)


# In[62]:


forest.score(x_train,y_train)


# In[63]:


from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[64]:


mnb = MultinomialNB()
mnb.fit(x_train,y_train)
predmnb = mnb.predict(x_test)
print(accuracy_score(y_test,predmnb))
print(confusion_matrix(y_test,predmnb))
print(classification_report(y_test,predmnb))


# In[66]:


svc = SVC(kernel='rbf') # kernel = 'rbf is -----by defult'
svc.fit(x_train,y_train)
svc.score(x_train,y_train)
presvc = svc.predict(x_test)
print(accuracy_score(y_test,presvc))
print(confusion_matrix(y_test,presvc))
print(classification_report(y_test,presvc))


# In[67]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
knn.score(x_train,y_train)
predknn=knn.predict(x_test)
print(accuracy_score(y_test,predknn))
print(confusion_matrix(y_test,predknn))
print(classification_report(y_test,predknn))

RandomForestClassifier has good score.
98 percetage.
compare to other alogrithm of machin learning