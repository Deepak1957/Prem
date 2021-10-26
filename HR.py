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


df = pd.read_excel('hr.xlsx')


# In[3]:


df


# In[6]:


df.head(10)


# In[7]:


df.shape


# In[9]:


df.dtypes


# In[11]:


#To check there has any null value or missing value is there.
df.isnull().values.any()


# In[10]:


#Check for any missing / null values in the data
df.isnull().sum()


# In[12]:


#view some statistics
df.describe()

Age:-
mean value 36.923810 
std dev. of age is 9.135373
diffrence is soo high hence skwness will be high.
75 % and max value has high difrence therefore outlier is present.

DailyRate:-
mean value 802.485714
std dev. of age is 403.509100
diffrence is soo high hence skwness will be high.
75 % and max value has high difrence therefore outlier is present.

Education:-
mean value 2.912925
std dev. of age is 1.024165	
diffrence is not that much hence skwness is not there.
75 % and max value has not diffrence therefore outlier is not present.



# # Univerate

# In[14]:


df['Attrition'].value_counts()


# In[17]:


sns.countplot(df['Attrition'])


# In[65]:


df['BusinessTravel'].value_counts()


# In[66]:


sns.countplot(df['BusinessTravel'])


# In[67]:


df['DailyRate'].value_counts()


# In[68]:


sns.countplot(df['DailyRate'])


# In[69]:


df['DistanceFromHome'].value_counts()


# In[70]:


sns.countplot(df['DistanceFromHome'])


# In[71]:


df['Education'].value_counts()


# In[72]:


sns.countplot(df['Education'])


# In[73]:


df['DailyRate'].value_counts()


# In[74]:


sns.countplot(df['DailyRate'])


# In[79]:


df['JobLevel'].value_counts()


# In[80]:


sns.countplot(df['JobLevel'])


# In[81]:


df['RelationshipSatisfaction'].value_counts()


# In[82]:


sns.countplot(df['RelationshipSatisfaction'])


# In[84]:


df['StockOptionLevel'].value_counts()


# In[85]:


sns.countplot(df['StockOptionLevel'])


# In[86]:


df['TotalWorkingYears'].value_counts()


# In[88]:


plt.subplots(figsize=(12,4))
sns.countplot(df['TotalWorkingYears'])


# In[89]:


df['TrainingTimesLastYear'].value_counts()


# In[90]:


sns.countplot(df['TrainingTimesLastYear'])


# In[91]:


df['WorkLifeBalance'].value_counts()


# In[92]:


sns.countplot(df['WorkLifeBalance'])


# In[93]:


df['YearsAtCompany'].value_counts()


# In[94]:


plt.subplots(figsize=(12,4))
sns.countplot(df['YearsAtCompany'])


# In[95]:


df['YearsInCurrentRole'].value_counts()


# In[96]:


sns.countplot(df['YearsInCurrentRole'])


# In[97]:


df['YearsSinceLastPromotion'].value_counts()


# In[98]:


sns.countplot(df['YearsSinceLastPromotion'])


# In[99]:


df['YearsWithCurrManager'].value_counts()


# In[100]:


sns.countplot(df['YearsWithCurrManager'])


# In[18]:


#To know the accuracy level
(1233-237)/1233


# In[21]:


#show the number of employees that left and stayed by age
plt.subplots(figsize=(12,4))
sns.countplot(x='Age',hue='Attrition',data = df, palette = 'colorblind')


# In[25]:


#29 and 31 age motly left the company


# In[27]:


#print all of the data types and their unique values
for column in df.columns:
    if df[column].dtype == object:
        print(str(column) + ":" + str(df[column].unique()))
        print(df[column].value_counts())
        print("-----------------------------")


# In[28]:


df.drop("StandardHours",axis=1,inplace=True)


# In[29]:


df.drop("Over18",axis=1,inplace=True)


# In[30]:


df.drop("EmployeeCount",axis=1,inplace=True)


# In[31]:


df.drop("EmployeeNumber",axis=1,inplace=True)


# In[32]:


# check the correlation
df.corr()


# In[36]:


#Lets visualize the correlation by heatmap
plt.figure(figsize = (14,14))
sns.heatmap(df.corr(),annot = True, fmt = ".0%")

total working year 68% correlated with the age value is showing good.
total working year 77% correlated with the Monthly income is showing good
Monthly income 95% correlated with job lavel
# In[37]:


from sklearn.preprocessing import LabelEncoder


# In[38]:


le = LabelEncoder()

# For encoding the datatype
second method

for column in df.columns:
    if df[column].dtype == np.number:
        continue
    df[cloumn] = LabelEncoder().fit_transform(df[column])
# In[41]:


list1=['Attrition','BusinessTravel','MaritalStatus','OverTime','JobRole','Gender','EducationField','Department' ]
for val in list1:
    df[val]=le.fit_transform(df[val].astype(str))
    
df


# In[ ]:


#create a new column for spliting the data make easy for iloc:-
#Create a new column
df['Age_year'] = df ['Age']
    


# In[49]:


df.head()


# In[ ]:


#split the data 


# In[163]:


x=df.iloc[:,: -1]


# In[164]:


x.head()


# In[165]:


y=df.iloc[:,-1]


# In[166]:


y.head()


# In[167]:


#split the data into 70% training and 25 % testing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=30, random_state= 0)


# In[168]:


# Use the random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators= 10, criterion = 'entropy', random_state = 0)
forest.fit(x_train, y_train)


# In[169]:


#get the accuracy score
forest.score(x_train,y_train)


# In[ ]:





# In[170]:


df.shape


# In[171]:


from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[172]:


mnb = MultinomialNB()
mnb.fit(x_train,y_train)
predmnb = mnb.predict(x_test)
print(accuracy_score(y_test,predmnb))
print(confusion_matrix(y_test,predmnb))
print(classification_report(y_test,predmnb))


# In[173]:


#Accuracy is not good is only 33 percent.


# In[174]:


svc = SVC(kernel='rbf') # kernel = 'rbf is -----by defult'
svc.fit(x_train,y_train)
svc.score(x_train,y_train)
presvc = svc.predict(x_test)
print(accuracy_score(y_test,presvc))
print(confusion_matrix(y_test,presvc))
print(classification_report(y_test,presvc))

accuracy lavel is in svc accuracy lavel is so low.
# In[176]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
knn.score(x_train,y_train)
predknn=knn.predict(x_test)
print(accuracy_score(y_test,predknn))
print(confusion_matrix(y_test,predknn))
print(classification_report(y_test,predknn))


# In[ ]:


accuracy of the knn is very low.

