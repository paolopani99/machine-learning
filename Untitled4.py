#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing


# In[2]:


path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv"


# In[4]:


df = pd.read_csv(path)
df.head()


# In[5]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# In[6]:


df['loan_status'].value_counts()


# In[7]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[8]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[9]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# In[10]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# In[11]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# In[12]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# In[13]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# In[14]:


df[['Principal','terms','age','Gender','education']].head()


# In[108]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# In[16]:


X = Feature
X[0:5]


# In[17]:


y = df['loan_status'].values
y[0:5]


# In[18]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[22]:


from sklearn.model_selection import train_test_split
x_train, x_test,  y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=4)


# In[23]:


from sklearn.neighbors import KNeighborsClassifier


# In[48]:


k=7
neigh= KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)


# In[137]:


y_hat=neigh.predict(x_test)


# In[50]:


metrics.accuracy_score(y_test, y_hat)


# In[51]:


from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree


# In[52]:


loantree= DecisionTreeClassifier(criterion="entropy", max_depth=4)
loantree


# In[59]:


loantree.fit(X, y)


# In[62]:


tree.plot_tree(loantree)
plt.figure(figsize=(1,4))
plt.show()


# In[63]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X, y) 


# In[67]:


from sklearn.linear_model import LogisticRegression
LR= LogisticRegression(C=0.01, solver="liblinear").fit(X,y)
LR


# In[65]:


yhat=clf.predict(X)


# In[68]:


y_hatlog=LR.predict(X)


# In[69]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# In[73]:


path_test="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv"


# In[82]:


test_df=pd.read_csv(path_test)
test_df.head()


# In[115]:


test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_df.groupby(['education'])['loan_status'].value_counts(normalize=True)
Feature = test_df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.dropna(inplace=True)
Feature
X_test=Feature 
X_test= preprocessing.StandardScaler().fit(X_test).transform(X_test)
y = test_df['loan_status'].values


# In[112]:


y_hatlog=LR.predict(X_test)


# In[123]:


jaccard_score(y, y_hatlog, pos_label="COLLECTION")


# In[126]:


y_hat=neigh.predict(X_test)


# In[127]:


jaccard_score(y, y_hat, pos_label="COLLECTION")


# In[128]:


y_hatt=loantree.predict(X_test)


# In[129]:


jaccard_score(y, y_hatt, pos_label="COLLECTION")


# In[131]:


f1_score(y, y_hatlog, average='weighted') 


# In[132]:


f1_score(y, y_hat, average='weighted') 


# In[133]:


f1_score(y, y_hatt, average='weighted') 


# In[134]:


yhat_prob=LR.predict_proba(X_test)


# In[136]:


log_loss(y, yhat_prob)


# In[ ]:




