#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[25]:


df_train=pd.read_csv(r'C:\Users\Noman\Desktop\Sales Prediction Project\train_data.csv')


# In[26]:


df_test=pd.read_csv(r'C:\Users\Noman\Desktop\Sales Prediction Project\test_data.csv')


# In[27]:


df_train.head()


# In[28]:


df_train.shape


# In[29]:


df_train.info


# In[30]:


df_train. columns. values


# In[31]:


df_train.isnull().sum()


# In[32]:


df_train.describe()


# In[33]:


mean_Item_Weight=df_train['Item_Weight'].mean()


# In[34]:


df_train['Item_Weight'].fillna(mean_Item_Weight,inplace=True)


# In[35]:


df_train.isnull().sum()


# In[39]:


df_train.Outlet_Size.value_counts()


# In[41]:


# filling with Unknown class
df_train = df_train.fillna("Unknown")
df_train


# In[44]:


df_train.Outlet_Size.value_counts()


# In[45]:


df_train.Item_Weight.value_counts()


# In[46]:


df_train.isnull().sum()


# In[47]:


sns.countplot(df_train['Outlet_Size'])


# In[48]:


plt.rcParams["figure.figsize"] = (10,6)
sns.countplot(df_train['Outlet_Type'])


# In[50]:


sns.countplot(df_train['Outlet_Location_Type'])


# In[53]:


sns.countplot(df_train['Outlet_Identifier'])


# In[56]:


plt.rcParams["figure.figsize"] = (25,10)
sns.countplot(df_train['Item_Type'])


# In[68]:


plt.bar(df_train['Outlet_Type'],df_train['Item_Outlet_Sales'])


# In[59]:


plt.bar(df_train['Outlet_Size'],df_train['Item_Outlet_Sales'])


# In[60]:


plots=sns.barplot(x='Outlet_Establishment_Year',y='Item_Outlet_Sales',data=df_train)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
plt.show()


# In[72]:


sns.boxplot(x=df_train['Item_Outlet_Sales'])


# In[73]:


fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df_train['Outlet_Type'], df_train['Item_Outlet_Sales'])
ax.set_xlabel('Outlet_Type')
ax.set_ylabel('Item_Outlet_Sales')
plt.show()


# In[86]:


fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df_train['Outlet_Identifier'], df_train['Item_Outlet_Sales'])
ax.set_xlabel('Outlet_Identifier')
ax.set_ylabel('Item_Outlet_Sales')
plt.show()


# In[87]:


plt.figure(figsize=(8,5))
sns.boxplot(x='Outlet_Identifier',y='Item_Outlet_Sales',data=df_train)


# In[88]:


plt.figure(figsize=(8,5))
sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',data=df_train)


# In[90]:


df_train.Item_Outlet_Sales.hist()


# In[91]:


print(df_train['Item_Outlet_Sales'].quantile(0.10))
print(df_train['Item_Outlet_Sales'].quantile(0.90))


# In[92]:


df_train["Item_Outlet_Sales"] = np.where(df_train["Item_Outlet_Sales"] <343.5528, 343.5528,df_train['Item_Outlet_Sales'])
df_train["Item_Outlet_Sales"] = np.where(df_train["Item_Outlet_Sales"] >4570.0512, 4570.0512,df_train['Item_Outlet_Sales'])
print(df_train['Item_Outlet_Sales'].skew())


# In[93]:


plt.figure(figsize=(8,5))
sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',data=df_train)


# In[94]:


df_train.Item_Outlet_Sales.hist()


# In[95]:


df_test.isnull().sum()


# In[96]:


mean_Item_Weight=df_test['Item_Weight'].mean()


# In[98]:


df_test['Item_Weight'].fillna(mean_Item_Weight,inplace=True)


# In[99]:


df_test.isnull().sum()


# In[103]:


df_test.Outlet_Size.value_counts()


# In[102]:


df_test.Outlet_Size.value_counts()


# In[104]:


# filling with Unknown class
df_test = df_test.fillna("Unknown")
df_test


# In[105]:


df_test.isnull().sum()


# In[121]:


# The Density Plot of SalePrice
sns.distplot(df_train['Item_Outlet_Sales'])


# In[143]:


# Import label encoder
from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df_train['Item_Fat_Content']= label_encoder.fit_transform(df_train['Item_Fat_Content'])
  
df_train['Item_Fat_Content'].unique()


# In[144]:


# split data into X and y
X = df_train.iloc[:, 1:2].values
y = df_train.iloc[:, 2].values


# In[145]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[146]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[157]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[161]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[162]:


# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X, y)
#regressor created properly with default parameter after execute the above line of code
#now we will check what was the actual sales after scaling 


# In[164]:


# Predicting a new result

y_pred = regressor.predict([[2500]])


# In[170]:


# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Outlet_Size')
plt.ylabel('Item_Outlet_Sales')
plt.show()


# In[167]:


# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Outlet_Identifier')
plt.ylabel('Item_Outlet_Sales')
plt.show()


# In[172]:


# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Item_Identifier')
plt.ylabel('Item_Outlet_Sales')
plt.show()


# In[177]:


from sklearn.cluster import KMeans 
#we are going to findout the optimal number of cluster & we have to use the elbow method
wcss = [] 


# In[178]:


for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




