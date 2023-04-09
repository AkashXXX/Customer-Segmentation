#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
customers = pd.read_csv(r'D:\General\Tech\customer segmentation\customers.csv')
customers


# In[3]:


customers= pd.get_dummies(customers, columns=['Gender'])


# In[5]:


customers


# In[6]:


customers.drop(['CustomerID'], axis=1, inplace=True)
customers


# In[7]:


from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()


# In[8]:


#Choosing the right no of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    km= KMeans(n_clusters=i, n_init=10)
    km.fit(customers)
    wcss.append(km.inertia_)

wcss_series= pd.Series(wcss, index= range(1,11))

plt.figure(figsize=(8,6))
ax= sns.lineplot(y=wcss_series, x= wcss_series.index)
ax= sns.scatterplot(y=wcss_series, x=wcss_series.index)


# In[9]:


#Analyze and interpret the clusters
km= KMeans(n_clusters=5, n_init=10)
km.fit(customers)


# In[10]:


km.cluster_centers_


# In[11]:


cluster_centers= pd.DataFrame(km.cluster_centers_, columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Female', 'Gender_Male'])


# In[12]:


cluster_centers


# In[15]:


plt.figure(figsize=(8,6))

ax=sns.scatterplot(data=customers, x='Annual Income (k$)', y='Spending Score (1-100)', hue = km.labels_)

ax=sns.scatterplot(data=cluster_centers, x='Annual Income (k$)', y='Spending Score (1-100)', hue = cluster_centers.index, marker='D', ec='black', legend=False, s=50)


# In[27]:


customers['cluster']= km.labels_.tolist()
customers


# In[29]:


y = km.predict([[24, 16, 76, 1, 0]])
y

