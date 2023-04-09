#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
customers = pd.read_csv(r'D:\General\Tech\customer segmentation\customers.csv')
customers


# In[24]:


customers


# In[25]:


customers.info()


# In[26]:


customers.describe(include = 'all').round(2)


# In[27]:


from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()


# In[35]:


ax = sns.boxplot(data=customers, x='Gender', y='Annual Income (k$)', palette='colorblind')


# In[36]:


ax = sns.boxplot(data=customers, x='Gender', y='Spending Score (1-100)', palette='colorblind')


# In[37]:


ax = sns.boxplot(data=customers, x='Gender', y='Age', palette='colorblind')


# In[45]:


ax = sns.scatterplot(data=customers, x='Annual Income (k$)', y='Spending Score (1-100)',s=150)


# In[48]:


customers[['Annual Income (k$)', 'Spending Score (1-100)']].describe().round(2)


# In[49]:


customer_scaled= customers[['Annual Income (k$)', 'Spending Score (1-100)']]


# In[67]:


customer_scaled


# In[55]:


#Choosing the right no of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    km= KMeans(n_clusters=i, n_init=10)
    km.fit(customer_scaled)
    wcss.append(km.inertia_)

wcss_series= pd.Series(wcss, index= range(1,11))

plt.figure(figsize=(8,6))
ax= sns.lineplot(y=wcss_series, x= wcss_series.index)
ax= sns.scatterplot(y=wcss_series, x=wcss_series.index)


# In[58]:


#Analyze and interpret the clusters
km= KMeans(n_clusters=5, n_init=10)
km.fit(customer_scaled)


# In[89]:


km


# In[63]:


cluster_centers= pd.DataFrame(km.cluster_centers_, columns = ['Annual Income (k$)', 'Spending Score (1-100)'])


# In[73]:


cluster_centers


# In[65]:


km.cluster_centers_


# In[64]:


km.labels_


# In[80]:


plt.figure(figsize=(8,6))

ax=sns.scatterplot(data=customer_scaled, x='Annual Income (k$)', y='Spending Score (1-100)', hue = km.labels_)

ax=sns.scatterplot(data=cluster_centers, x='Annual Income (k$)', y='Spending Score (1-100)', hue = cluster_centers.index, marker='D', ec='black', legend=False, s=50)


# In[82]:


customers['cluster']= km.labels_.tolist()
customers


# In[88]:


customers= pd.get_dummies(customers, columns=['Gender'])
customers


# In[108]:


y = km.predict([[201, 30, 18, 41, 1, 0]])
y

