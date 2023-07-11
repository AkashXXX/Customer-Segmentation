
import pandas as pd
customers = pd.read_csv(r'D:\General\Tech\customer segmentation\customers.csv')

                              # DATA EXPLORATION AND PREPARATION
customers.info()

customers.describe(include = 'all').round(2)

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()

ax = sns.boxplot(data=customers, x='Gender', y='Annual Income (k$)', palette='colorblind')

ax = sns.boxplot(data=customers, x='Gender', y='Spending Score (1-100)', palette='colorblind')

ax = sns.boxplot(data=customers, x='Gender', y='Age', palette='colorblind')

ax = sns.scatterplot(data=customers, x='Annual Income (k$)', y='Spending Score (1-100)',s=150)


customers[['Annual Income (k$)', 'Spending Score (1-100)']].describe().round(2)

customer_scaled= customers[['Annual Income (k$)', 'Spending Score (1-100)']]


                                            # MODELLING AND EVALUATION
# Choosing the right no of clusters
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


# Analyze and interpret the clusters
km= KMeans(n_clusters=5, n_init=10)
km.fit(customer_scaled)

km

cluster_centers= pd.DataFrame(km.cluster_centers_, columns = ['Annual Income (k$)', 'Spending Score (1-100)'])
cluster_centers

km.cluster_centers_

km.labels_

# PLOTTING THE FINAL CLUSTERED DATA
plt.figure(figsize=(8,6))

ax=sns.scatterplot(data=customer_scaled, x='Annual Income (k$)', y='Spending Score (1-100)', hue = km.labels_)

ax=sns.scatterplot(data=cluster_centers, x='Annual Income (k$)', y='Spending Score (1-100)', hue = cluster_centers.index, marker='D', ec='black', legend=False, s=50)

customers['cluster']= km.labels_.tolist()
customers

customers= pd.get_dummies(customers, columns=['Gender'])
customers

# PREDICTING VALUE FOR NEW INPUT
y = km.predict([[201, 30, 18, 41, 1, 0]])
y

