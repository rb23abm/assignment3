#!/usr/bin/env python
# coding: utf-8

# # Total reserves (includes gold, current USD)

# In[23]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import scipy.optimize as opt
import errors
warnings.filterwarnings("ignore")

# https://data.worldbank.org/indicator/FI.RES.TOTL.CD?view=chart


# In[2]:


def worldbank_data(filename):
    """
    Reads World Bank data from a CSV file.

    Parameters:
    - filename (str): The path to the CSV file containing World Bank data.

    Returns:
    - years_df (DataFrame): Transposed DataFrame with 'Country Name' as the index.
    - worldbank_df (DataFrame): Original DataFrame read from the CSV file.
    """
    worldbank_df = pd.read_csv(filename, skiprows=4)
    years_df = worldbank_df.set_index(['Country Name']).T
    return years_df, worldbank_df


# In[3]:


years_df, countries_df = worldbank_data('API_FI.RES.TOTL.CD_DS2_en_csv_v2_6298254.csv')


# In[4]:


data = countries_df[['Country Name', 'Indicator Name'] + list(map(str, range(2000, 2021)))]


# In[5]:


data = data.dropna()


# In[6]:


data.head()


# In[7]:


growth = data[["Country Name", "2010", "2020"]].copy()
growth = growth.assign(Growth_Percentage=lambda x: 100.0 * (pd.to_numeric(x["2020"], errors='coerce') - pd.to_numeric(x["2010"], errors='coerce')) / pd.to_numeric(x["2010"], errors='coerce'))
growth.head()


# In[9]:


growth.describe()


# In[10]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x=growth["2020"], y=growth["Growth_Percentage"], label="Total reserves (includes gold, current US$)", color="green")
plt.xlabel("Total reserves")
plt.ylabel("Growth (%) from 2000 to 2021")
plt.title("Scatter Plot")
plt.show()


# In[11]:


def one_silhouette(xy, n):
    """Calculates silhouette score for n clusters"""
    kmeans = KMeans(n_clusters=n, n_init=10)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score


# In[12]:


scaler = StandardScaler()
growth2 = growth[["2020", "Growth_Percentage"]]
scaler.fit(growth2)
norm = scaler.transform(growth2)

silhouette_scores = []
for i in range(2, 11):
    score = one_silhouette(norm, i)
    silhouette_scores.append(score)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()


# In[13]:


kmeans = KMeans(n_clusters=6, n_init=10)
kmeans.fit(norm)
labels = kmeans.labels_
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]

plt.figure(figsize=(8, 8))
plt.scatter(growth["2020"], growth["Growth_Percentage"], 10, labels, marker="o", cmap="ocean")
plt.scatter(xkmeans, ykmeans, 50, "k", marker="d")
plt.xlabel("Total reserves")
plt.ylabel("Growth (%) from 2000 to 2021")
plt.title("K-Means Clustering")
plt.show()


# In[14]:


years_df.head()


# In[15]:


uk_data = years_df[['United Kingdom']].loc['2000':'2020'].reset_index().rename(columns={'index': 'Years', 'United Kingdom': 'Reserves (USD)'})
uk_data.columns.name = 'Index'
uk_data['Years'] = uk_data['Years'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
uk_data['Reserves (USD)'] = uk_data['Reserves (USD)'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
uk_data.head()


# In[16]:


uk_data.describe()


# In[17]:


plt.figure(figsize=(8, 6))
sns.lineplot(data=uk_data, x='Years', y='Reserves (USD)')
plt.xlabel('Year')
plt.ylabel('Reserves (USD)')
plt.title('Reserves (USD) Over Time (2000-2020)')
plt.show()


# In[18]:


def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 2000
    f = n0 * np.exp(g*t)
    return f


# In[19]:


param, covar = opt.curve_fit(exponential, uk_data["Years"], uk_data["Reserves (USD)"], p0=(1.1e5, 0.2))


# In[20]:


uk_data["fit"] = exponential(uk_data["Years"], *param)
plt.figure(figsize=(8, 6))
sns.lineplot(data=uk_data, x="Years", y="Reserves (USD)", label="Reserves (USD)", color="red")
sns.lineplot(data=uk_data, x="Years", y="fit", label="Exponential Fit", color="green")
plt.xlabel("Year")
plt.ylabel("Total Reserves (USD)")
plt.title("Total Reserves (USD) with Exponential Fit")
plt.legend()
plt.show()


# In[27]:


years_future = np.arange(2020, 2041, 1)
predictions = exponential(years_future, *param)
confidence_range = errors.error_prop(years_future, exponential, param, covar)

plt.figure(figsize=(10, 6))
sns.lineplot(data=uk_data, x="Years", y="Reserves (USD)", label="Reserves (USD)", color="red")
sns.lineplot(x=years_future, y=predictions, label="Best Fitting Function", color='blue')
sns.lineplot(x=years_future, y=predictions - confidence_range, color='green', alpha=0.2, label="Confidence Range")
sns.lineplot(x=years_future, y=predictions + confidence_range, color='green', alpha=0.2)
plt.xlabel("Year")
plt.ylabel("Total Reserves (USD)")
plt.title("Total Reserves (USD) Prediction in United Kingdom")
plt.legend()
plt.show()


# In[ ]:




