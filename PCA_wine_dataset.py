# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 21:36:35 2023

@author: Vaibhav Bhorkade
Assignment : Dimension Reduction With PCA

Problem Statement : 
Perform hierarchical and K-means clustering on the dataset. 
After that, perform PCA on the dataset and extract the first 
3 principal components and make a new dataset with these 3 principal
 components as the columns. Now, on this new dataset, perform 
 hierarchical and K-means clustering. Compare the results of 
 clustering on the original dataset and clustering on the principal
 components dataset (use the scree plot technique to obtain the
optimum number of clusters in K-means clustering and check if 
you’re getting similar results with and without PCA).
"""

"""
Business Objective
Minimize : Alcohol percentage
Maximaze : 
Business constraints  
"""

"""
Data Dictionary

Name of features          Type     Relevance      Description
0              Type       Nominal  Relevant  Type of alcohol
1           Alcohol    Continuous  Relevant         Alcohol 
2             Malic    Continuous  Relevant            Malic
3               Ash    Continuous  Relevant              Ash
4        Alcalinity    Continuous  Relevant       Alcalinity
5         Magnesium  Quantitative  Relevant        Magnesium
6           Phenols    Continuous  Relevant          Phenols
7        Flavanoids    Continuous  Relevant       Flavanoids
8     Nonflavanoids    Continuous  Relevant    Nonflavanoids
9   Proanthocyanins    Continuous  Relevant  Proanthocyanins
10            Color    Continuous  Relevant            Color
11              Hue    Continuous  Relevant              Hue
12         Dilution    Continuous  Relevant         Dilution
13          Proline  Quantitative  Relevant          Proline

"""
# EDA - Exploratory data analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("C:/datasets/wine.csv")
print(df.head())
# describe - 5 number summary
df.describe()
df.shape
# 178 rows and 14 columns
df.columns
# value counts
df['Type'].value_counts()
# Type Counts
# 2    71
# 1    59
# 3    48

# Check for null values
df.isnull()
# False
df.isnull().sum()
# All sum is 0 . There is no null value

# Scatter plot
sns.set_style("whitegrid");
sns.FacetGrid(df, hue="Type") \
   .map(plt.scatter, "Alcohol", "Phenols") \
   .add_legend();
plt.show();
# Blue points is Type 1 , orange is Type 2 and Green is Type 3  
# But red and green data points cannot be easily seperated.

# displot for Alcohol on Type
sns.FacetGrid(df, hue="Type") \
   .map(sns.distplot, "Alcohol") \
   .add_legend();
plt.show();

# displot for color on Type
sns.FacetGrid(df, hue="Type") \
   .map(sns.distplot, "Color") \
   .add_legend();
plt.show();

# Pairwise scatter plot: Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="Type");
plt.show()

# boxplot
# boxplot on Alcohol column
sns.boxplot(df.Alcohol)
# In alcohol column no outliers 

# boxplot on df column
sns.boxplot(df)
# There is outliers on some columns

# boxplot on Malic column
sns.boxplot(df.Malic)
# There is 3 outliers on column
# boxplot on df column
sns.boxplot(df.Ash)
# There is 3 outliers on column

# histplot
sns.histplot(df['Alcohol'],kde=True)
# data is right-skew and the not normallly distributed
# It is right - skew not symmetric

sns.histplot(df['Ash'],kde=True)
# data is right-skew and the not normallly distributed
# not symmetric

sns.histplot(df,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# Data Preproccesing
df.dtypes
# Type and Proline in int others all columns in float data types

# Identify the duplicates
duplicate=df.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series is created

duplicate
# False
sum(duplicate)
# output is zero

# We found outliers in some columns 
# Outliers treatment

# Winsorizer
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Malic']
                  )
df_t=winsor.fit_transform(df[['Malic']])

sns.boxplot(df[['Malic']])
# There is outliers

# check after applying the Winsorizer
sns.boxplot(df_t['Malic'])
# Outliers is removed

# Label encoder
# preferaly for nominal data

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# creating instance of label
labelencoder=LabelEncoder()
# split your data into input and output variables
x=df.iloc[:,0:]
y=df['Type']
df.columns

# we have nominal data Type
# we want to convert to label encoder
x['Typ']=labelencoder.fit_transform(x['Typ'])
# label encoder y
y=labelencoder.fit_transform(y)
# This is going to create an array, hence convert
# It is back to dataframe
y=pd.DataFrame(y)
df_new=pd.concat([x,y],axis=1)
# If you will see variables explorer, y do not have column name

# hence the rename column
df_new=df_new.rename(columns={0:'Typ'})

# Normalization

# Normalization function
# whenever there is mixed data apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm_func(df) 
# you can check the df_norm dataframe which is scaled between values from 0 and 1
b=df_norm.describe()
# Data is normalize
# in 0-1 

# Before we can apply clustering , need to plot dendrogram first
# now to create dendrogram , we need to measure distance,

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
# Linkage function gives us hierarchical clustering
z=linkage(df_norm,method="complete",metric='euclidean')
plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering dendrogram")
plt.xlabel("Index")
plt.ylabel("Distance")
# ref help of dendrogram
# sch.dendrogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
# dendrongram()
# applying agglomerative clustering choosing 4 as clustrers
# from dendrongram
# whatever has been displayed in dendrogram is not clustering
# It is just showing number of possible clusters
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=4,linkage='complete',affinity="euclidean").fit(df_norm)
# apply labels to clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)

# Assign this series to df Dataframe as column and name the column
df['clust']=cluster_labels
# we want to restore the column 7 to 0 th position
df1=df.iloc[:,:]
# now check the df dataframe

df1.to_csv("first.csv",encoding="utf-8")
import os
os.getcwd()

# K-Means
from  sklearn.cluster import KMeans

TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)# total within sum of square


TWSS
# As k value increases the TWSS the TWSS value decreases
plt.plot(k,TWSS,'ro-')
plt.xlabel("No_of_clusters")
plt.ylabel("Total_within_ss")
'''
elbow curve - when k changes from 2 to 3 , then decrease in twss is higher than 
when k chages from 3 to 4
Whwn k value changes from 5 to 6 decreases in twsss is higher than when k chages 3 to 4 .
When k values changes from 5 to 6 decrease
in twss is considerably less , hence considered k=3
'''

model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)

df['clust']=mb
df.head()
df=df.iloc[:,[7,0,1,2,3,4,5,6]]
df
df.iloc[:,2:8].groupby(df.clust).mean()

df.to_csv("Kfirst.csv",encoding="utf-8")
import os
os.getcwd()

# PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

# considering only numerical data
df.data=df.iloc[:,:]

# Normalizing the numerical data
uni_normal=scale(df.data)
uni_normal

pca=PCA(n_components=6)
pca_values=pca.fit_transform(uni_normal)

# The amount of variance that each examples is
var=pca.explained_variance_ratio_

# PCA weights
# pca.components_
# pca.components_[0]

# Cumulative variance
var1=np.cumsum(np.round(var,decimals=4)*100)
var1

# Variance plot for PCA components obtained
plt.plot(var1,color="red")

# pca_scores
pca_values


pca_data=pd.DataFrame(pca_values)
pca_data.columns="comp0","comp1","comp2","comp3","comp4","comp5"
final=pd.concat([df,pca_data.iloc[:,0:3]],axis=1)

# Scatter diagrams
import matplotlib.pyplot as plt
ax=final.plot(x="comp0",y="comp1",kind="scatter",figsize=(12,8))
final[['comp0','comp1','Type']].apply(lambda x:ax.text(*x),axis=1)

# from PCA we reduce the dimensionality of large data sets
#  and identify the underlying structure of the data.