# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:35:58 2023

@author: Vaibhav Bhorkade

Problem Statement : 
A pharmaceuticals manufacturing company is conducting a study on a 
new medicine to treat heart diseases. The company has gathered data 
from its secondary sources and would like you to provide high level 
analytical insights on the data. Its aim is to segregate patients 
depending on their age group and other factors given in the data. 
Perform PCA and clustering algorithms on the dataset and check if 
the clusters formed before and after PCA are the same and provide 
a brief report on your model. You can also explore more ways to 
improve your model. 
Note: This is just a snapshot of the data. The datasets can be 
downloaded from AiSpry LMS in the Hands-On Material section.

"""
"""
Business Objective
Minimize : heart disease and colestrol
Maximaze : Treatment and awarness
Business constraints: Health Policy
"""
"""
Data Dictionary

 Name of Features        Type Relevance Description
0               age   Discreate  Relavant         age of person
1               sex     Nominal  Relavant         sex or Gender
2                cp     Nominal  Relavant          cp 
3          trestbps  continuous  Relavant    trestbps
4              chol  Continuous  Relavant        chol
5               fbs     Nominal  Relavant         fbs
6           restecg     Nominal  Relavant     restecg
7           thalach   discreate  Relavant     thalach
8             exang     Nominal  Relavant       exang
9           oldpeak   discreate  Relavant     oldpeak
10            slope     Nominal  Relavant       slope
11               ca     Nominal  Relavant          ca
12             thal     Nominal  Relavant        thal
13           target     Nominal  Relavant      target

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("heart disease.csv")
print(df)

# EDA - Exploratory data analysis
df.head()
df.tail()
# Five number summary
df.describe()
# Value counts
df.sex.value_counts()
# 1    207
# 0     96
# Value count for Target
df.target.value_counts()
# 1 is 165 and 0 is 138

# Check for NULL values
df.isnull()
# False
df.isnull().sum()
# Sum is 0 , No null values
df.shape
# 303 rows and 14 columns
df.columns
'''
'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
'''
# Scatter plot
sns.set_style("whitegrid");
sns.FacetGrid(df, hue="sex") \
   .map(plt.scatter, "trestbps", "chol") \
   .add_legend();
plt.show();
# Blue points is  0 or female type , orange is Type 1 or male type 
# But red and blue data points cannot be easily seperated.

# displot for Alcohol on Type
sns.FacetGrid(df, hue="sex") \
   .map(sns.distplot, "chol") \
   .add_legend();
plt.show();
# chol is higher for 1 (male) represented using orange

# displot for color on Type
sns.FacetGrid(df, hue="sex") \
   .map(sns.distplot, "trestbps") \
   .add_legend();
plt.show();
# trestbps is higher for 0 

# Pairwise scatter plot: Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="Type");
plt.show()

# Vizualize boxplot for finding outliers
# boxplot on age column
sns.boxplot(df.age)
# In alcohol column no outliers 

# boxplot on trestbps column
sns.boxplot(df.trestbps)
# There is Many outliers on column
# boxplot on df column
sns.boxplot(df.chol)
# There is Many outliers on column
# Let check for DataFrame
# boxplot on df column
sns.boxplot(df)
# There is outliers on some columns

# histplot
sns.histplot(df['age'],kde=True)
# data is right-skew and the not distributed
# It is right - skew and not symmetric

sns.histplot(df['chol'],kde=True)
# data is left-skew and the not normallly distributed
# not symmetric

sns.histplot(df,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# Correlation Heatmap
sns.heatmap(df.corr())
# age is 1 

# Data Preproccesing
df.dtypes
# All data i int type only oldpeak in float

# Identify the duplicates
duplicate=df.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series is created

duplicate
# False
sum(duplicate)
# output is 1

# Duplicate Treatment
df.drop_duplicates(inplace=True)
# check sum
duplicate=df.duplicated()
sum(duplicate)
# Now sum is 0

# We found outliers in some columns 
# Outliers treatment

# Winsorizer
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['trestbps']
                  )
df_t=winsor.fit_transform(df[['trestbps']])

sns.boxplot(df[['trestbps']])
# There is outliers

# check after applying the Winsorizer
sns.boxplot(df_t['trestbps'])
# Outliers is removed
df.trestbps=df_t

from sklearn.preprocessing import StandardScaler
df.describe()
a=df.describe()
# Initialize the scalar
scalar=StandardScaler()
df1=scalar.fit_transform(df)
df=pd.DataFrame(df1)
res=df.describe()
# here if you will check res , in variable environment then


# Label encoder
# preferaly for nominal data

from sklearn.preprocessing import LabelEncoder

# creating instance of label
labelencoder=LabelEncoder()
# split your data into input and output variables
x=df.iloc[:,0:]
y=df['sex']
df.columns

# we have nominal data Type
# we want to convert to label encoder
x['sex']=labelencoder.fit_transform(x['sex'])
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
df_norm=norm_func(df_new) 
# you can check the df_norm dataframe which is scaled between values from 0 and 1
b=df_norm.describe()
# Data is normalize
# in 0-1 
df=df_norm

# PCA - 
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

df.dropna(inplace=True)
# considering only numerical data
df.data=df.iloc[:,1:]

# Normalizing the numerical data
df_normal=scale(df.data)
df_normal

pca=PCA(n_components=6)

pca_values=pca.fit_transform(df_normal)

# The amount of variance that each examples is
var=pca.explained_variance_ratio_

# Cumulative variance
var1=np.cumsum(np.round(var,decimals=4)*100)
var1

# Variance plot for PCA components obtained
plt.plot(var1,color="red")

# pca_scores
pca_values

pca_data=pd.DataFrame(pca_values)
pca_data.columns="pca0","pca1","pca2","pca3","pca4","pca5"
final=pd.concat([df,pca_data.iloc[:,0:3]],axis=1)

# Scatter diagrams
ax=final.plot(x="pca0",y="pca1",kind="scatter",figsize=(12,8))
final[['pca0','pca1','sex']].apply(lambda x:ax.text(*x),axis=1)

# Before we can apply clustering , need to plot dendrogram first
# dendrogram
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
plt.show()
'''
elbow curve - when k changes from 2 to 3 , then decrease in twss is higher than 
when k chages from 3 to 4
Whwn k value changes from 5 to 6 decreases in twsss is higher than when k chages 3 to 4 .
When k values changes from 5 to 6 decrease
in twss is considerably less , hence considered k=4
'''

model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)

df['clust']=mb
df.head()
df=df.iloc[:,:]
df
df.iloc[:,2:8].groupby(df.clust).mean()

df.to_csv("Kfirst.csv",encoding="utf-8")
import os
os.getcwd()
# The new dataframe is saved in respective directory
# with all the data cleaning and processing .