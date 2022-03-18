import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
wine=pd.read_csv("C:/Users/HAI/Kmeans_wine_data.csv")
wine=wine.drop(['Unnamed: 0'],axis=1)
def normalize(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=normalize(wine)
df_norm.describe()
from scipy.cluster.hierarchy import linkage   
import scipy.cluster.hierarchy as sch
link=linkage(df_norm,method='complete',metric='euclidean')
# DendoDiagram
plt.figure(figsize=(15, 8))
plt.title('Hierarchical Clustering Dendrogram')
#plt.xlabel('Index')
#.ylabel('Distance')

sch.dendrogram(link, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()
# kmeans cluster after the pca
# kmeans clustering of the wine dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
# Kmeans on wine Data set 
wine_data = pd.read_csv("C:/Users/HAI/Kmeans_wine_data.csv")

wine_data.describe()
wine_data = wine_data.drop(["Unnamed: 0"], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
wine_norm = norm_func(wine_data.iloc[:, 1:])
from sklearn.cluster import KMeans
###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(wine_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(wine_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
wine_data['clust'] = mb # creating a  new column and assigning it to new column 

wine_data.head()
wine_norm.head()

wine_data = wine_data.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine_data.head()
wine_data.iloc[:, 1:].groupby(wine_data.clust).mean()

wine_data.to_csv("Kmeans_wine_data.csv", encoding = "utf-8")

import os