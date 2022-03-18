import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
wine=pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/PCA/Datasets_PCA/wine.csv")
# Normalize the data
def normalize(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=normalize(wine.iloc[:, 1:])
df_norm.describe()
from scipy.cluster.hierarchy import linkage   
import scipy.cluster.hierarchy as sch
# calculate the distance b/w each datapoint using euclidean distance
link=linkage(df_norm,method='complete',metric='euclidean')
# DendoDiagram
plt.figure(figsize=(15, 8))
plt.title('Hierarchical Clustering Dendrogram _wine dataset')
#plt.xlabel('Index')
#.ylabel('Distance')

sch.dendrogram(link, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# kmeans clustering of the wine dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
# Kmeans on wine Data set 
wine_data = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/PCA/Datasets_PCA/wine.csv")

wine_data.describe()


# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
wine_norm = norm_func(wine_data.iloc[:, 1:])

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
os.getcwd()
#pca apply on the wine dataset


import pandas as pd
import numpy as np

data = pd.read_csv("C:/Users/HAI/Kmeans_wine_data.csv")
data.describe()

data.info()
data = data.drop(["Unnamed: 0"], axis = 1)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
#uni.data = data.iloc[:, 1:]

# Normalizing the numerical data 
data_normal = scale(data)
data_normal

pca = PCA(n_components = 3)
pca_values = pca.fit_transform(data_normal)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

# PCA weights
pca.components_
pca.components_[0] # it will display the 1st row 

# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2"
final = pd.concat([data, pca_data.iloc[:, 0:3]], axis = 1)

# Scatter diagram
import matplotlib.pylab as plt
ax = final.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8))
final[['comp0', 'comp1', 'Type']].apply(lambda x: ax.text(*x), axis=1)

