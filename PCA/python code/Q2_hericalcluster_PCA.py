import numpy as np
import pandas as pd
medical_data=pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/PCA/Datasets_PCA/heart disease.csv")
med_data = medical_data.drop(["sex"], axis=1)
    
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
med_norm = norm_func(med_data.iloc[:, :])
med_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(med_norm, method = "complete", metric = "euclidean")
import matplotlib.pyplot as plt
# Dendrogram
plt.figure(figsize=(15, 8))
plt.title('Hierarchical Clustering Dendrogram of medical_data')
#plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(med_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

med_data['clust'] = cluster_labels # creating a new column and assigning it to new column 

new_data = med_data.iloc[:, [13,0,1,2,3,4,5,6,7,8,9,10,11,12]]
new_data.head()

# Aggregate mean of each cluster
new_data.iloc[:, 1:].groupby(new_data.clust).mean()

# creating a csv file 
new_data.to_csv("medical_clust.csv", encoding = "utf-8")

import os
os.getcwd()

# now apply the PCA on the herarical clustering new_data
import pandas as pd
import numpy as np
data=pd.read_csv("C:/Users/HAI/medical_clust.csv")
data.describe()

data.info()
data_new=data.drop(["Unnamed: 0"],axis=1)
# applying the pca for the dataset

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
#uni.data = uni.iloc[:, 1:]

# Normalizing the numerical data 
medical_normal = scale(data_new)
medical_normal
# select the number of pca
pca = PCA(n_components = 6)
med_pca_values = pca.fit_transform(medical_normal)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

# PCA weights
pca.components_
pca.components_[0]

# Cumulative variance 
# rounding the values 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# PCA scores of medical
med_pca_values
#creating a new dataframe of mde_pca_values
pca_data = pd.DataFrame(med_pca_values)
pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5"
final = pd.concat([data_new, pca_data.iloc[:, 0:6]], axis = 1)

# Scatter diagram
import matplotlib.pylab as plt
ax = final.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8))
final[['comp0', 'comp1', 'target']].apply(lambda x: ax.text(*x), axis=1)
