import pandas as pd
import numpy as np
medical_data=pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/PCA/Datasets_PCA/heart disease.csv")
medical_data.describe()

medical_data.info()
medical_data = medical_data.drop(["sex"], axis = 1)
# applying the pca for the dataset

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
#uni.data = uni.iloc[:, 1:]

# Normalizing the numerical data 
medical_normal = scale(medical_data)
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
final = pd.concat([medical_data, pca_data.iloc[:, 0:6]], axis = 1)

# Scatter diagram
import matplotlib.pylab as plt
ax = final.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8))
final[['comp0', 'comp1', 'target']].apply(lambda x: ax.text(*x), axis=1)

#apply the herical cluster on the final data
# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(final, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8))
plt.title('Hierarchical Clustering Dendrogram of final_PCA')
#plt.xlabel('Index')
#plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(final) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

final['clust'] = cluster_labels # creating a new column and assigning it to new column 

final_data = final.iloc[:, [16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
Univ1.head()

# Aggregate mean of each cluster
final_data.iloc[:, 1:].groupby(final_data.clust).mean()

# creating a csv file 
final_data.to_csv("final_medical_PCA.csv", encoding = "utf-8")

import os
os.getcwd()

