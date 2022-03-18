import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
crime=pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Hirerical clustering/Dataset_Assignment Clustering/crime_data.csv")
crime.describe()
crime.info()
crime.iloc(: ,1:)
New_crime = crime.drop(["Unnamed: 0"], axis=1)# Drop the cloumn 
# Normilization function
def norm(i):
    x=i-i.min()/(i.max()-i.min())
    return x
crime_data=norm(New_crime)
crime_data.describe()
#for  Dnendrogram we can import the from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z=linkage(crime_data,method='complete',metric='euclidean')
# dendrogram -----> clustering the same data
plt.figure(figsize=(15, 8))
plt.title('Crime_data Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)

# Elbow curve or sree plot
# for elow curve we can import the from sklearn.cluster import KMeans
from sklearn.cluster import KMeans

TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(crime_data)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-')
plt.title('Crime_data Clustering of elbow curve')
plt.xlabel("No_of_Clusters") # X axis No of clusters
plt.ylabel("total_within_SS")# y axis total with in ss
plt.show() # To display the elbow curve
