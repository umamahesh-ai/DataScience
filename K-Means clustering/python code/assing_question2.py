import pandas as pd
import numpy as np
from matplotlib.pylab import plt
crime_data=pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/k-mean clustering/Datasets_Kmeans/crime_data (1).csv")
crime_data.info
crime_data.describe()

# normilize function
def norm(i):
    x=((i-i.min())/(i.max()-i.min()))
    return x
new_crime=norm(crime_data.iloc[:,1:])

# import KMeans

from sklearn.cluster import KMeans

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(new_crime) # .fit-->apply the algorithm of dataset

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
new_crime['mb'] = mb # creating a  new column and assigning it to new column 

df = new_crime.iloc[:,[4,0,1,2,3]]

###### scree plot or elbow curve ############

TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")


