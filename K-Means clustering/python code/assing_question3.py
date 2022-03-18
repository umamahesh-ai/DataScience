import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Insurance=pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/k-mean clustering/Datasets_Kmeans/Insurance Dataset.csv")
Insurance.describe()
Insurance.info
# normalize the dataset
def norm(i):
    x=((i-i.min()/i.max()-i.min()))
    return x
new_insurance=norm(Insurance)
# select the number of clusters 
from sklearn.cluster import KMeans
model=KMeans(n_clusters=3)
# apply the algorithm
model.fit(new_insurance)
model.labels_
new_model=pd.DataFrame(model.labels_)
new_insurance['new_model']=new_model
df=new_insurance.iloc[:,[5,0,1,2,3,4]]
# plot the elbow curve or sree plot
TWSS=[]
k=list(range(1,8))
for i in k:
    kmean=KMeans(n_clusters=i)
    kmean.fit(df)
    TWSS.append(kmean.inertia_)
TWSS
# graph 
plt.plot(k,TWSS,'ro-')
plt.xlabel('number of clusters')
plt.ylabel('Totel_within_ss')
