import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
AutoInsurance=pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/k-mean clustering/Datasets_Kmeans/AutoInsurance (1).csv")
AutoInsurance.describe()
AutoInsurance.shape
AutoInsurance.apply(lambda x: sum(x.isnull()),axis=0)# to find the null values
# convert the categorical data into numerical data
from sklearn.preprocessing import LabelEncoder
categorical=AutoInsurance.select_dtypes(include=['object'])
numerical=AutoInsurance.select_dtypes(exclude=['object'])
New_Insurance=pd.concat((categorical,numerical),axis=1)
new_df=New_Insurance.loc[:,'Customer':'Vehicle Size'].apply( LabelEncoder().fit_transform)
data=pd.concat((new_df,numerical),axis=1)
# normalise the data
def norm(i):
    x=((i-i.min())/(i.max()-i.min()))
    return x
data_norm=norm(data)
# select the number of clusters
from sklearn.cluster import KMeans
model=KMeans(n_clusters=3)
#apply the algorithm
model.fit(data_norm)
model.labels_
new_model=pd.DataFrame(model.labels_)
data_norm['new_model']=new_model

# plot the sree plot or elbow curve

TWSS=[]
k=list(range(1,10))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(data_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# elbow curve
plt.plot(k,TWSS,'ro-')
plt.xlabel('number of clusters')
plt.ylabel('Totel_number_AutoInsurance')
    