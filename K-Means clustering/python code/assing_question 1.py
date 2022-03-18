import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.ExcelFile("C:/Users/HAI/Desktop/360DigitMG/Assingment/k-mean clustering/Datasets_Kmeans/EastWestAirlines (1).xlsx")
new_df=pd.read_excel(df,1)
# Normilization function
def norm(i):
    x=(i-i.min()/(i.max()-i.min()))
    return x
data=norm (new_df.iloc[:,1:])


from sklearn.cluster import KMeans


TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-')  # plot the graph
plt.xlabel("No_of_Clusters") # x label
plt.ylabel("total_within_SS") # y label

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3) # select the number of clusters
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
Univ['clust'] = mb # creating a  new column and assigning it to new column 


from sklearn.cluster import KMeans
data=new_df.iloc[:,1:]



kmean=KMeans(n_clusters=3) # select the number of clusters
kmean.fit(data)
cluster=kmean.fit_predict(data)

data['cluster']=cluster












