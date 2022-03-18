import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.ExcelFile("C:/Users/HAI/Desktop/360DigitMG/Assingment/k-mean clustering/Datasets_Kmeans/Telco_customer_churn (1).xlsx")
Telco_customer_churn=pd.read_excel(df)
Telco_customer_churn.describe()
Telco_customer_churn.info()
# convert the categorical data into numerical data
from sklearn.preprocessing import LabelEncoder
categorical=Telco_customer_churn.select_dtypes(include=['object'])
numerical=Telco_customer_churn.select_dtypes(exclude=['object'])
new_telco=pd.concat((categorical,numerical),axis=1)
new_df=new_telco.loc[:,'Customer ID':'Payment Method'].apply( LabelEncoder().fit_transform)

data_concat=pd.concat((new_df,numerical),axis=1)
# drop the columns using  column names
new_data=data_concat.drop(['Quarter','Count'],axis=1,inplace=True)
# normalise the data
def norm(i):
    x=((i-i.min())/(i.max()-i.min()))
    return x
data_norm=norm(data_concat)
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
k=list(range(1,9))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(data_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# elbow curve
plt.plot(k,TWSS,'ro-')
plt.xlabel('number of clusters')
plt.ylabel('Totel_number_telco')
    


