import pandas as pd
import matplotlib.pylab as plt
read=pd.ExcelFile("C:/Users/HAI/Desktop/360DigitMG/Assingment/Hirerical clustering/Dataset_Assignment Clustering/EastWestAirlines.xlsx")
df=pd.read_excel(read,1)
df.describe()
df.info
# Normilization function
def normalize(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=normalize(df.iloc[:, 1:])
df_norm.describe()
from scipy.cluster.hierarchy import linkage   
import scipy.cluster.hierarchy as sch
link=linkage(df_norm,method='complete',metric='euclidean')
# DendoDiagram
plt.figure(figsize=(15, 8))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')

sch.dendrogram(link, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


