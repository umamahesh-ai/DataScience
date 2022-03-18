
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt 

insta = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Network analytics/Datasets_Network Analytics/instagram.csv")
insta = insta.iloc[:,:]

insta_1 = nx.Graph()

insta_1 = nx.from_pandas_edgelist(insta, source = '1', target = '8')

print(nx.info(insta_1))
# Degree Centrality
A = nx.degree_centrality(insta_1)  
print(A) 

pos = nx.spring_layout(insta_1,k=.15)
nx.draw_networkx(insta_1, pos, node_size = 25, node_color = 'red')
 # closeness centrality
closeness = nx.closeness_centrality(insta_1)  
print(closeness)
 # Betweeness_Centrality
b = nx.betweenness_centrality(insta_1) 
print(b)
 # Eigen vector centrality
evg = nx.eigenvector_centrality(insta_1)
print(evg)

# cluster coefficient
cluster_coeff = nx.clustering(insta_1)
print(cluster_coeff)

# Average clustering
cc = nx.average_clustering(insta_1) 
print(cc)



