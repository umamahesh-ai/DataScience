
import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt

link = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Network analytics/Datasets_Network Analytics/linkedin.csv")

link_new = nx.Graph()

link_new = nx.from_pandas_edgelist(link, source = '1', target = '13')
#2 nodes and 2 edges
print(nx.info(link_new))  
 # Degree Centrality
D = nx.degree_centrality(link_new) 
print(D) 

pos = nx.spring_layout(link_new,k=.15)
nx.draw_networkx(link_new, pos, node_size = 25, node_color = 'blue')

# closeness centrality
closeness = nx.closeness_centrality(link_new)
print(closeness)
# Betweeness_Centrality
B = nx.betweenness_centrality(link_new) 
print(B)
# Eigen vector centrality
evc = nx.eigenvector_centrality(link_new) 
print(evc)

# cluster coefficient
cluster_coeff = nx.clustering(link_new)
print(cluster_coeff)

# Average clustering
cc = nx.average_clustering(link_new) 
print(cc)
