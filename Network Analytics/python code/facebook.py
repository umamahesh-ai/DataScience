
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

fb = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Network analytics/Datasets_Network Analytics/facebook.csv")

fb_1 = nx.Graph()
fb_1 = nx.from_pandas_edgelist(fb, source = '1', target = '9')
print(nx.info(fb))
 # Degree Centrality
d = nx.degree_centrality(fb_1) 
print(d) 

pos = nx.circular_layout(fb_1)
nx.draw_networkx(fb_1, pos, node_size = 25, node_color = 'blue')

# closeness centrality
closeness = nx.closeness_centrality(fb_1)
print(closeness)
 # Betweeness_Centrality
A = nx.betweenness_centrality(fb_1)
print(A)
# Eigen vector centrality
evc = nx.eigenvector_centrality(fb_1) 
print(evc)

# cluster coefficient
cluster_coeff = nx.clustering(fb_1)
print(cluster_coeff)

# Average clustering
ac = nx.average_clustering(fb_1) 
print(ac)
