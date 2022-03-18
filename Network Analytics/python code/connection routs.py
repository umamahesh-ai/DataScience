

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

route = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Network analytics/Datasets_Network Analytics/connecting_routes.csv")
route = route.iloc[0:500, 0:10]
route.columns

del route['Unnamed: 6'] # removing the unwanted (zero variance) column

route.columns = ["flights", "ID", "main Airport" , "main Airport ID", "Destination","Destination  ID","haults","machinerY"]
del route['haults']
route_1 = nx.Graph()

route_1 = nx.from_pandas_edgelist(route, source = 'main Airport', target = 'Destination')
print(nx.info(route_1))   # 227 nodes,265 edges
 # Degree Centrality
b = nx.degree_centrality(route_1) 
print(b)
pos = nx.spring_layout(route_1, k = 0.15)
nx.draw_networkx(route_1,pos, node_size = 25, node_color = 'blue')
# closeness centrality
closeness = nx.closeness_centrality(route_1)  
print(closeness)
# Betweeness Centrality 
betweenness = nx.betweenness_centrality(route_1)   
print(betweenness)
