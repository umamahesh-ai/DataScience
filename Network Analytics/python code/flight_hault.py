
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

flight = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Network analytics/Datasets_Network Analytics/flight_hault.csv")
flight = flight.iloc[0:500, 0:12]
flight.columns
flight.columns = ["ID","Name","City","Country","IATA_FAA","ICAO","Latitude","Longitude","Altitude","Time","DST","Tz database time"]

route_new = nx.Graph()
route_new = nx.from_pandas_edgelist(flight, source = "Name", target = "Country")
print(nx.info(route_new))  # 518 nodes and 500 edges
#Degree ofn centrality
A = nx.degree_centrality(route_new)
print(A)

pos = nx.spring_layout(route_new, k = 0.15)
nx.draw_networkx(route_new, pos, node_size = 25, node_color = 'blue')

closeness = nx.closeness_centrality(route_new)
print(closeness)

betweenness = nx.betweenness_centrality(route_new)
print(betweenness)
