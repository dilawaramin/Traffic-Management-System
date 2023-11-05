"""
Generate city and all related functions
To be used by Q-Learning algorithm to create traffic management system
Author: Dilawar Amin
Date: 11/05/2023
"""
# Use NetworkX library, represent city as a graph with nodes and edges
# NOTE: Use "pip install networkx" beforehand
import networkx as nx 
# NOTE: Use "pip install matplotlib" to use the visualization functions
import matplotlib.pyplot as plt
import numpy as np
import random
seed = 0
random.seed(0)
np.random.seed(0)


def generate_city(grid_size, num_intersections, num_streets):
    """
    Function that creates cities
        Parameters:
            grid_size: desired overall size of city
            num_intersections: desired # of intersections
            num_streets: desired number of streets
        Returns:
            G: Graph, representing city as a network of nodes and edges
    """
    # Initialize Graph object
    City = nx.Graph()

    # Create Intersections (represented as nodes) following grid structure
    for x in range(grid_size):
        for y in range(grid_size):
            node_name = f"I{x}, {y}"
            City.add_node(node_name, pos=(x, y), type='intersection')
    
    # Create streets (represented as edges) to connect intersections
    for x in range(grid_size):
        for y in range(grid_size):
            if x < grid_size - 1:
                node1 = f"I{x}, {y}"
                node2 = f"I{x+1}, {y}"
                City.add_edge(node1, node2, road_type="main")
            if y < grid_size - 1:
                node1 = f"I{x}, {y}"
                node2 = f"I{x}, {y+1}"
                City.add_edge(node1, node2, road_type="main")
        
        
    return City

# Test city
City = generate_city(10, 10, 10)
# # Visualize city
# nx.draw(City)
# plt.margins(0.2)
# plt.show()
# Define a custom node position layout
pos = {node: (City.nodes[node]['pos'][0], City.nodes[node]['pos'][1]) for node in City.nodes}

# Draw the graph with the spring layout
nx.draw(City, pos, with_labels=True, node_size=200, node_color='lightblue', edge_color='gray', font_size=8, font_color='black')

plt.show()