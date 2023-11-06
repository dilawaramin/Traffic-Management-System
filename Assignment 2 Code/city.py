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
            node_name = f"I{x},{y}"
            City.add_node(node_name, pos=(x, y), type='intersection')
    
    # Create streets (represented as edges) to connect intersections
    for x in range(grid_size):
        for y in range(grid_size):
            if x < grid_size - 1:
                node1 = f"I{x},{y}"
                node2 = f"I{x+1},{y}"
                City.add_edge(node1, node2, road_type="main")
            if y < grid_size - 1:
                node1 = f"I{x},{y}"
                node2 = f"I{x},{y+1}"
                City.add_edge(node1, node2, road_type="main")
        
        
    return City


def print_city(City):
    """
    Function to create visual representation of city object
        Parameters: 
            City: City graph object created by generate_city()
        Returns:
            None
    """
    # Create custom positions for all nodes
    pos = {node: (City.nodes[node]['pos'][0], City.nodes[node]['pos'][1]) for node in City.nodes}
    # Formatting for the graph that is to be shown
    nx.draw(City, pos, with_labels=True, node_size=200, node_color='lightblue', edge_color='gray', font_size=8, font_color='black')
    # Show the city graph
    plt.show()
    
    return 


def print_start_end(City, SP, EP):
    """
    Function that prints the city and highlights starting point and destination
        Parameters: 
            City: City graph object created by generate_city()
            SP: starting point
            EP: Ending point
        Returns:
            None
    """
    # Create custom positions for all nodes
    pos = {node: (City.nodes[node]['pos'][0], City.nodes[node]['pos'][1]) for node in City.nodes}
    # color mapping
    def __node_color(node):
        if node == SP or node == EP:
            return 'red' 
        else:
            return 'lightblue'
    # create list of colors for nodes
    node_colors = [__node_color(node) for node in City.nodes()]
    # Formatting for the graph that is to be shown
    nx.draw(City, pos, with_labels=True, node_size=200, node_color=node_colors, edge_color='gray', font_size=8, font_color='black')
    # Show the city graph
    plt.show()

    return 


# Test city
City = generate_city(10, 10, 10)
