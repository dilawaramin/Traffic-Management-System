"""
Generate city and all related functions
Author: Dilawar Amin
Date: 11/05/2023
"""
# Use NetworkX library, represent city as a graph with nodes and edges
# NOTE: Use "pip install networkx" beforehand
import networkx as nx 


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
    
    return