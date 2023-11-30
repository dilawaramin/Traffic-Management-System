"""
Author:         Javier Rodrigues
Date Created:   11/05/2023

BFS Search algorithm for traffic management system. Contains all functions that are
directly related to running the main BFS algorithm.
Uses city objects and helper functions from city.py
"""

########################## IMPORTS ####################################################

import city as C
import time
from collections import deque




########################## BFS ALGORITHM ####################################################

def bfs_search(city, start, end):
    """
    Perform BFS search on the city graph from start to end node.
    Parameters:
        city: City graph object created by generate_city()
        start: Starting node in format "I{x},{y}"
        end: Destination node in format "I{x},{y}"
    Returns:
        path: List containing nodes on the path from start to end
    """
    visited = set()
    queue = deque([(start, [start])])
    terminal_nodes = C.get_perimeter_nodes(city)
    traffic_nodes = C.get_traffic_nodes(city)

    start_time = time.time()

    while queue:
        (vertex, current_path) = queue.popleft()
        if vertex not in visited and vertex not in terminal_nodes and vertex not in traffic_nodes:
            if vertex == end:
                path = current_path
                break

            visited.add(vertex)

            for neighbor in city.neighbors(vertex):
                if neighbor not in visited and neighbor not in terminal_nodes and neighbor not in traffic_nodes:
                    queue.append((neighbor, current_path + [neighbor]))
    
    end_time = time.time()
    if not path:
        print("BFS FAILED TO FIND PATH.")
        return
    else:
        print_path_and_metrics(city, start, end, path, start_time, end_time, visited)
        return path




########################## PRINT FUNCTION ####################################################

def print_path_and_metrics(city, start, end, path, start_time, end_time, visited):
    """
    Print the path and metrics for the BFS search.
    Parameters:
        city: City graph object created by generate_city()
        start: Starting point node
        end: Ending point node
        path: List containing nodes on the path
        start_time: Time when BFS started
        end_time: Time when BFS ended
        visited: Set of visited nodes during BFS
    """
    duration = end_time - start_time
    num_computations = len(visited)
    # Output similar to qlearn.py
    C.print_path(city, start, end, path)
    print(f"Path found by BFS in {duration:.4f} seconds with {num_computations} computations.")
