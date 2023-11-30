# Astar.py

"""
Author:         Shane Shamku
Date Created:   11/27/2023

A* Search algorithm for traffic management system. Contains all functions that are
directly related to running the main A* algorithm.
Uses city objects and helper functions from city.py
"""

########################## IMPORTS ####################################################

import numpy as np
import heapq
import city as C
import time




########################## A* AND HELPER ####################################################

def heuristic(node, goal):
    """
    Heuristic function for A* algorithm, calculates Manhattan distance.
    """
    x1, y1 = C.current_xy(node)
    x2, y2 = C.current_xy(goal)
    return abs(x1 - x2) + abs(y1 - y2)

def astar_search(city, start, end):
    """
    Perform A* search on the city graph from start to end node.
    Parameters:
        city: City graph object created by generate_city()
        start: Starting node in format "I{x},{y}"
        end: Destination node in format "I{x},{y}"
    Returns:
        path: List containing nodes on the path from start to end
    """
    # Priority queue for A*
    frontier = []
    heapq.heappush(frontier, (0, start, [start]))
    
    # Costs and visited dictionaries
    costs = {start: 0}
    visited = {start: None}

    # Start timer
    start_time = time.time()

    while frontier:
        _, current, current_path = heapq.heappop(frontier)
        
        if current == end:
            # Stop timer
            end_time = time.time()
            # Print the path and metrics
            print_path_and_metrics(city, start, end, current_path, start_time, end_time, visited)
            return current_path

        for neighbor in city.neighbors(current):
            new_cost = costs[current] + 1  # Cost between nodes is assumed to be 1
            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, end)
                heapq.heappush(frontier, (priority, neighbor, current_path + [neighbor]))
                visited[neighbor] = current

    # If no path found
    end_time = time.time()
    print("No path found.")
    print_path_and_metrics(city, start, end, [], start_time, end_time, visited)
    return []




########################## PRINT FUNCTION ####################################################

def print_path_and_metrics(city, start, end, path, start_time, end_time, visited):
    """
    Print the path and metrics for the A* search.
    Parameters:
        city: City graph object created by generate_city()
        start: Starting point node
        end: Ending point node
        path: List containing nodes on the path
        start_time: Time when A* started
        end_time: Time when A* ended
        visited: Set of visited nodes during A*
    """
    duration = end_time - start_time
    num_computations = len(visited)
    # Output similar to qlearn.py
    C.print_path(city, start, end, path)
    print(f"Path found by A* in {duration:.4f} seconds with {num_computations} computations.")

