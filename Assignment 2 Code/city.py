"""
Generate city and all related functions
To be used by Q-Learning algorithm to create traffic management system

Author: Dilawar Amin
Date: 11/05/2023
"""
########################## IMPORTS ####################################################

# Use NetworkX library, represent city as a graph with nodes and edges
# NOTE: Use "pip install networkx" beforehand
import networkx as nx 

# NOTE: Use "pip install matplotlib" to use the visualization functions
import matplotlib.pyplot as plt

# NOTE: Use "pip install numpy" before hand as well
import numpy as np
import random





########################## GLOBAL VARIABLES ####################################################

REWARD = 20            # Reward for goal
NEIGHBOURS = 20         # Reward for being close to goal (neighbouring nodes)
VISITED = -50           # Negative reward for states that have already been visited
DEFAULT = -1            # Default reward for regular states
TERMINAL = -100         # Negative reward associated with terminal states
REVISIT_PENALTY = -99   # Extra penalty to be added to reward function for revisiting states
TRAFFIC = -20     # Negative reward for traffic nodes




########################## City generation  ####################################################

def generate_city(horizontal, vertical):
    """
    Function that creates cities.
    All outer nodes are considered terminal states. Starting and End point, as well as routing,
    must take place within.
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
    for x in range(horizontal):
        for y in range(vertical):
            if x ==  horizontal - 1 or y == vertical - 1:
                node_name = f"I{x},{y}"
                City.add_node(node_name, pos=(x, y), type='intersection', reward=TERMINAL) # terminal states
            elif x ==  0 or y == 0:
                node_name = f"I{x},{y}"
                City.add_node(node_name, pos=(x, y), type='intersection', reward=TERMINAL) #terminal states
            else:    
                node_name = f"I{x},{y}"
                City.add_node(node_name, pos=(x, y), type='intersection', reward=DEFAULT) # default reward
    
    # Create streets (represented as edges) to connect intersections
    for x in range(horizontal):
        for y in range(vertical):
            if x < horizontal - 1:
                node1 = f"I{x},{y}"
                node2 = f"I{x+1},{y}"
                City.add_edge(node1, node2, road_type="main")
            if y < vertical - 1:
                node1 = f"I{x},{y}"
                node2 = f"I{x},{y+1}"
                City.add_edge(node1, node2, road_type="main")
    # Create Q-Table for city
    q_values = create_q_table(City)
    return City, q_values




########################## Initialization Functions  ####################################################

def initialize_city():
    """
    Function that initializes the city object via user inputs
        Parameters: 
            None
        Returns:
            city: City object, consisting of nodes and edges via networkx library
    """
    print("\nNOTE: Actual city dimensions will be 1 unit x 1 unit smaller than input, due " +
          "to coordinates starting at (0, 0). Outter edge of city is used as a terminal "
          + "state perimeter. We do not recommend a city size of greater than 15x15, for " +
          "performance related reasons. Minimum allowed city size is 4x4, which translates" +
           "to a 2x2 useable grid.\n")
    horizontal = int(input("Please enter horizontal dimension of city: "))
    while horizontal <= 3:
            horizontal = int(input("Invalid city size. Please enter horizontal dimension: "))
    vertical = int(input("Please enter vertical dimension of city: "))
    while vertical <= 3:
            vertical = int(input("Invalid city size. Please enter vertical dimension: "))
    print()
    city, q_values = generate_city(horizontal, vertical)
    return city, q_values
    

def get_start(City):
    """
    Function that initializes the starting point via user inputs
        Parameters: 
            None
        Returns:
            SP: Node name of starting point
    """
    # Remind user about invalid inputs
    print("NOTE: Cannot use outer points as start or end, as they are reserved as " +
          "City perimeter. For example, (0, 0) is invalid. and for a city size of 7x7, " +
          "a starting point of (6, 6) is invalid.\n")
    # get city dimensions for error checking
    x, y = get_dimensions(City)
    start_X = int(input("Enter X coordinate of starting point: "))
    while start_X >= x or start_X <= 0:
            start_X = int(input("Invalid input. Please enter Starting point X coordinate: "))
    start_Y = int(input("Enter Y coordinate of starting point: "))
    while start_Y >= y or start_Y <= 0:
            start_Y = int(input("Invalid input. Please enter Starting point Y coordinate: "))
    SP = current_node(start_X, start_Y)
    print()
    return SP

def get_destination(City, SP):
    """
    Function that initializes the starting point via user inputs. Will call "set_definition(City, DP)"
    to set the reward for destination as well.
        Parameters: 
            None
        Returns:
            SP: Node name of starting point
    """
    # get city dimensions for error checking
    x, y = get_dimensions(City)
    # parse starting point
    s, p = current_xy(SP)
    start_X = int(input("Enter X coordinate of Destination: "))
    while start_X >= x or start_X <= 0:
            start_X = int(input("Invalid input. Please enter Destination X coordinate: "))
    start_Y = int(input("Enter Y coordinate of Destination: "))
    while start_Y >= y or start_Y <= 0:
            start_Y = int(input("Invalid input. Please enter Destination Y coordinate: "))
    DP = current_node(start_X, start_Y)
    if start_X == s and start_Y == p:
        print("Destination cannot be the same as starting point!!!")
        DP = get_destination(City, SP)
        # Set reward for destination
    set_destination(City, DP)
    print()
    return DP




########################## City Manipulation ####################################################

def generate_traffic(City):
    """
    Function that generates procedurally generates traffic, according to 
    Function will consider the overall size of the city, to prevent causing unrealistic
    over congestion. 
    TO BE CALLED AFTER INITIALIZING CITY, STARTING PONIT, AND  DESTINATION.
        Parameters:
            City: city object on which to generate traffic
        Returns:
            None - function modifies existing city object
    """
    city_x, city_y = get_dimensions(City)
    total_perimeter = city_x + city_y
    # cases for city sizes
    # 2 spots of congestion for a city 15x15 or smaller
    if total_perimeter <= 15:
        for i in range(1):
            __generate_traffic(City)
    # 4 spots of congestion for a city 10x10 or smaller
    elif total_perimeter <= 20:
        for i in range(3):
            __generate_traffic(City)
    # 7 spots of congestion for a city 15x15 or smaller
    elif total_perimeter <= 30:
        for i in range (6):
            __generate_traffic(City)
    # 9 spots of congestion for a city 25x25 or smaller
    elif total_perimeter <= 50:
        for i in range(8):
            __generate_traffic(City)
            
    return




########################## Helper functions  ####################################################

def get_dimensions(City):
    """
    Function that finds the dimensions of the city object
        Parameters: 
            City: City graph object created by generate_city()
        Returns:
            Horizontal: Horizontal length of city
            Vertical: Vertical length of City
    """
    names = get_nodes(City)
    horizontal = 0
    vertical = 0
    for name in names:
        X, Y = current_xy(name)
        if X > horizontal:
            horizontal = X
        if Y > vertical:
            vertical = Y
        # PREVIOUS VERSION:
        # if int(name[1]) > horizontal:
        #     horizontal = int(name[1])
        # if int(name[3]) > vertical:
        #     vertical = int(name[3])
    return horizontal + 1, vertical + 1

def create_q_table(City):
    """
    Function that initializes empty q table
        Parameters: 
            City: City graph object created by generate_city()
        Returns:
            q_values: 3d array of q values intialized to 0
    """
    h, v = get_dimensions(City)
    q_values = np.zeros((h, v, 4))
    return q_values

def is_terminal_state(City, name):
    """
    Function that determines if a node is a terminal state or not
        Parameters: 
            City: City graph object created by generate_city()
            name: Name of node to be checked
        Returns:
            True or False
    """
    rewards = nx.get_node_attributes(City, 'reward')
    # check if terminal state then return
    bool = int(rewards[name]) == TERMINAL or int(rewards[name]) == REWARD
    return bool

def get_rewards(City):
    """
    Function that returns a dictionary with rewards of all nodes
        Parameters: 
            City: City graph object created by generate_city()
        Returns:
            rewards: Dictionary with rewards, name of nodes as keys
    """
    rewards = nx.get_node_attributes(City, 'reward')
    return rewards

def get_nodes(City):
    """
    Function that returns a list of all nodes
        Parameters: 
            City: City graph object created by generate_city()
        Returns:
            Nodes: List of all nodes in city
    """
    nodes = City.nodes()
    return nodes

def get_traffic_nodes(City):
    """
    Function that returns a list of all traffic nodes in a given city.
    To be run after generating traffic with generate_traffic()
        Parameters: 
            City: City graph object created by generate_city()
        Returns:
            traffic_nodes: List of all traffic nodes
    """
    # Initialize traffic nodes, all nodes, and rewards
    traffic_nodes = []
    nodes = get_nodes(City)
    rewards = get_rewards(City)
    for node in nodes:
        if rewards[node] == TRAFFIC:
            traffic_nodes.append(node)
    return traffic_nodes

def get_random_node(City):
    """
    Function that returns a random node.
        Parameters: 
            City: City graph object created by generate_city()
        Returns:
            Nodes: List of all nodes in city
    """
    # various traffic depending on city size
    city_x, city_y = get_dimensions(City)
    rand_x = random.randint(1, city_x - 1)
    rand_y = random.randint(1, city_y - 1)
    random_node = current_node(rand_x, rand_y)

    return random_node

def current_node(horizontal, vertical):
    """
    Function that returns returns name of current node
        Parameters: 
            Horizontal: Current horizontal index
            Vertical: Current vertical index
        Returns:
            name: Name of current node
    """
    name = f"I{horizontal},{vertical}"
    return name

def SET_ASIDE_current_xy(name):     # set aside while testing V.2 of this function
    """
    Function that returns returns x and y position of the current node
        Parameters: 
            name: Name of current node   
        Returns:
            Horizontal: Current horizontal index
            Vertical: Current vertical index
    """
    # make sure name is valid
    if len(name) < 3:
        print("\nInvalid name")
        return
    else:
        horizontal = name[1]
        vertical = name[3]
    return int(horizontal), int(vertical)

def set_destination(City, DP):
    """
    Function to set the reward for the destination point. Sets a medium reward for neighbors of dest.
        Parameters: 
            City: City graph object created by generate_city()
            DP: Name of node which is destination
        Returns:
            None
    """
    neighbors = nx.neighbors(City, DP)
    rewards = get_rewards(City)
    for node in neighbors:
        h, v = current_xy(node)
        if rewards[node] != -100:
            nx.set_node_attributes(City, {f"I{h},{v}":{'reward':NEIGHBOURS}})
    h, v = current_xy(DP)
    nx.set_node_attributes(City, {f"I{h},{v}":{'reward':REWARD}})
    return

def set_reward(City, node, reward):
    """
    Function to set a custom reward for a given node
        Parameters: 
            City: City graph object created by generate_city()
            DP: Name of node which is destination
            reward: Custom reward amount
        Returns:
            None
    """
    h, v = current_xy(node)
    nx.set_node_attributes(City, {f"I{h},{v}":{'reward':reward}})
    return

def current_xy(name):
    """
    Function that returns returns x and y position of the current node.
        Parameters: 
            name: Name of current node   
        Returns:
            Horizontal: Current horizontal index
            Vertical: Current vertical index
    """
    # if city dimensions are single digits
    if len(name) == 4:
        horizontal = name[1]
        vertical = name[3]
    # Parse the string if coordinates are double digit
    else:
        name = name[1:]
        horizontal, vertical = name.split(',')
        horizontal = horizontal
        vertical = vertical
    return int(horizontal), int(vertical)

def __generate_traffic(City):
    """
    Private function that generates traffic for generate_traffic() function:
        Parameters: 
            City: city object  
        Returns:
            None: modifies existing city object
    """
    # Helper
    rewards = get_rewards(City)
    # Create random traffic starting point
    traffic_node = get_random_node(City)
    while rewards[traffic_node] != DEFAULT:
        traffic_node = get_random_node(City)
    # get neighbouring nodes to create congestion
    trafficNX = nx.neighbors(City, traffic_node)
    traffic = []
    for node in trafficNX:
        traffic.append(node)
    traffic.append(traffic_node)
    for node in traffic:
        if rewards[node] == DEFAULT:
            set_reward(City, node, TRAFFIC)
    return
    



########################## Auxiliary print functions ####################################################

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
    labels = nx.get_node_attributes(City, 'reward')
    nx.draw(City, pos, labels=labels, node_size=200, node_color='lightblue', edge_color='gray', font_size=8, font_color='black')
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
    # Traffic nodes
    traffic = get_traffic_nodes(City)
    # color mapping
    def __node_color(node):
        if node == SP:
            return 'blue'
        elif node == EP:
            return 'red'
        elif node in traffic:
            return 'orange' 
        else:
            return 'lightblue'
    # create list of colors for nodes
    node_colors = [__node_color(node) for node in City.nodes()]
    # Formatting for the graph that is to be shown
    nx.draw(City, pos, with_labels=True, node_size=200, node_color=node_colors, edge_color='gray', font_size=8, font_color='black')
    # Show the city graph
    plt.show()

    return 

def print_path(City, SP, EP, path):
    """
    Function that prints the city and highlights starting point and destination
        Parameters: 
            City: City graph object created by generate_city()
            SP: starting point
            EP: Ending point
            path: List containing nodes on path
        Returns:
            None
    """
    # Create custom positions for all nodes
    pos = {node: (City.nodes[node]['pos'][0], City.nodes[node]['pos'][1]) for node in City.nodes}
    # Traffic nodes
    traffic = get_traffic_nodes(City)
    # color mapping
    def __node_color(node):
        if node == SP:
            return 'blue'
        elif node == EP:
            return 'red' 
        elif node in path:
            return 'green'
        elif node in traffic:
            return 'orange'
        else:
            return 'lightblue'
    # create list of colors for nodes
    node_colors = [__node_color(node) for node in City.nodes()]
    # Formatting for the graph that is to be shown
    nx.draw(City, pos, with_labels=True, node_size=200, node_color=node_colors, edge_color='gray', font_size=8, font_color='black')
    # Show the city graph
    plt.show()

    return 




########################## TESTING  ####################################################

def main():
    city, q_values = initialize_city()
    # print_city(city)

    # if is_terminal_state(city, "I1,1"):
    #     print('This is a terminal state')
    # else:
    #     print("not a terminal state")
    # h, v = get_dimensions(city)
    # print(f"Horizontal: {h}")
    # print(f"Vertical: {v}")
    # print_city(city)
    # set_destination(city, "I3,3")
    # print_city(city)
    SP = get_start(city)
    print(SP)
    print()
    DP = get_destination(city, SP)
    print(DP)
    generate_traffic(city)
    print_start_end(city, SP, DP)
    nodes = city.nodes()
    print(nodes)
    print(type(nodes))

if __name__ == '__main__':
    main()
    