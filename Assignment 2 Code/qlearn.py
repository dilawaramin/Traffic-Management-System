"""
Q Learning algorithm for traffic management system.
Use city objects from city.py

Author: Dilawar Amin
Date: 11/05/2023
"""
# Import city, to create and print cities
import city as C
import random
import numpy as np
import copy

# Global Variables
# City:
city = C.generate_city(7, 7)
# q table
q_values = C.create_q_table(city)
# actions
# Define actions (0 = up, 1 = right, 2 = down, 3 = left)
actions = ['up', 'right', 'down', 'left']




########################## Initialize Q-Learning ####################################################

def init_qlearn():
    """
    Function that initializes the starting point via user inputs
        Parameters: 
            None
        Returns:
            num_episodes: How many times the algorithm will run a session to train the agent
            learning_rate: How aggresively the algorithm will update
            discount_factor: How strongly to value future v.s. current rewards
            exploration_prob: How often to make random moves, encouraging exploration v.s. exploitation
            
    """
    print("Lets initialize some important values for our Q-Learning algorithm.\n")
    num_episodes = int(input("Episodes: The number of training episodes you wish to run (Ex. 1000): "))
    learning_rate = float(input("Learning Rate: Controls how quickly the agent learns (Ex. 0.1): "))
    discount_factor = float(input("Discount: Weighs immediate v.s. future rewards (Ex. 0.9 for near-term focus):"))
    exploration_prob = float(input("Exploration: Rate at which to Balance exploration v.s. exploitation (Ex. 0.9 for exploration)."))
    print()
    return num_episodes, learning_rate, discount_factor, exploration_prob





########################## Helper Functions ####################################################

def get_next_action(horizontal, vertical, epsilon):
    """
    Function that determines the next action to take
        Parameters: 
            Horizontal: Current horizontal index
            Vertical: Current vertical index
            Epsilon: Exploration v.s. exploitation factor
        Returns:
            Integer between 0 and 3 which corresponds with action
    """
    # if randomly chosen value less than epsilon, use q table value
    randt = np.random.random()
    #print()
    #print(randt)
    if randt < epsilon:
        action = np.argmax(q_values[horizontal, vertical])
        #print(f"Action: {action}\n")
        return action
    # else select a random action
    else:
        return np.random.randint(4)
    
def get_next_location(horizontal, vertical, action):
    """
    Function that takes current node position (x, y) and an action and 
    returns the new node location 
        Parameters: 
            Horizontal: Current horizontal index
            Vertical: Current vertical index
            action: Index number of action
        Returns:
            new_horz: new x position of node
            new_vert: new y position of node
    """
    new_horz = horizontal
    new_vert = vertical
    h, v = C.get_dimensions(city)
    if actions[action] == 'up' and new_vert < v - 1:
        new_vert += 1
    elif actions[action] == 'right' and new_horz < h - 1:
        new_horz += 1
    elif actions[action] == 'down' and new_vert > 0:
        new_vert -= 1
    elif actions[action] == 'left' and new_horz > 0:
        new_horz -= 1
    return new_horz, new_vert

def visualize_path(q_values, city, start, end):
    """
    Function that takes final Q-values to show what the agent learned 
        Parameters: 
            q_values: Q-table, make sure to train agent before hand
        Returns:
            None
    """
    # initialize starting node and path
    current_node = start
    C.set_destination(city, end)
    path = [start]
    print(start)
    while C.is_terminal_state(city, current_node) != True:
        curr_x, curr_y = C.current_xy(current_node)
        # Use q values to find best action and make move
        action = np.argmax(q_values[curr_x, curr_y])
        print(f"H:{curr_x}, V:{curr_y}, A:{actions[action]}")
        new_x, new_y = get_next_location(curr_x, curr_y, action)
        current_node = C.current_node(new_x, new_y)
        print(f"New Node: H:{new_x}, V:{new_y}\n")
        path.append(current_node)
    # call function in city.py to create visual graph
    C.print_path(city, start, end, path)
    # i think thats it
    return



########################## Main q learning function  ####################################################

def q_learning(city, start_node, end_node, num_episodes, learning_rate, discount_factor, epsilon):
    """
    Q Learning algorithm to route a vehicle from point A to B within the city
        Parameters:
            City: graph object representing city
            start_node: Starting point of vehicle. Represnting as "I{X},{Y}" format
            end_node: Destination of vehicle
            num_episodes: Total # of times that q learning algorithm should 
                           repeat to train the A.I.
            learning_rate: Rate defining how aggressively we wish for agent to learn
            discount_rate: Factor by which to multiply future rewards (instant vs later reward)
            exploration_prob: Factor determining exploration v.s. exploitation
        Returns:
            None
    """
    #Q = {node: {neighbor: 0 for neighbor in city.neighbors(node)} for node in city.nodes()} # chatgpt
    # Get all nodes on standby
    nodes = C.get_nodes(city)
    # set destination
    C.set_destination(city, end_node)
    # obtain list of reward
    rewards = C.get_rewards(city)
    print(rewards[end_node])
    C.print_city(city)
    # Run through the algorithm according to predined num_episodes variable
    for episode in range(num_episodes):
        #print(f"Starting Episode {episode + 1}.")
        current_node = start_node
        curr_horz, curr_vert = C.current_xy(current_node)
        #print(C.is_terminal_state(city, "I0,1"))
        # initialize a list for explored paths
        path = [current_node]
        C.set_reward(city, current_node, -50)
        while C.is_terminal_state(city, current_node) != True:
            path.append(current_node)
            # debugging print
            #print(f"Current Node: {current_node}.")
            # set a negative reward for returning to same node
            if rewards[current_node] != 250 and rewards[current_node] != -100:
                #print("setting node to -50")
                C.set_reward(city, current_node, -50)
                rewards = C.get_rewards(city)
            # choose next action index
            action = get_next_action(curr_horz, curr_vert, epsilon)
            #print(f"Action: {actions[action]}")
            
            # store old node position, obtain new node position
            old_horz = curr_horz
            old_vert = curr_vert
            curr_horz, curr_vert = get_next_location(curr_horz, curr_vert, action)
            current_node = C.current_node(curr_horz, curr_vert)
            #print(f"New node: {current_node}")
            
            # get reward for action, calculate temporal difference
            reward = rewards[current_node]
            old_q_value = q_values[old_horz, old_vert, action]
            temp_difference = reward + (discount_factor * np.max(q_values[curr_horz, curr_vert])) - old_q_value
            #print(f"Reward: {reward}")
            #print(f"Old Q : {old_q_value}")
            #print(f"Temp D: {temp_difference}")
            
            # update q-value for the previous state and action pair
            new_q_value = old_q_value + (learning_rate * temp_difference)
            q_values[curr_horz, curr_vert, action] = new_q_value
            #print(f"New Q:{new_q_value}")
            #print(f"Is terminal:{C.is_terminal_state(city, current_node)}\n")
            
            # debugging prints
        for node in path:
            C.set_reward(city, node, -1)
            
        # progress prints
        print(f"Episode {episode + 1}/{num_episodes} complete!")
    print()



########################## TESTING ####################################################

def main():
    
    print("Begin testing:")
    #C.print_city(city)
    city = C.generate_city(7, 7)
    # Set start and end points NOTE: make sure they are not the outer nodes
    start_node = "I1,1"
    end_node = "I4,4"
    C.print_start_end(city, start_node, end_node)

    # Q-Learning hyperparameters
    num_episodes = 5000
    learning_rate = 0.9
    discount_factor = 0.5
    epsilon = 0.9

    print("prepare to start Q-Learning:")
    # save original q table
    og_q = copy.deepcopy(q_values)
    # Run Q-learning algorithm
    q_learning(city, start_node, end_node, num_episodes, learning_rate, discount_factor, epsilon)
    print("Finished running q_learning()\n")
    print(f"Original q table: {og_q}\n")
    print(f"Updated q table: {q_values}\n")
    
    visualize_path(q_values, city, start_node, end_node)
    print("stop here")
    

if __name__ == '__main__':
    main()
    print("Program terminated successfully")
