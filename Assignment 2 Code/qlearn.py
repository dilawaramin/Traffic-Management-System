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

# Global Variables
# City:
city = C.generate_city(7, 7)
# q table
q_values = C.create_q_table(city)
# actions
# Define actions (0 = up, 1 = right, 2 = down, 3 = left)
actions = ['up', 'right', 'down', 'left']



# def visualize_path(Q, start_node, end_node):
#     current_node = start_node
#     while current_node != end_node:
#         best_action = max(Q[current_node], key=Q[current_node].get)
#         print(f"From {current_node} to {best_action}")
#         current_node = best_action

# Helper Functions

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
    if np.random.random() < epsilon:
       return np.argmax(q_values[horizontal, vertical])
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
    if actions[action] == 'up' and horizontal > 0:
        new_horz -= 1
    elif actions[action] == 'right' and vertical < v - 1:
        new_vert += 1
    elif actions[action] == 'down' and horizontal < h - 1:
        new_horz += 1
    elif actions[action] == 'left' and vertical > 0:
        new_vert -= 1
    return new_horz, new_vert
    

# Main q learning function
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
    # Get all nodes and rewards dictionary on standby
    nodes = C.get_nodes(city)
    rewards = C.get_rewards(city)
    # Initialize the destination node
    C.set_destination(city, end_node)
    # Run through the algorithm according to predined num_episodes variable
    for episode in range(num_episodes):
        current_node = start_node
        curr_horz, curr_vert = C.current_xy(current_node)
        while C.is_terminal_state(city, current_node) != True:
            
            # choose next action index
            action = get_next_action(curr_horz, curr_vert, epsilon)
            
            # store old node position, obtain new node position
            old_horz = curr_horz
            old_vert = curr_vert
            curr_horz, curr_vert = get_next_location(curr_horz, curr_vert, action)
            
            # get reward for action, calculate temporal difference
            reward = rewards[C.current_node(curr_horz, curr_vert)]
            old_q_value = q_values[old_horz, old_vert, action]
            temp_difference = reward + (discount_factor * np.max(q_values[curr_horz, curr_vert])) - old_q_value
            
            # update q-value for the previous state and action pair
            new_q_value = old_q_value + (learning_rate * temp_difference)
            q_values[curr_horz, curr_vert, action] = new_q_value

            # debugging prints
            
        # progress prints
        print(f"Episode {episode}/{num_episodes} complete!")



### TESTING ###

print("Begin testing:")
# generate city
city = C.generate_city(5, 5)
#C.print_city(city)

# Set start and end points NOTE: make sure they are not the outer nodes
start_node = "I1,1"
end_node = "I2,2"
C.print_start_end(city, start_node, end_node)

# Q-Learning hyperparameters
num_episodes = 100
learning_rate = 0.1
discount_factor = 0.9
exploration_prob = 0.5

 # Run Q-learning algorithm
q_learning(city, start_node, end_node, num_episodes, learning_rate, discount_factor, exploration_prob)
print("Finished running q_learning()")
