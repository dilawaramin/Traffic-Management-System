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
    num_episodes = int(input("Episodes: The number of training episodes you wish to run (Suggested: 1000): "))
    learning_rate = float(input("Learning Rate: Controls how quickly the agent learns (Suggested: 0.9): "))
    discount_factor = float(input("Discount: Weighs immediate v.s. future rewards (Suggested: 0.9): "))
    epsilon = float(input("Exploration: Rate at which to Balance exploration v.s. exploitation (Suggested: 0.9): "))
    print()
    return num_episodes, learning_rate, discount_factor, epsilon




########################## Helper Functions ####################################################

def get_next_action(q_values, horizontal, vertical, epsilon):
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
    
def get_next_location(city, horizontal, vertical, action):
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
    path = [start]
    #print(start)
    while C.is_terminal_state(city, current_node) != True:
        curr_x, curr_y = C.current_xy(current_node)
        # Use q values to find best action and make move
        action = np.argmax(q_values[curr_x, curr_y])
        #print(f"H:{curr_x}, V:{curr_y}, A:{actions[action]}")
        new_x, new_y = get_next_location(city, curr_x, curr_y, action)
        current_node = C.current_node(new_x, new_y)
        #print(f"New Node: H:{new_x}, V:{new_y}\n")
        path.append(current_node)
    # call function in city.py to create visual graph
    C.print_path(city, start, end, path)
    # i think thats it
    return



########################## Main Q-learning function  ####################################################

def q_learning(city, start_node, end_node, num_episodes, learning_rate, discount_factor, epsilon, q_values):
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
    # obtain list of rewards
    rewards = C.get_rewards(city)
    
    # Run through the algorithm according to predined num_episodes variable
    for episode in range(num_episodes):
        
        # set current node
        current_node = start_node
        curr_horz, curr_vert = C.current_xy(current_node)
        
        # begin looping until a terminal state is reached
        while C.is_terminal_state(city, current_node) != True:
            # debugging print
            #print(f"Current Node: {current_node}.")

            # choose next action index
            action = get_next_action(q_values, curr_horz, curr_vert, epsilon)
            #print(f"Action: {actions[action]}")
            
            # store old node position
            old_horz = curr_horz
            old_vert = curr_vert
            
            # Obtain new location, with action
            curr_horz, curr_vert = get_next_location(city, curr_horz, curr_vert, action)
            
            # Contruct X and Y coordinates into new current node
            current_node = C.current_node(curr_horz, curr_vert)
            #print(f"New node: {current_node}")
            
            # get reward for action
            reward = rewards[current_node]
            #print(f"Reward: {reward}")

            # Find Q value of old position
            old_q_value = q_values[old_horz, old_vert, action]
            #print(f"Old Q : {old_q_value}")
            
            # Calculate Temporal Difference
            temp_difference = reward + (discount_factor * np.max(q_values[curr_horz, curr_vert])) - old_q_value
            #print(f"Temp D: {temp_difference}")
            ##### temp_difference = reward + (discount_factor * np.max(q_values[curr_horz, curr_vert])) - old_q_value
            # temp_difference = reward + (discount_factor * np.max(old_q_value)) - old_q_value
            
            # update q-value for the previous state and action pair
            new_q_value = old_q_value + (learning_rate * temp_difference)
            q_values[old_horz, old_vert, action] = new_q_value
            #print(f"New Q: {new_q_value}")

            #print(f"Q-Table: \n{q_values}\n")
            # debugging prints

        # progress prints
        print(f"Episode {episode + 1}/{num_episodes} complete!")
    print()




########################## AUXILIARY FUNCTIONS ####################################################




########################## TESTING ####################################################

def main():
    
    print("Begin testing:")
    #C.print_city(city)
    city = C.generate_city(9, 9)
    q_values = C.create_q_table(city)
    # Set start and end points NOTE: make sure they are not the outer nodes
    start_node = "I2,2"
    end_node = "I7,7"
    C.print_start_end(city, start_node, end_node)

    # Q-Learning hyperparameters
    num_episodes = 1000
    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 0.9

    print("prepare to start Q-Learning:")
    # save original q table
    og_q = copy.deepcopy(q_values)
    # Run Q-learning algorithm
    q_learning(city, start_node, end_node, num_episodes, learning_rate, discount_factor, epsilon, q_values)
    print("Finished running q_learning()\n")
    print(f"Original q table: \n{og_q}\n")
    print(f"Updated q table: \n{q_values}\n")
    
    visualize_path(q_values, city, start_node, end_node)
    print("stop here")
    

if __name__ == '__main__':
    main()
    print("Program terminated successfully")
