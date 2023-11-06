"""
Q Learning algorithm for traffic management system.
Use city objects from city.py

Author: Dilawar Amin
Date: 11/05/2023
"""
# Import city, to create and print cities
import city as C
import random


def visualize_path(Q, start_node, end_node):
    current_node = start_node
    while current_node != end_node:
        best_action = max(Q[current_node], key=Q[current_node].get)
        print(f"From {current_node} to {best_action}")
        current_node = best_action


def q_learning(city, start_node, end_node, num_episodes, learning_rate, discount_factor, exploration_prob):
    """
    Q Learning algorithm to route a vehicle from point A to B within the city
        Parameters:
            City: graph object representing city
            start_node: Starting point of vehicle
            end_node: Destination of vehicle
            num_episodes: Total # of times that q learning algorithm should 
                           repeat to train the A.I.
            learning_rate: Rate defining how aggressively we wish for agent to learn
            discount_rate: Factor by which to multiply future rewards (instant vs later reward)
            exploration_prob: Factor determining exploration v.s. exploitation
        Returns:
            None
    """
    # Initialize Q-table as a dictionary with default values
    Q = {node: {neighbor: 0 for neighbor in city.neighbors(node)} for node in city.nodes()}
    print(Q.keys()) # DEBUGGING
    for episode in range(num_episodes):
        current_node = start_node
        while current_node != end_node:
            if random.uniform(0, 1) < exploration_prob:
                next_node = random.choice(list(city.neighbors(current_node)))
            else:
                next_node = max(city.neighbors(current_node), key=lambda neighbor: Q[current_node][neighbor])
            reward = city[current_node][next_node].get('reward', 0)
            Q[current_node][next_node] = (1 - learning_rate) * Q[current_node][next_node] + \
                learning_rate * (reward + discount_factor * max(Q[next_node].values()))
            current_node = next_node
       # Print Q-values for each episode
        print(f"Q-values after episode {episode + 1}:")
        for node in Q:
            print(f"{node}: {Q[node]}")

        # Print progress updates
        print(f"Episode {episode + 1}/{num_episodes} completed.")


    # Test if the algorithm works
    #visualize_path(Q, start_node, end_node)

### TESTING ###

print("Begin testing:")
# generate city
city = C.generate_city(5, 10, 10)
#C.print_city(city)

# Set start and end points
start_node = "I0,0"
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


