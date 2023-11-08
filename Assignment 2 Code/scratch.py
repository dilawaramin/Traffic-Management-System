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
    #Q = {node: {neighbor: 0 for neighbor in city.neighbors(node)} for node in city.nodes()} # chatgpt
    # set destination
    C.set_destination(city, end_node)
    # Initialize visited states
    visited = set()
    # obtain list of reward
    rewards = C.get_rewards(city)
    print(rewards[end_node])
    C.print_city(city)
    # Run through the algorithm according to predined num_episodes variable
    for episode in range(num_episodes):
        
        # Hyperparameter decay rate
        if episode > (num_episodes / 2):
            epsilon = 0.9
            learning_rate = 0.8
            discount_factor = 0.7
        
        current_node = start_node
        curr_horz, curr_vert = C.current_xy(current_node)
        #print(C.is_terminal_state(city, "I0,1"))
        # initialize a list for explored paths
        while C.is_terminal_state(city, current_node) != True:
            # debugging print
            #print(f"Current Node: {current_node}.")

            # choose next action index
            action = get_next_action(q_values, curr_horz, curr_vert, epsilon)
            #print(f"Action: {actions[action]}")
            
            # store old node position, obtain new node position
            old_horz = curr_horz
            old_vert = curr_vert
            curr_horz, curr_vert = get_next_location(city, curr_horz, curr_vert, action)
            current_node = C.current_node(curr_horz, curr_vert)
            #print(f"New node: {current_node}")
            
            # get reward for action, calculate temporal difference
            if current_node in visited:     # check if current state has been visited
                reward = rewards[current_node] - C.REVISIT_PENALTY
            else:
                reward = rewards[current_node]
            old_q_value = q_values[old_horz, old_vert, action]
            ##### temp_difference = reward + (discount_factor * np.max(q_values[curr_horz, curr_vert])) - old_q_value
            temp_difference = reward + (discount_factor * np.max(old_q_value)) - old_q_value
            #print(f"Reward: {reward}")
            #print(f"Old Q : {old_q_value}")
            #print(f"Temp D: {temp_difference}")
            
            # update q-value for the previous state and action pair
            new_q_value = old_q_value + (learning_rate * temp_difference)
            ##### q_values[curr_horz, curr_vert, action] = new_q_value
            q_values[old_horz, old_vert, action] = new_q_value
            #print(f"New Q:{new_q_value}")
            #print(f"Is terminal:{C.is_terminal_state(city, current_node)}\n")
            #print(f"Q-Table: \n{q_values}\n")
            # add visited node to visited set
            visited.add(current_node)
            
            # debugging prints

            
        # progress prints
        print(f"Episode {episode + 1}/{num_episodes} complete!")
    print()