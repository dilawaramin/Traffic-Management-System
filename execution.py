"""
Author:         Dilawar Amin
Date Created:   11/29/2023

Execution.py is where the execution of all algorithms is defined, so that they may 
conveniently be called from other files, such as main.py. Use this to define both
demonstration use cases and testing/research use cases (Ex. performance comparison
between different algorithms). 

NOTE: NetworkX, matplotlib.pyplot, and numpy are required for this program to run
Ensure they are installed by using: 

"pip install networkx"
"pip install matplotlib"
"pip install numpy"

Ensure they are installed and operational before running the program.
"""

########################## IMPORTS ####################################################

import qlearn as Q
import BFS as BFS
import Astar as A
import city as C 
import os
import time




########################## CITY CREATION ####################################################

def init_city_run():
    """
    This function calls all city initialization functions, and creates a city object to
    be passed to search algorithms 
        Parameters:
            None
        Returns:
            city: City graph object
            q_values: initialized q table
            SP: starting point on city object
            DP: destination point on city object 
    """
    # Initialize city, starting point, destination, and traffic
    city, q_values = C.initialize_city()
    SP = C.get_start(city)
    DP = C.get_destination(city, SP)
    C.initialize_traffic(city)
    print("A pop up window will display your city layout, as well " +
          "as starting and ending points as blue and red, respectively. " +
          "Traffic congestion is indicated with orange. To continue program, exit the window. \n")
    os.system('pause')
    C.print_start_end(city, SP, DP)
    return city, q_values, SP, DP



########################## ALGORITHM EXECUTION ####################################################

def run_astar(city, SP, DP):
    """
    This function calls all initialization functions, and then proceeds to run A* Search
        Parameters:
            None
        Returns:
            None 
    """
    # Run A* Search
    print("Beginning A* search!\n")
    A.astar_search(city, SP, DP)
    return


def run_bfs(city, SP, DP):
    """
    This function calls all initialization functions, and then proceeds to run DFS
        Parameters:
            None
        Returns:
            None 
    """    
    # Run DFS
    print("Beginning DFS search!\n")
    BFS.bfs_search(city, SP, DP)
    return


def run_q_learning(city, q_values, SP, DP):
    """
    This function calls all initialization functions, and then proceeds to run Q-Learning
        Parameters:
            None
        Returns:
            None 
    """    
    # Initialize Q-Learning variables
    ans = input("\nWould you like to set your own Q-Learning hyperparameters (Y/N)?\n")
    while ans.lower() != 'y' and ans.lower() != 'n':
        ans = input("Invalid Input. Do you wish to set your own parameters (Y/N)?")
    if ans == 'y':
        num_episodes, learning_rate, discount_factor, epsilon = Q.init_qlearn()
    else:
        num_episodes, learning_rate, discount_factor, epsilon = Q.init_qlearn_default()
    
    # Run Q-Learning training
    print("Beginning Q-Learning training!")
    os.system('pause')
    print()
    start = time.time()
    Q.q_learning(city, SP, DP, num_episodes, learning_rate, discount_factor, epsilon, q_values)
    end = time.time()
    tam = end - start
    print(f"Q-Learning training completed in {tam:.4} seconds.\n")
    
    # Show what the agent learned
    print("The agent has learned the following route (A separate window will open):")
    print("NOTE: you may need to scale up the window for larger city sizes\n")
    os.system('pause')
    print()
    # Q.visualize_path(q_values, city, SP, DP)
    Q.qlearn_timed(q_values, city, SP, DP)
    print()
    return




########################## HELPER FUNCTIONS ####################################################

def repeat(): 
    """
    Function to get input from user to repeat program
        Parameters:
            None
        Returns:
            Again: User input for running program again or not
    """
    choices = ['q', 'r']
    again = input("Would you like to run the program again (Q to quit, R to repeat)? ")
    while again.lower() not in choices:
        again = input("\nInvalid input. Quit or Repeat (Q/R)? ")
    return again.lower()


def repeat_city(city, q_values, SP, DP):
    """
    Function to get input from user to create a new city or not
        Parameters:
            None
        Returns:
            Again: User input for running program again or not
    """
    choice = input("Would you like to create a new city or reuse the current one? (N = New, R = Reuse): ")
    while choice.lower() not in ('r, n'):
        choice = input("\nInvalid input. New or Reuse (N/R)? ")
    if choice.lower() == 'r':
        print("You have selected to reuse the current city!\n")
        return city, q_values, SP, DP
    elif choice.lower() == 'n':
        print("You have chosen to create a new city!\n")
        city, q_values, SP, DP = init_city_run()
        return city, q_values, SP, DP