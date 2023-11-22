"""
Author:         Dilawar Amin
Date Created:   11/05/2023

Q Learning algorithm for traffic management system. Uses city.py to generate a city, and 
runs/trains algorithm from qlearn.py. 

NOTE: NetworkX, matplotlib.pyplot, and numpy are required for this program to run
Ensure they are installed by using: 

"pip install networkx"
"pip install matplotlib"
"pip install numpy"

Ensure they are installed and operational before running the program.
"""

########################## IMPORTS ####################################################

import qlearn as Q
import city as C 
import os
import time




########################## Program Function  ####################################################

def run_q_learning():
    """
    This function calls all initialization functions, and then proceeds to run Q-Learning
        Parameters:
            None
        Returns:
            None 
    """
    # Initialize city, starting point, destination, and traffic
    City, q_values = C.initialize_city()
    Start = C.get_start(City)
    Destination = C.get_destination(City, Start)
    C.initialize_traffic(City)
    
    # Initialize Q-Learning variables
    ans = input("Would you like to set your own Q-Learning hyperparameters (Y/N)?\n")
    while ans.lower() != 'y' and ans.lower() != 'n':
        ans = input("Invalid Input. Do you wish to set your own parameters (Y/N)?")
    if ans == 'y':
        num_episodes, learning_rate, discount_factor, epsilon = Q.init_qlearn()
    else:
        num_episodes, learning_rate, discount_factor, epsilon = Q.init_qlearn_default()
    
    # Show starting and end points on city map
    print("A pop up window will display your city layout, as well " +
          "as starting and ending points as blue and red, respectively. " +
          "Traffic congestion is indicated with orange. \n")
    os.system('pause')
    print()
    C.print_start_end(City, Start, Destination)
    
    # Run Q-Learning training
    print("Beginning Q-Learning training!")
    os.system('pause')
    print()
    start = time.time()
    Q.q_learning(City, Start, Destination, num_episodes, learning_rate, discount_factor, epsilon, q_values)
    end = time.time()
    tam = end - start
    print(f"Q-Learning training completed in {tam:.4} seconds.\n")
    
    # Show what the agent learned
    print("The agent has learned the following route (A separate window will open):")
    print("NOTE: you may need to scale up the window for larger city sizes\n")
    os.system('pause')
    Q.visualize_path(q_values, City, Start, Destination)
    print()
    return

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




########################## MAIN  ####################################################

def main():
    
    # Run the program initially 
    print("Welcome! Let's get started.\n")
    run_q_learning()
    
    # Ask to repeat or quit
    again = repeat()
    while again == 'r':
        run_q_learning()
        again = repeat()
    print("\nThank you for trying out our Q-Learning demonstration. Goodbye!\n")
    
    
if __name__ == '__main__':
    main()
    print("Program Terminated")


        