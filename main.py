"""
Author:         Dilawar Amin
Date Created:   11/05/2023

Q Learning algorithm for traffic management system. Uses city.py to generate a city, and 
runs/trains algorithm from qlearn.py. Also has a Depth-First Search (DFS) algorithm and 
an A* Heuristic search aglorithm, for testing and comparison purposes.

NOTE: NetworkX, matplotlib.pyplot, and numpy are required for this program to run
Ensure they are installed by using: 

    "pip install networkx"
    "pip install matplotlib"
    "pip install numpy"

Ensure they are installed and operational before running the program.
"""

########################## IMPORTS ####################################################

import execution as EXE
import os
import time




########################## Program Function  ####################################################

# Program functions




########################## MAIN  ####################################################

def main():
    
    # Run the program initially 
    print("Welcome! Let's get started.\n")
    print("Lets start by initializing your city!")
    
    # Create initial city
    city, q_values, SP, DP = EXE.init_city_run()
    
    # Run algorithm on the
    print("NOTE: There are known issues that will occasionally appear with the BFS algorithm.\n")   # NOTE: BFS disclaimer
    choice = input("Choose the algorithm you wish to use to find route. (Q for Q-Learning, B for BFS, A for A*): ").lower()
    while choice not in ('q', 'b', 'a'):
        choice = input("Invalid input. Choose 'Q' for Q-Learning, 'B' for BFS, or 'A' for A*: ").lower()
    # Run requested algorithm
    if choice == 'q':
        EXE.run_q_learning(city, q_values, SP, DP)
    elif choice == 'b':
        EXE.run_bfs(city, SP, DP)
    elif choice == 'a':
        EXE.run_astar(city, SP, DP)
    
    # Ask to repeat or quit
    again = EXE.repeat()
    # recreate city or reuse
    EXE.repeat_city(city, q_values, SP, DP)

    while again == 'r':
        choice = input("\nChoose the algorithm to solve the problem (Q for Q-Learning, B for BFS, A for A*): ").lower()
        if choice == 'q':
            EXE.run_q_learning(city, q_values, SP, DP)
            again = EXE.repeat()
            EXE.repeat_city(city, q_values, SP, DP)
        elif choice == 'b':
            EXE.run_bfs(city, SP, DP)
            again = EXE.repeat()
            EXE.repeat_city(city, q_values, SP, DP)
        elif choice == 'a':
            EXE.run_astar(city, SP, DP)
            again = EXE.repeat()
            EXE.repeat_city(city, q_values, SP, DP)

    while choice not in ('q', 'b', 'a'):
        choice = input("Invalid input. Choose 'Q' for Q-Learning, 'B' for BFS, or 'A' for A*: ").lower()


    print("\nThank you for trying out our Q-Learning demonstration. Goodbye!\n")
    
    
if __name__ == '__main__':
    main()
    print("Program Terminated")
