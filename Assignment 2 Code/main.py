"""
Q Learning algorithm for traffic management system. Uses city.py to generate a city, and 
runs/trains algorithm from qlearn.py. 

NOTE: NetworkX, matplotlib.pyplot, and numpy are required for this program to run
Ensure they are installed by using "pip install networkx", "pip install matplotlib" and 
"pip install numpy" before running this program.

Author:     Dilawar Amin
Date:       11/05/2023
"""
import qlearn as Q
import city as C 
import os
import time




def main():
    
    # Initialize city, starting point, and destination
    City, q_values = C.initialize_city()
    Start = C.get_start(City)
    Destination = C.get_destination(City, Start)
    C.generate_traffic(City)
    
    # Initialize Q-Learning variables
    num_episodes, learning_rate, discount_factor, epsilon = Q.init_qlearn()
    
    # Show starting and end points on city map
    print("A pop up window will display your city layout, as well " +
          "as starting and ending points as blue and red, respectively. " +
          "Traffic congestion is indicatedwith orange. ")
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
    os.system('pause')
    Q.visualize_path(q_values, City, Start, Destination)
    print()
    
    
    
if __name__ == '__main__':
    main()
    print("Program Terminated")


        