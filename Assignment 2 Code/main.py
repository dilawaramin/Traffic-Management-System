"""
Q Learning algorithm for traffic management system. Uses city.py to generate a city, and 
runs/trains algorithm from qlearn.py. 

Author:     Dilawar Amin
Date:       11/05/2023
"""
import qlearn as Q
import city as C 

def main():
    
    # Initialize city, starting point, and destination
    City = C.initialize_city()
    Start = C.get_start(City)
    Destination = C.get_destination(City, Start)
    
    # Initialize Q-Learning variables
    num_episodes, learning_rate, discount_factor, exploration_prob = Q.init_qlearn()
    
    # Show starting and end points on city map
    print("A pop up window will display your city layout, as well " +
          "as starting and ending points.\n")
    C.print_start_end(City, Start, Destination)
    
if __name__ == '__main__':
    main()
    print("Program Terminated")


        