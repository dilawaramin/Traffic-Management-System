"""
Author:         Dilawar Amin
Date Created:   11/20/2023

Hard coded sample city to further test Q-Learning algorithm. Use to model real 
life cities or for performance testing purposes
"""

########################## IMPORTS ####################################################

import city as C 
import qlearn as Q
import time 




########################## CITY CREATION ####################################################

def create_city_1():
    """
    Function that creates a single city for testing.
    All parameters such as dimensions and traffic are hard coded. Includes 
    popular destinations.
        Parameters:
            None: All values are hard coded
        Returns:
            City: City object
    """
    city, init_q_learn = C.generate_city(15, 15)
    
    