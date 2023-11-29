# Q-Learning Traffic Management System

Author: Dilawar Amin
Date Created: 11/05/2023

## Description 

This repository contains a Python implementation of a Q-Learning algorithm for a traffic management system. The system uses a city represented as a graph, where intersections are nodes and streets are edges. 
The Q-Learning algorithm is applied to find optimal paths in the city, taking into account traffic and other factors.

## Dependencies 

Ensure the following Python libraries are installed:

- NetworkX
- Matplotlib
- NumPy

Install dependencies using the following commands:

```bash
pip install networkx
pip install matplotlib
pip install numpy

```

## Usage

To run the program, execute the main.py script. It will prompt you for various inputs, such as the dimensions of the city, starting and destination points, and Q-Learning hyperparameters. 
Follow the on-screen instructions for a successful run.

## Files

main.py:      The main script to run the Q-Learning algorithm for traffic management.
qlearn.py:    Contains the Q-Learning algorithm implementation.
city.py:      Functions for initializing, generating, and modifying the city object.
Other files:  Scatch file for testing purporses, and files containing unreleased code for future updates


## City Generation

The city is represented as a graph using NetworkX, with intersections as nodes and streets as edges. The city can be customized in terms of dimensions, traffic levels, and starting/destination points.

## Q-Learning Parameters

The user has the option to set Q-Learning hyperparameters, such as the number of episodes, learning rate, discount factor, and exploration rate (epsilon).

## Visualization

The program provides visualizations of the city layout, starting and ending points, and learned paths. The visualizations use Matplotlib for graphical representation.

## License

This project is licensed under the MIT License.
Feel free to explore and modify the code according to your needs. Contributions are welcome!

























