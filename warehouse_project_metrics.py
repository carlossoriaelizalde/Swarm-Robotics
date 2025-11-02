# -*- coding: utf-8 -*-
"""
Carlos Soria Elizalde

Operative Robots in Store

Nov 2nd 2025 Version - Subject to future changes

"""


# The purpose of this code is to create a store which works with robots; they fulfil clients' demands, which consist of combinations of products. 
# When a robot receives a client, it calculates the optimal order to visit the boxes that contain the required products, and goes for them.
    # It uses previous information to simplify calculations.
    # In the beggining, there's a panel which splits the store into two uncommunicated sections or areas. 
    # At a certain moment of the simulation, the panel vanishes, and the robots can go to the entire shop.
# When the robot has every item it is looking for, it returns to its stall.
# If two robots meet, they tell each other the optimal routes they know to get a command, the number of commands served and the total distance traveled.
# When robots finish attending a client, they may want to change the positions of certain boxes. For doing so, they use a genetic algorithm (GA).
    # If a robot has done an iteration of the GA, all robots stop finish their current command and remain in their stalls. 
    # When all robots are in their respective stalls, the change of the positions of boxes takes place (and they are aware of the new positions).

# First of all, I have to be able to create a logistics centre with hyperparameters to be chosen by the user: 
    
    # Number of different items.
    # Number of different robots / stations.
    # In this version, both must be even (because we are going to divide the center in two parts).
    
# Based on that, build a sufficiently large logistics centre with the following structure:

#   ----------------------------------------------------------------------------------------------------
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                           Warehouse                                                |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                      Imaginary Line                                                |
#  |----------------------------------------------------------------------------------------------------|
#  |                                                 |                                                  |                      
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  |                                       Customer Service Area                                        |
#  |                                                 |                                                  |
#  |                                                 |                                                  |
#  | Stall 1    Stall 2 ...                          |                                   Stall R        |
#   ----------------------------------------------------------------------------------------------------
#
#
#                                                Clients
#


import simpy
import numpy as np
import simpy.rt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import io
import random
from itertools import product, combinations, permutations 
from collections import defaultdict
import heapq
import copy
import sys
from datetime import datetime
from math import prod
import pandas as pd

from deap import algorithms, base, creator, tools


# There are three important classes in this problem: the Store class (which holds the other two classes) and defines the "game rules", 
# the robot class (that are part of the swarm) and the client class (just generates a queue of elements in order to keep the robots busy).

class Store(object):
    
    # Constructor: Given a number of boxes, a number of robots and a maximum of items in an command (and the simpy environment), we build the store center.
    def __init__(self, num_boxes, num_robots, max_items_order, length_weights, item_probs, env):
        
        self.num_boxes = num_boxes
        self.num_items = int(self.num_boxes / 2)
        self.num_robots = num_robots
        self.max_items_order = max_items_order
        self.env = env
        self.length_weights = length_weights
        self.item_probs = item_probs
        
        # Dictionary to hold the robots.
        self.dict_robots = None
        
        # Calculate minimum of squares in the matrix.
        
        # The positions of the boxes cannot touch each other horizontal, vertical or diagonally: the distance must be greater than sqrt(2).
        # How do I know if the storage area is large enough to hold all the boxes?
        #
        # A box takes up a 3x3 space if they cannot touch each other even diagonally.
        # So, if we have n boxes, we must have at least nx(3x3) squares of storage space = 9n squares.
        # Also, if we have R robots and stalls, if we want each robot (which occupies 1x1 of space) to have its own stall,
        # Then the length of the centre has to be max(R, int(sqrt(9n))+1).
        
        self.length_store = max(self.num_robots, int((9*self.num_boxes)**0.5) + 1)
        
        # In this problem, if length_store is odd, make it even.
        if self.length_store % 2 != 0:
            self.length_store = self.length_store + 1
        
        # And the height?
        
        # We will leave 3 spaces for customer service area (no boxes in the way) and one extra line for the customers.
        # So the height of the store is 3 + 1 + [int(sqrt(9n)) + 1] = 5 + int(sqrt(9n)).
        
        self.height_store = int((9*self.num_boxes)**0.5) + 5
        
        # Therefore, we initialize the matrix like this (the '100' represents the empty space).
        
        self.matrix = np.zeros((self.height_store, self.length_store)) + 100
        
        # We keep the number of the cell where we will place the screen and also its position.
        self.screen = self.length_store / 2
        
        self.center_stalls_complete = tuple([self.height_store - 2, int(self.length_store/2)])
        
        # Also for the metaheuristic, we need to know the centre of the stalls in the subproblem.
        self.center_stalls_west = tuple([self.height_store - 2, int(self.length_store/4)])
        self.center_stalls_east = tuple([self.height_store - 2, int(self.length_store*0.75)])
        
        # We initialize four dictionaries for keeping track of the positions of the robots, the boxes, the stalls and where the clients stand. 
        self.positions_robots = dict()
        self.positions_boxes = dict()
        self.positions_stalls = dict()
        self.positions_clients = dict()
        
        # Also, for the mataheuristic, we are going to create a dictionary with the structure key: box (from 0 to num_boxes - 1), value: position in store.
            # BE CAREFUL: the key: box is not the box itself, but the "shelf that contains it" (it won't ever change).
        self.dict_box_pos_complete = dict() 
        
        # Now, I must choose randomly the positions of the boxes and locate the stalls in the most balanced way.
        
        # Boxes: As there are initially two sections of the store, we must put half the boxes in the west side and the another half in the east side.
            
            # West: They must be in the submatrix(:-5, :self.screen) because we must preserve the last 4 lines for the customer service area and the customer zone,
            # and because they must be in the west side.
        
        max_row_box = self.height_store - 4
        
        boxes_placed = 0
        
        # We generate all the possible positions (initially they are all available).
        
        feasible_places = list(product(range(max_row_box), range(self.length_store)))
        
        # Until we have not placed every box in the west side:
        
        while boxes_placed < self.num_boxes/2:
            
            # In case the assignation is unfeasible.
            
            if boxes_placed < self.num_boxes/2 and len(feasible_places) == 0:
                
                print("Dimmension Problem in the west side! Try running the code again")
                sys.exit(0)  
            
            # We take a random position.
            
            pos = random.choice(feasible_places)
            
            # If it is in the west side.
            if pos[1] < self.screen:
                
                # We add in the dictionary of the positions and their respective boxes the added box.
             
                self.positions_boxes[tuple(pos)] = boxes_placed
                
                # For the metaheuristic.
                self.dict_box_pos_complete[boxes_placed] = tuple(pos) 
                
                # We add in the store (matrix) the number of the box in its respective cell.
                
                self.matrix[pos[0],pos[1]] = boxes_placed
                print("Box ",boxes_placed, ": ",[pos[0],pos[1]])
                
                # We must eliminate those positions that are horizontally/diagonally/vertically next to this position (taking into account boundaries).
                # For doing so, we use the own function that takes the neighbors of the cell that contains the new box (implemented later).
                
                neighbors = self.generate_neighbors(pos, max_row_box - 1, self.length_store - 1)
                
                # Update the list of feasible places for boxes.
                feasible_places = list([pair for pair in feasible_places if tuple(pair) not in neighbors])
                
                # Update the number of boxes_placed.
                boxes_placed = boxes_placed + 1
                
        # Now we do the same with the other half of the store.
        
        # East: They must be in the submatrix(:-5, self.screen:) because we must preserve the last 4 lines for the customer service area and the customer zone,
        # and because they must be in the east side.
        
        while boxes_placed < self.num_boxes:
            
            # In case the assignation is unfeasible.
            
            if boxes_placed < self.num_boxes and len(feasible_places) == 0:
                
                print("Dimmension Problem in the east side! Try running the code again")
                sys.exit(0) 
                
            # We take a random position.
            
            pos = random.choice(feasible_places)
            
            # If it is in the east side.
            if pos[1] >= self.screen:
                
                # We add in the dictionary of the positions and their respective boxes the added box.
                self.positions_boxes[tuple(pos)] = boxes_placed
                
                # For the metaheuristic. 
                self.dict_box_pos_complete[boxes_placed] = tuple(pos) 
                
                # We add in the store (matrix) the number of the box in its respective cell.
                self.matrix[pos[0],pos[1]] = boxes_placed
                print("Box ",boxes_placed, ": ",[pos[0],pos[1]])
                
                # We must eliminate those positions that are horizontally/diagonally/vertically next to this position (taking into account boundaries).
                # For doing so, we use the own function that takes the neighbors of the cell that contains the new box (implemented later).
                
                neighbors = self.generate_neighbors(pos, max_row_box - 1, self.length_store - 1)
                
                # Update the list of feasible places for boxes.
                feasible_places = list([pair for pair in feasible_places if tuple(pair) not in neighbors])
                
                # Update the number of boxes_placed.
                boxes_placed = boxes_placed + 1
        
            
        # Once here, the boxes have been successfully placed.
        
        # Now, for the metaheuristic, we create the west and east dictionary of boxes and positions in the store (it never changes).
        self.dict_box_pos_west  = {k: v for k, v in self.dict_box_pos_complete.items() if 0 <= k < (num_boxes/2)}
        self.dict_box_pos_east = {k: v for k, v in self.dict_box_pos_complete.items() if k >= (num_boxes/2)}
        
        # Now, we create matrices of distances between boxes and stall centers (complete, east, west; also for the metaheuristic), they never change.
        def manhattan_distance_df(dict_box_pos, center_stalls, center_label):
            """
            dict_box_pos: dict[int -> (r, c)].
            center_stalls: (r, c).
            center_label: label for the center in the index/columns.
            """
            # Stable order by key.
            box_keys = sorted(dict_box_pos.keys())
            
            # Points in the desired order: first boxes, at the end the center. 
            points = [dict_box_pos[k] for k in box_keys] + [center_stalls]
            labels = box_keys + [center_label]   # The boxes remain as int, the center as str. 
            
            coords = np.asarray(points, dtype=int)          # (N, 2).
            dist_mat = np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=2)  # (N, N).
            
            return pd.DataFrame(dist_mat, index=labels, columns=labels)

        self.name_center_stalls = "center_stalls"

        self.distances_matrix_complete = manhattan_distance_df(self.dict_box_pos_complete, self.center_stalls_complete, self.name_center_stalls)
        self.distances_matrix_west = manhattan_distance_df(self.dict_box_pos_west, self.center_stalls_west, self.name_center_stalls)
        self.distances_matrix_east = manhattan_distance_df(self.dict_box_pos_east, self.center_stalls_east, self.name_center_stalls)
        
        # Now, we create a dict where the key is the number of the item and the value a list with the two boxes (one per side) that have the item.
            # For example, if there are 8 boxes (4 articles/items), boxes 0 and 4 will have item 0 -> 0: [0, 4], boxes 1 and 5 the item 1 -> 1: [1, 5], ...
        self.dict_items_boxes_complete = dict()
        for i in range(int(self.num_boxes/2)):
            self.dict_items_boxes_complete[i] = [i, int((self.num_boxes/2)+i)]
            
        # Also, for the metaheuristic, we create two dictionaries of the split sections of the store.
        self.dict_items_boxes_west = dict()
        self.dict_items_boxes_east = dict()

        for item in range(self.num_items):
            
            self.dict_items_boxes_west[item] = tuple([item])
            self.dict_items_boxes_east[item] = tuple([item + self.num_items])
        
        # We create a dict where the key is the number of the box and the value its associated item.
            # If there are 8 boxes (4 articles/items), box 0 : item 0, box 1: item 1, ... , box 4 : item 0, ...
        self.box_item = dict()
        for i in range(int(self.num_boxes/2)):
            self.box_item[i] = i
            self.box_item[i + int(self.num_boxes/2)] = i
            
        
        # The next step is to distribute the stalls where the robots attend clientes; for doing so, we have created a function that does so (implemented later).
        stalls_positions = self.distribute_stalls(self.num_robots, self.length_store)
        
        # Now, we add the stalls to the dictionaries (we can not assign yet the robot to its stall, for that we need to initialize the robots).
        for i,stall in enumerate(stalls_positions):
            
            self.positions_stalls[i] = np.array([self.height_store - 2, stall]) 
            
            self.positions_clients[i] = np.array([self.height_store - 1, stall])
        
        # The last important thing to consider is the articles and their chances of being asked (alone or in groups).
        # For doing so, we use a function that is implemented later.
        self.prob_itemsets = self.design_article_dict()
        
        # Now, we create several objects related to the execution of the simulation.
        
        # Create a waiting line for the clients.
        self.waiting_line = simpy.Store(env)
        
        # Create a list of traffic lights for the stalls, so that the client remains in the stall until it is served.
        self.traffic_lights_stalls = [simpy.Resource(self.env, capacity=1) for _ in range(self.num_robots)]
        
        # Dictionary of robots and the items they have still to go for in a certain moment of the simulation.
        self.robot_items_not_visited = dict()
        
        # Create a list of graphics to show it later.
        self.list_graphics= []
        
        # Boolean to stop the simulation.
        self.stop_simulation = False
        
        # Define dictionary of colors so that a key is a "general" color, and its value is a list with colors ordered so that 
        # the first colors are really different between each other.
        self.palette = {
                        "red": ["#DC143C", "#800000", "#FF2400", "#FF007F", "#CB4154"],
                        "green": ["#32CD32", "#228B22", "#808000", "#98FF98", "#50C878"],
                        "blue": ["#4169E1", "#87CEEB", "#008080", "#0047AB"],
                        "purple": ["#8A2BE2", "#E6E6FA", "#DDA0DD", "#9966CC", "#DA70D6"],
                        "brown": ["#D2691E", "#D2B48C", "#A0522D", "#635147", "#F5F5DC"],
                        "orange": ["#FFA500", "#CC5500", "#FFDAB9", "#FBCEB1", "#B7410E"],
                    }
        
        # We create a dictionary of colors and codes.
        self.create_colors_dict()
        
        # And the last thing we are going to consider is how to control the robots so that:
                # Only one robot at a time does the GA.
                # It only exectutes the algorithm when all the other robots of the section it considers are at their respective stalls .
                    # It may happen that a robot is waiting for the robots of its own section (west/east) but meanwhile it changes to global (then all robots are considered).
                # The other robots wait until the end of the GA.
                
        # So that only one robot does the GA.
        self.ga_lock = simpy.Resource(env, capacity=1)
        # Is there a GA already being executed?
        self.ga_in_progress = False             
        # In order to differentiate between "complete", "east" and "west".
        self.ga_scope = None
        # IDs of the robots that are affected by the GA.              
        self.ga_participants = set()
        # Robots that are ready in the stalls.           
        self.ga_ready_events = {}               
        # To let other robots know that the GA has finished.
        self.ga_done_event = None               
        # ID of the robot that executes the GA.
        self.ga_requester = None  
        # To indicate if there has been "recently" an execution of the GA.
        self.ga_recent_creation = False

        # For the analysis of the improvement obtained with the GA.
        self.metrics = []                  
        self.ga_version = 0               
        self.current_ga_scope = None          
        
        # We split robots by section.
        def robots_in_section(scope):
            n = self.num_robots
            half = n // 2
            if scope == "west":
                return set(range(0, half))
            if scope == "east":
                return set(range(half, n))
            if scope == "complete":
                return set(range(n))
            raise ValueError("scope unknown")
        
        # We are going to keep it as simple method.
        self.robots_in_section = robots_in_section  
        
    # Funcion that creates the dictionary which associates colors with numbers of the matrix (objects of the store).
    def create_colors_dict(self):
        
        # 100 - "white" - Nothing.
        # 101 - "black" - Robot.
        # 102 - "grey" - Stall.
        # 103 - "pink" - Client.
        # Other numbers and colors are for the different boxes.
        self.colors = {100: 'white', 101: 'black', 102: 'grey', 103: "#FF00FF"}
        
        # Depending on the number of articles, we select different colors.
        # For doing so, we call a function that adds these colors (implemented just below).
        self.get_k_elements_by_order()
        
    # Function that gets different colors for the boxes (trying to have the most diversed colors possible). 
    def get_k_elements_by_order(self):
        max_length = max(len(variations) for variations in self.palette.values())
        count = 0
        finish = False
        
        for i in range(max_length):
            for color in self.palette:
                if i < len(self.palette[color]):
                    self.colors[count] = self.palette[color][i]
                    count += 1
                    if count == int(self.num_boxes/2):
                        finish = True
                        break
            if finish: 
                break
        
    # Function that initializes a robot and its related objects (adding its related information to the related dictionaries).   
    def initialize_robot(self, num):
        
        pos = self.positions_stalls[num]
            
        self.positions_robots[num] = pos 
        
        self.robot_items_not_visited[num] = []
        
        traffic_light_stall = self.traffic_lights_stalls[num]
        
        return pos, traffic_light_stall
    
    
    # Function to work out the neighbors of a selected position horizontal/vertical/diagonally (1 = distance).
    def generate_neighbors(self, pos, x_max, y_max):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                new_x = pos[0] + dx
                new_y = pos[1] + dy
                if 0 <= new_x <= x_max and 0 <= new_y <= y_max:
                    neighbors.append((new_x, new_y))
        return neighbors
    
    
    # Function to distribute the stalls in the most balanced way.
    def distribute_stalls(self, num_robots, length_store):
        
        # Step size (cells per robot) as a float.
        step_size = (length_store - 1) / (num_robots - 1)
    
        # Generate positions with rounding (to ensure position of robot 0, 1, ... r-1).
        positions = [round(i * step_size) for i in range(num_robots)]
    
        # Ensure uniqueness and within range (that a cell is not repeated for two robots).
        positions = sorted(set(min(p, length_store - 1) for p in positions))
    
        # Adjust if there are fewer positions than robots.
        while len(positions) < num_robots:
            for i in range(length_store):
                if i not in positions:
                    positions.append(i)
                    if len(positions) == num_robots:
                        break
    
        return sorted(positions)
 
    
    # Function to create the probabilities of each possible command.
    def design_article_dict(self):
        """
        Build probabilities for all itemsets of sizes 1..max_articles.

        Steps:
          1) Item probabilities P(i): if provided, normalize; else sample U(0,1) and normalize (sum_i P(i)=1).
          2) Base itemset probability: P(S) = ∏_{i in S} P(i).
          3) Absolute length targets:
             - Normalize user weights so they sum to 1 over GENERATED lengths.
             - For every generated length r:
                 * If r in weights_norm: scale all itemsets of length r so they sum EXACTLY to weights_norm[r].
                 * Else: set probabilities of that length to 0 (i.e., unlisted lengths get zero mass).
             - This guarantees sum over ALL itemsets = 1, and per-length sums match the targets.

        Returns:
          - final_probs: {itemset: probability}
          - diagnostics: dict with useful internals for checks/prints
        """
        # Item probabilities.
        n_items = int(self.num_boxes / 2)
        max_articles = min(self.max_items_order, n_items)

        # Prepare item probabilities.
        if self.item_probs is None:
            raw = {i: random.uniform(0, 1) for i in range(n_items)}
        else:
            raw = {int(i): float(v) for i, v in self.item_probs.items()}
            for i in range(n_items):
                if i not in raw:
                    raw[i] = 0.0

        s = sum(raw.values())
        if s <= 0:
            P = {i: 1.0 / n_items for i in range(n_items)}
        else:
            P = {i: v / s for i, v in raw.items()}

        # Now, generate itemsets and base probabilities as products ∏ P(i).
        all_sets = []
        for r in range(1, max_articles + 1):
            all_sets.extend(combinations(range(n_items), r))

        base_set_probs = {itemset: prod(P[i] for i in itemset) for itemset in all_sets}

        # Last, rescale considering prob of command of a certain length.
        # Sums by length BEFORE scaling.
        base_sums_by_len = defaultdict(float)
        for S, p in base_set_probs.items():
            base_sums_by_len[len(S)] += p

        generated_lengths = sorted(base_sums_by_len.keys())

        # Consider only generated lengths when normalizing user weights.
            # (keys outside generated lengths are ignored).
        usable_lengths = [r for r in self.length_weights if r in generated_lengths]

        # Normalize user-provided weights over usable lengths (e.g., 1,2,2 -> 0.2,0.4,0.4).
        weights = {int(r): max(0.0, float(self.length_weights[r])) for r in usable_lengths}
        w_sum = sum(weights.values())
        if w_sum <= 0 and usable_lengths:
            weights_norm = {r: 1.0 / len(usable_lengths) for r in usable_lengths}
        else:
            weights_norm = {r: (weights[r] / w_sum) for r in usable_lengths} if usable_lengths else {}

        # Build final probabilities with absolute targets:
            # - For r in weights_norm: scale to sum exactly weights_norm[r].
            # - For r not in weights_norm (but generated): set to 0.
        final_probs = {}

        # Counts per length and current sums (for scaling).
        counts_by_len = defaultdict(int)
        for S in base_set_probs:
            counts_by_len[len(S)] += 1

        # First pass: compute scaled values per length.
        for r in generated_lengths:
            target_sum = weights_norm.get(r, 0.0)  # 0 if not specified.
            current_sum = base_sums_by_len[r]
            if target_sum == 0.0:
                # zero out this length.
                for S, p in base_set_probs.items():
                    if len(S) == r:
                        final_probs[S] = 0.0
            else:
                if current_sum > 0:
                    scale = target_sum / current_sum
                    for S, p in base_set_probs.items():
                        if len(S) == r:
                            final_probs[S] = p * scale
                else:
                    # Extremely unlikely with positive products; still, distribute uniformly.
                    c = counts_by_len[r]
                    uniform_val = target_sum / max(c, 1)
                    for S, _ in base_set_probs.items():
                        if len(S) == r:
                            final_probs[S] = uniform_val

        return final_probs


# Second class: Robot Class.
class Robot(object):
    
    def __init__(self, env, num, store, traffic_light_move, time_merge, cxpb, mutpb, ngen, num_individuals, module_for_genetic):
        
        self.env = env
        self.num = num
        self.store = store
        self.traffic_light_move = traffic_light_move
        self.time_merge = time_merge
        
        # We call to the function of the Store class "initialize_robot" in order to get its initial position and the traffic light that prevents it 
        # from moving if there is no client.
        self.pos, self.traffic_light_stall = self.store.initialize_robot(self.num)
        
        # The position in which the client it serves stands is adding 1 to the row of the pos.
        self.pos_its_client = np.array([self.pos[0]+1, self.pos[1]])
        self.pos_its_stall = self.pos
        #print("Robot ",num,", stall position: ",self.pos, ", client pos: ",self.pos_its_client)
        
        # It must keep some information of the things the different robots are doing:
            # To keep orders a robot is doing and how many times.
            # Key: Command, Value: List where pos j means how many times robot j has completed the command.
        self.dict_orders_freq = {key : [0]*self.store.num_robots for key in self.store.prob_itemsets}
        
            # To keep orders and number of movements for complying with.
            # Key: Command, Value: List where pos j means the sum of the steps for robot j to complete the command.
        self.dict_orders_movements = {key : [0]*self.store.num_robots for key in self.store.prob_itemsets}
        
        # To know which other robots have the latest version of the information of a particular one:
            # 0 -> Not Updated.
            # 1 -> Updated.
            # -1 -> Itself.
        self.updated_info = [1] * self.store.num_robots
        self.updated_info[self.num] = -1
        
        # For helping us determining the frequency of executing a step of the genetic algorithm, we will use the sigmoid function:
            # The more it approaches to this value or any multiple, more chances of executing the next step.
            # If it has recently worked out rules, until it doesn't surpass the following multiple of module, it can't execute again the GA.
        self.module_for_genetic = module_for_genetic
        self.total_commands_served = 0
        
        # Now, we assign the parameters received for the GA to its objects of the same store.
            # Probability that two parents mate.
        self.cxpb = cxpb

            # Probability that a new individual mutates.
        self.mutpb = mutpb

            # Number of generations.
        self.ngen = ngen
        
            # Number of individuals.
        self.num_individuals = num_individuals
        
        
        # Dictionary that contains the optimal order for a certain command.
            # Key: Command, Value: Optimal order of BOXES, not items. 
            # This will be used for avoiding using Dijkstra in certain situations.
            # Ex: If we know that the optimal order for command (0,1,2) is (2,1,0), and we have the command (0,1,2,3,4) (no more items exist):
                # Suppose that the screen still exists and that we are in the left side.
                # The number of possible combinations are 5! = 120.
                # But considering this information, the number of possible combinations reduces drastically.
                    # We must put two DIFFERENT elements in five possible positions: 2 times (5 over 2) = 2 x 10 = 20.
            # Ex: If we know that the optimal order for command (0,1,2) is (2,1,0), and we have the command (0,1,2,3,4) (no more items exist):
                # Suppose that the screen doesn't exist anymore.
                # Now possibilities open, because we must consider these commands and two different scenarios:
                    # It is the first time that we consider this command after the merging, then we must consider:
                        # all mentioned before (20).
                        # all combinations of 0,1,2 (they must appear in that order) and 8 and 9 (20 as before).
                        # all combinations of 0,1,2 (they must appear in that order) and 3 and 9 (20 as before).
                        # all combinations of 0,1,2 (they must appear in that order) and 8 and 4 (20 as before).
                        # all combinations of 5,6,7,8,9 (120).
                        # all combinations of 5,6,7,3,4 (120).
                        # all combinations of 5,6,7,3,9 (120).
                        # all combinations of 5,6,7,8,4 (120).
                    
                    # Total = 560
                    
                    # If it is not the first time after the merging, we do not consider 5, 6, 7, then the combinations are: 
                        # all mentioned before (20)
                        # all combinations of 0,1,2 (they must appear in that order) and 8 and 9 (20 as before).
                        # all combinations of 0,1,2 (they must appear in that order) and 3 and 9 (20 as before).
                        # all combinations of 0,1,2 (they must appear in that order) and 8 and 4 (20 as before).
                    
                    # Total = 80
                    
                # It may happen that the route we get to attend items 0,1,2,3,4 is not optimal, but it will be a good route obtained at a low computational cost.
                # There will be other robots that may work out an optimal route for this command and share this information when they communicate between each other.
                
                # However, if we tried all possible combinations, we would have more combinations:
                    # Imagine we consider combinations of 0,1,2,3,4: 5! = 120.
                    # We can change boxes of 0,1,2,3,4 by their respective mates: 5,6,7,8,9. We can make from 0 to 5 changes.
                        # If we change nothing, we have 1 combination (itself) = (5 over 0).
                        # If we change 1 item, we can change the first one (0 -> 5), the second one (1 -> 6) ... Total: (5 over 1) changes = 5.
                        # If we change 2 items, we have (5 over 2) changes = 10.
                        # ...
                    # Therefore, the total number of combinations is 5! [(5 over 0) + (5 over 1) + ... (5 over 5)] = 120 * (1 + 5 + 10 + 10 + 5 + 1) = 120 * 32 = 3840.
                    
        self.optimal_order_boxes = dict()
        
        # Create also a list with those commands whose optimal routes have been calculated before the merging.
        # This helps us to determine whether an optimal route (associated with the command of the list)
        # has been optimized in the "after merge" phase or not.
            # In the beginning, before merging, all commands whose optimal route has been obtained are added.
            # After merging, the robot sees if the command is in here:
                # If it is, we must take into account possible boxes in the other half of the store.
                # If it is not, or its route hasn't been calculated, or it has been updated with the boxes of the other half.
        
        self.list_routes_before_merging = []
        
        # Last, we are going to create a dictionary like key: command, value: least number of movements achieved.
            # This will help us to keep the best orders of visiting boxes when two robots interact with each other.
        self.command_min_moves = dict()
   

    # Function to draw the graphic of the situation in a certain moment.
    def graphic(self):
        
        # Adjust size of figure.
        fig = plt.figure(figsize=(21, 14))  
        
        # Create a grid of 2x2.
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])  
        
        # The graphic is the upper half.
        ax = fig.add_subplot(gs[0, :])  
        
        # The robots-boxes info is the lower-left quarter.
        info_ax = fig.add_subplot(gs[1, 0])  
        
        # Legend in lower right quarter.
        legend_ax = fig.add_subplot(gs[1, 1])  
    
        # Hide axis in the lower subplots.
        info_ax.axis('off')
        legend_ax.axis('off')
        
    
        # Draw store situation. For each position: 
        for x in range(self.store.height_store):
            for y in range(self.store.length_store):
                
                # We obtain the value of the cell.
                value = self.store.matrix[x, y]
    
                # There is a special case: when a robot is on a stall. In that case, we print the robot with the label on the cell.
                if value == 102 and any(np.array_equal(value, np.array([x, y])) for value in self.store.positions_robots.values()):
                    ax.add_patch(plt.Rectangle((y, x), 1, 1, color=self.store.colors[101]))
                    
                    # Get number of robot and put label.
                    robot_num = next((k for k, v in self.store.positions_robots.items() if np.array_equal(v, np.array([x, y]))), None)
                    ax.text(y + 0.5, x + 0.5, str(int(robot_num)), color='white', ha='center', va='center', fontsize=12)
                
                # Else if it is a box, take into account the item it has.
                elif value < 100:
                    ax.add_patch(plt.Rectangle((y, x), 1, 1, color=self.store.colors[int(value%(self.store.num_boxes/2))]))
                    # If it's box, we put its label on the cell.
                    box_text = "Box: "+str(int(value))+", Item: "+str(int(value%(self.store.num_boxes/2)))
                    ax.text(y + 0.5, x + 0.5, box_text, color='black', ha='center', va='center', fontsize=12)
                    
                # Else, print the color represented by the value.   
                else:
                    ax.add_patch(plt.Rectangle((y, x), 1, 1, color=self.store.colors[value]))
    
                # If it's a robot, guess which robot it is and put its label on the cell.
                if value == 101:
                    robot_num = next((k for k, v in self.store.positions_robots.items() if np.array_equal(v, np.array([x, y]))), None)
                    ax.text(y + 0.5, x + 0.5, str(int(robot_num)), color='white', ha='center', va='center', fontsize=12)
    
        # Set limits and grid. 
        ax.set_xlim(0, self.store.length_store)
        ax.set_ylim(self.store.height_store, 0)
        
        # Horizontal Separation Line.
        ax.axhline(self.store.height_store - 1, color='black', linewidth=2)  
        
        # Vertical Separation Line (if screen has not vanished yet).
        if env.now < time_merge: 
            ax.axvline(self.store.screen, color='black', linewidth=2)  
            
        ax.set_xticks(range(self.store.length_store))
        ax.set_yticks(range(self.store.height_store))
        ax.grid()
        ax.set_title("t = " + str(round(self.env.now, 4)))
    
        # Create information of robots and boxes to visit. 
        robot_info = "Robots and Target Items:\n"
        for robot_id, boxes in self.store.robot_items_not_visited.items():
            if len(boxes) == 0:
                robot_info += "Robot "+str(robot_id)+": {}\n"
            else:
                robot_info += f"Robot {robot_id}: {boxes}\n"
    
        # Draw the information in the downleft subplot.
        info_ax.text(0.02, 0.5, robot_info, fontsize=18, va='center', ha='left',
                     bbox=dict(boxstyle="round", facecolor="lightgrey", alpha=0.5))
    
        # Create the legend and put it in the downright subplot.
        legend_patches = [
            mpatches.Patch(color='white', label='Empty Space'),
            mpatches.Patch(color='black', label='Robot'),
            mpatches.Patch(color='grey', label='Stall'),
            mpatches.Patch(color="#FF00FF", label='Client')
        ]
        legend_ax.legend(handles=legend_patches, loc='center', fontsize=18)
    
        # Save plot in list of plots.
        self.store.list_graphics.append(fig)
    
        plt.close(fig)


    # Function to calculate the shortest path between two places using Dijkstra.
        # Excluding blocked cells (boxes, robots, shelves).
    def calculate_shortest_path(self, start, goal, box, exclude=None):
        
        INF = 10**9

        start = tuple(start); goal = tuple(goal)
        # normalize exclude to set consisting on tuples.
        exclude_set = set()
        if exclude:
            for e in exclude:
                # e can come as a tuple or as an np.array.
                try:
                    exclude_set.add(tuple(e))
                except TypeError:
                    exclude_set.add(tuple(e.tolist()))
    
        def in_bounds(nx, ny):
            H = self.store.height_store
            W = self.store.length_store
            if self.env.now < self.time_merge:
                # Before merge: split by section.
                if self.num < (self.store.num_robots / 2):        # west.
                    return 0 <= nx < H and 0 <= ny < self.store.screen
                else:                                              # east.
                    return 0 <= nx < H and self.store.screen <= ny < W
            else:
                return 0 <= nx < H and 0 <= ny < W
    
        def is_walkable(nx, ny):
            # 100 empty, 101 robot, 102 stalls (become blocked except yours).
            cell = self.store.matrix[nx, ny]
            # Allow step your own stall even if it is 102.
            if (nx, ny) == tuple(self.pos_its_stall):
                return True
            # Allow target box if it is there (when box is int).
            if box is not None and cell == box:
                return True
            # In general, only 100 (empty) and 101 (robot) can be gone through.
            return cell in (100, 101)
    
        def get_neighbors(pos):
            x, y = pos
            for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                nx, ny = x+dx, y+dy
                if not in_bounds(nx, ny): 
                    continue
                if (nx, ny) in exclude_set:
                    continue
                if is_walkable(nx, ny):
                    yield (nx, ny)
    
        # Trivial case.
        if start == goal:
            return []  # You are already there.
    
        # Dijkstra (unit cost).
        queue = [(0, start)]
        distances = {start: 0}
        previous = {start: None}
        # Hard limit to avoid deadly loops.
        H = self.store.height_store
        W = self.store.length_store
        max_expansions = H * W + 1000
        expansions = 0
    
        while queue:
            dist, u = heapq.heappop(queue)
    
            # Discard obsolete inputs from heap.
            if dist != distances.get(u, INF):
                continue
    
            if u == goal:
                break
    
            expansions += 1
            if expansions > max_expansions:
                # Security -> Without reasonable way here. 
                return []
    
            for v in get_neighbors(u):
                nd = dist + 1
                if nd < distances.get(v, INF):
                    distances[v] = nd
                    previous[v] = u
                    heapq.heappush(queue, (nd, v))
    
        # Re-build.
        if goal not in previous:
            return []  # No way right now. 
    
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = previous[node]
        path.reverse()
    
        # If we are looking for a box, not considering origin and box cell; if it is headed for the stall, just exclude origin.
        return (path[1:-1] if box is not None else path[1:])
    
    # Function that calculates the total distance for a given order of visiting boxes.
    def calculate_total_distance(self, order):
        
        total_distance = 0
        current_pos = tuple(self.pos)
        
        # For each box, calculate the distance between the current box/pos and the following one (using Dijkstra).
        for box in order:
            box_pos = next((key for key, value in self.store.positions_boxes.items() if value == box), None)
            
            if box_pos is None:
                return float('inf')
            
            #print(box, box_pos, type(box_pos))
            path = self.calculate_shortest_path(current_pos, box_pos, box)
            
            if current_pos != box_pos and len(path) == 0:
                return float('inf')
            
            total_distance += len(path) # -1
            current_pos = box_pos
            
        # Work out distance from last visited box to initial position (=stall).
        path_back = self.calculate_shortest_path(current_pos, tuple(self.pos_its_stall), None)
        
        if current_pos != tuple(self.pos_its_stall) and len(path_back) == 0:
            return float('inf')
        
        total_distance += len(path_back) #- 1
        #print("Total_distance: ", total_distance)
        return total_distance



    # Function that finds the optimal order of visiting boxes to minimize total distance.
    def find_optimal_order(self, possible_client_requests):
        
        # assert: all elements from "order" must be boxes -> exist in positions_boxes as value.
        def _is_box_id(x): 
            return any(value == x for value in self.store.positions_boxes.values())
        
        best_order = None
        min_distance = float('inf')
        # For each possible order, calculate total distance.
        for order in possible_client_requests:
            # dentro del bucle for order in possible_client_requests:
            assert all(_is_box_id(b) for b in order), f"find_optimal_order received items intead of boxes: {order}"
            
            print("Possible order: ",order, type(order))
            distance = self.calculate_total_distance(order)
            if distance < min_distance:
                min_distance = distance
                best_order = order
        # We only keep the order with the lowest distance traveled.
        print("best order: ", best_order)
        return list(best_order)
    
    
    
    # Funciton that searches the optimal_order_boxes dictionary for a key that contains ALL elements of input_tuple (it may have more that are useless for our input_tuple).
        # We iterate in reverse order (because if we start at the beginning, it may happend that a better route has been found later), 
        # and return the optimal route based on its associated value.
    def find_partial_route(self, input_tuple):
        
        # Iterate through the dictionary in reverse order (because they are newest).
        for key in reversed(self.optimal_order_boxes.keys()):
            
            # Check if all elements of input_tuple are in the key.
            if all(elem in key for elem in input_tuple):
                
                # Get the value tuple associated with the found key.
                value_tuple = self.optimal_order_boxes[key]
                
                # Filter the elements of value_tuple to include only those in input_tuple.
                ordered_route = tuple(elem for elem in value_tuple if (elem % int(self.store.num_boxes/2)) in input_tuple)
                
                return ordered_route
          
            
        # If no match is found.
        print("No partial_route found")
        return None

    # Function that finds key in dict of optimal_order_boxes so that it contains the highest number of elements possible, return order of related boxes.    
    def find_biggest_subset(self, new_command):
        
        # We consider first subsets of high length because fewer combinations of optimal routes for a command will have to be calculated.
        for length in range(len(new_command), 1, -1):
            guide = next(
                (subset for subset in combinations(new_command, length) if subset in self.optimal_order_boxes),
                None
            )
            if guide:
                return self.optimal_order_boxes[guide]
        
        return None
    
    # Function that creates intelligent permutations to try all possible optimal routes for a certain command.
        # Intelligent because the elements of base must be in the same order in complete; therefore, less combinations are considered.
    def generate_intelligent_combinations(self, complete, base):

        # Generate all possible permutations. 
        all_permutations = permutations(complete)

        combinations = []
        
        # Map item -> box fixed by the base.
        base_item_to_box = { self.store.box_item[b]: b for b in base }  
    
        # Item sequence whose order must be preserved (derived from base).
        base_items_order = [ self.store.box_item[b] for b in base ]

        for comb in all_permutations:
            # Check if 'base_items_order' order is respected.
            ok = True
            for i in range(1, len(base_items_order)):
                if comb.index(base_items_order[i-1]) > comb.index(base_items_order[i]):
                    ok = False
                    break
            if not ok:
                continue
    
            # Go from Items to Boxes.
            boxes_seq = []
            for it in comb:
                if it in base_item_to_box:
                    boxes_seq.append(base_item_to_box[it])
                else:
                    # Choose box considering phase or section.
                    if self.env.now < self.time_merge:
                        # Before merge, each robot can only operate in its own side. 
                        if self.num < (self.store.num_robots / 2):   # west.
                            boxes_seq.append(it)
                        else:                                        # east.
                            boxes_seq.append(it + int(self.store.num_boxes/2))
                    else:
                        # After merge, choose the nearest to the robot to keep "intelligent" combinations.
                        west_box = it
                        east_box = it + int(self.store.num_boxes/2)
                        pos_w = next(k for k,v in self.store.positions_boxes.items() if v == west_box)
                        pos_e = next(k for k,v in self.store.positions_boxes.items() if v == east_box)
                        d_w = abs(pos_w[0]-self.pos[0]) + abs(pos_w[1]-self.pos[1])
                        d_e = abs(pos_e[0]-self.pos[0]) + abs(pos_e[1]-self.pos[1])
                        boxes_seq.append(west_box if d_w <= d_e else east_box)
    
            combinations.append(tuple(boxes_seq))
    
        return combinations
    
    
    # Function that makes robots share information of what they know (optimal_routes, optimal orders and number of movements for fulfilling a command)
    # with robots that are next to it.
    def check_neighbors_update_info(self):
        
        # Function that makes two robots update the information they have.
        def share_info(robot_num):
            
            # Get number of the other robot.
            other_robot = self.store.dict_robots[robot_num]
            """
            print(self.num,"=self.dict_orders_freq before merging", self.dict_orders_freq)
            print(robot_num,"=other_robot.dict_orders_freq before merging", other_robot.dict_orders_freq)
            print(self.num,"=self.dict_orders_movements before merging", self.dict_orders_movements)
            print(robot_num,"=other_robot.dict_orders_movements before merging", other_robot.dict_orders_movements)
            """
            
            # For each command:
            for command in self.dict_orders_freq:
                
                # We will have two lists of frequencies like:
                    # Self: [freq_robot0, ..., freq_robot_r-1].
                    # Self: [freq_robot0', ..., freq_robot_r-1'].
                
                # And two lists of average necessary movements like:
                    # Self: [moves_robot0, ..., moves_robot_r-1].
                    # Self: [moves_robot0', ..., moves_robot_r-1'].
                
                # First we update the frequencies of the commands and the number of necessary movements.
                
                # Convert lists to NumPy arrays.
                freq_self = np.array(self.dict_orders_freq[command])
                freq_other = np.array(other_robot.dict_orders_freq[command])
                
                # Calculate maximum of each pair of elements.
                    # [max(freq_robot0, freq_robot0'), ..., max(freq_robot_r-1, freq_robot_r-1')].
                updated_array_freq = np.maximum(freq_self, freq_other)
                
                # Create a mask to determine which elements were taken from self and which from the other robot.
                mask = freq_self >= freq_other
                
                # Select movements based on the mask.
                updated_array_move = np.where(mask, np.array(self.dict_orders_movements[command]), np.array(other_robot.dict_orders_movements[command]))
                
                # Update info of movements for commands.
                self.dict_orders_movements[command] = updated_array_move.tolist()
                other_robot.dict_orders_movements[command] = updated_array_move.tolist()
                
                # Update info of number of commands.
                self.dict_orders_freq[command] = updated_array_freq.tolist()
                other_robot.dict_orders_freq[command] = updated_array_freq.tolist()
                
                # Now update if necessary the optimal order.
                # If both robots have info about an optimal route for a certain command, they stay with the lowest one.
                # But if not, they update with the other robot's route.
                # In case none of them has an optimal route for a certain command, nothing happens.
                """
                print("-----------------------------------------------------------------------------------------------")
                print("command: ", command)
                print("self.optimal_order_boxes: ", self.optimal_order_boxes)
                print("other_robot.optimal_order_boxes: ", other_robot.optimal_order_boxes)
                print("self.command_min_moves: ", self.command_min_moves)
                print("other_robot.command_min_moves: ", other_robot.command_min_moves)
                print("-----------------------------------------------------------------------------------------------")
                """
                if command in self.optimal_order_boxes and command not in other_robot.optimal_order_boxes:
                    other_robot.optimal_order_boxes[command] = self.optimal_order_boxes[command]
                    if command in self.command_min_moves:
                        other_robot.command_min_moves[command] = self.command_min_moves[command]
                    
                elif command in other_robot.optimal_order_boxes and command not in self.optimal_order_boxes:
                    self.optimal_order_boxes[command] = other_robot.optimal_order_boxes[command]
                    if command in other_robot.command_min_moves:
                        self.command_min_moves[command] = other_robot.command_min_moves[command]
                    
                elif command in other_robot.optimal_order_boxes and command in self.optimal_order_boxes:
                    
                    self_min  = self.command_min_moves.get(command)
                    other_min = other_robot.command_min_moves.get(command)
                    
                    if self_min is None and other_min is not None:
                        self.command_min_moves[command] = other_min
                        
                    elif self_min is not None and other_min is None:
                        other_robot.command_min_moves[command] = self_min
                        
                    elif self_min is not None and other_min is not None:
                        
                        if self_min < other_min:
                            other_robot.command_min_moves[command] = self_min
                            other_robot.optimal_order_boxes[command] = self.optimal_order_boxes[command]
                            
                        elif self_min > other_min:
                            self.command_min_moves[command] = other_min
                            self.optimal_order_boxes[command] = other_robot.optimal_order_boxes[command]
                
            # Update update_info lists of both robots.
            self.updated_info = [0]*self.store.num_robots
            self.updated_info[self.num] = -1
            self.updated_info[robot_num] = 1
            
            other_robot.updated_info = [0]*self.store.num_robots
            other_robot.updated_info[other_robot.num] = -1
            other_robot.updated_info[self.num] = 1
            """
            print(self.num,"=self.dict_orders_freq after merging", self.dict_orders_freq)
            print(robot_num,"=other_robot.dict_orders_freq after merging", other_robot.dict_orders_freq)
            print(self.num,"=self.dict_orders_movements after merging", self.dict_orders_movements)
            print(robot_num,"=other_robot.dict_orders_movements after merging", other_robot.dict_orders_movements)
            """
        # After seeing that it is not moving, it checks if there are other robots next to it.
            # Not diagonal movements.
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            
            # Check limits and see if there's another robot.
                # We must differentiate between before and after the merging.
            
            # Before the merging.
            if self.env.now < self.time_merge:
                
                # Again, two cases: west and east.
                # West.
                if self.num < (self.store.num_robots / 2):
                    
                    if 0 <= nx < self.store.height_store and 0 <= ny < self.store.screen and self.store.matrix[nx, ny] == 101:
                        
                        # Detect which robot it is.
                        robot_num = next((k for k, v in self.store.positions_robots.items() if np.array_equal(v, np.array([nx, ny]))), None)
                        
                        # If it doesn't have the updated info of both robots.
                        if self.updated_info[robot_num] == 0:
                            
                            share_info(robot_num)
                        
                # East.
                else:
                    
                    if 0 <= nx < self.store.height_store and self.store.screen <= ny < self.store.length_store and self.store.matrix[nx, ny] == 101:
                        
                        # Detect which robot it is.
                        robot_num = next((k for k, v in self.store.positions_robots.items() if np.array_equal(v, np.array([nx, ny]))), None)
                        
                        # If it doesn't have the updated info of both robots.
                        if self.updated_info[robot_num] == 0:
                            
                            share_info(robot_num)
                        
            # After the merging.        
            else:
                
                if 0 <= nx < self.store.height_store and 0 <= ny < self.store.length_store and self.store.matrix[nx, ny] == 101:
                    
                    # Detect which robot it is.
                    robot_num = next((k for k, v in self.store.positions_robots.items() if np.array_equal(v, np.array([nx, ny]))), None)
                    
                    # If it doesn't have the updated info ob both robots.
                    if self.updated_info[robot_num] == 0:
                        
                        share_info(robot_num)
                        
        
    # Adjusted sigmoid function (to work out probability of executing the following step of the GA).
        # x: Number of commands served.
        # k: Steepness and derivative of the curve:
            # if k>0: the function grows as x does; if k<0, it decreases.
            # if |k| is big, more steep; if small, less steep.
        # x0: Position of the curve in x axis (in fact, it's the central point of the curve).
    def sigmoid(self, x, k, x0):
        return 1 / (1 + np.exp(-k * (x - x0)))
    
    
    
    # Function to calculate all the possible permutations if the robot doesn't know any optimal route.
    def calculate_permutations_boxes_visit(self, command):
        # Differentiate between before/after merging.
        # Before:
        if self.env.now < self.time_merge:
            
            # West:
            if self.num < (self.store.num_robots/2):
              
                    return list(permutations(command))
                
            # East: Add to each element of command self.store.num_boxes/2 units.
            else:
                command_bis = []
                for element in command:
                    command_bis.append(element + int(self.store.num_boxes/2))
                return list(permutations(tuple(command_bis)))
            
        # After:
        else:
            
            # Firstly, obtain the natural combinations.
            total_combinations = list(permutations(command))
            raw = copy.deepcopy(total_combinations)
            # Then, for each element of command, get same combinations but changing that element (representing the box).
            ## with the other one possible.
            for element in command:
                     
                for permutation in raw:
                    
                    permutation = list(permutation)
                    
                    for i in range(len(permutation)):
                        
                        if permutation[i] == element:
                            
                            permutation[i] = element + int(self.store.num_boxes/2)
                
                total_combinations.append(tuple(permutation))
            
            return total_combinations
    
    def genetic_algorithm(self, section, dict_commands_freq):
        
        def objective_function_complete(
            individual,
            *,
            num_items,
            dict_commands_freq,
            dict_items_boxes_complete,
            distances_matrix_complete,
            name_center_stalls,): 
            """
            Individual: numbers 0 to num_boxes - 1 not necessarily in order (list)
            num_items: int
            dict_commands_freq: Dictionary of the form key: command (tuple), value: associated frequency (int)
            center_stalls_complete: int that refers to the column that is at the middle of the store
            dict_items_boxes_complete: Dictionary of the form key: item (int), value: tuple of associated boxes (tuple)
            distances_matrix_complete: Matrix with rows and columns = 0,...,num_boxes - 1 and centre_stall (Pandas df)
                (i,j) Distance between box "i"/stall center and box "j"/stall center
            name_center_stalls: In order to access to its related values in the dataframe (string)
            
            
            """
            # First, we calculate certain figures.
                # Total number of commands attended.
            num_commands = sum(sum(value) for value in dict_commands_freq.values())
            
                # % of appearance of each item.
            dict_item_appear = dict()
            
            for item in range(num_items):
                
                dict_item_appear[item] = sum(sum(value) for key, value in dict_commands_freq.items() if item in key)/num_commands
            
                # Average size of command.
            avg_size_command = sum(dict_item_appear.values())
            
                # % of appearance of each PAIR of items.
            dict_pair_item_appear = dict()
            
            # All possible pairs (i, j) with i<j.
            possible_pairs = list(combinations(range(num_items), 2))
            
            for pair_items in possible_pairs:
                
                dict_pair_item_appear[pair_items] = sum(sum(value) for key, value in dict_commands_freq.items() if pair_items[0] in key and pair_items[1] in key)/num_commands
            
            # Second, we must calculate for each sole item and pair of items the smallest distance possible to get it/them from the stalls_center.
                # Sole item.
            dict_item_min_distance = dict()
            
            for item in range(num_items):
                
                # Get two associated boxes.
                box_a, box_b = dict_items_boxes_complete[item]
                
                # Get positions of associated boxes (it depends on the individual).
                pos_box_a = individual.index(box_a)
                pos_box_b = individual.index(box_b)
                
                # Calculate distances and keep lowest.
                dict_item_min_distance[item] = 2 * min(distances_matrix_complete.loc[name_center_stalls, pos_box_a], distances_matrix_complete.loc[name_center_stalls, pos_box_b])
                
                # Every possible pair.
            dict_pair_items_min_distance = dict()
            
            for pair_items in possible_pairs:
                
                # Get items.
                item_1 = pair_items[0]
                item_2 = pair_items[1]
                
                # Get associated boxes.
                item_1_box_a, item_1_box_b = dict_items_boxes_complete[item_1]
                item_2_box_a, item_2_box_b = dict_items_boxes_complete[item_2]
                
                # Get positions of associated boxes (it depends on the individual).
                pos_item_1_box_a = individual.index(item_1_box_a)
                pos_item_1_box_b = individual.index(item_1_box_b)
                pos_item_2_box_a = individual.index(item_2_box_a)
                pos_item_2_box_b = individual.index(item_2_box_b)
                
                # Calculate distances and keep lowest.      
                path1 = distances_matrix_complete.loc[name_center_stalls, pos_item_1_box_a] + distances_matrix_complete.loc[pos_item_1_box_a, pos_item_2_box_a] + distances_matrix_complete.loc[pos_item_2_box_a, name_center_stalls]
                path2 = distances_matrix_complete.loc[name_center_stalls, pos_item_1_box_a] + distances_matrix_complete.loc[pos_item_1_box_a, pos_item_2_box_b] + distances_matrix_complete.loc[pos_item_2_box_b, name_center_stalls]
                path3 = distances_matrix_complete.loc[name_center_stalls, pos_item_1_box_b] + distances_matrix_complete.loc[pos_item_1_box_b, pos_item_2_box_a] + distances_matrix_complete.loc[pos_item_2_box_a, name_center_stalls]
                path4 = distances_matrix_complete.loc[name_center_stalls, pos_item_1_box_b] + distances_matrix_complete.loc[pos_item_1_box_b, pos_item_2_box_b] + distances_matrix_complete.loc[pos_item_2_box_b, name_center_stalls]
                
                dict_pair_items_min_distance[pair_items] = min(path1, path2, path3, path4)
            
            # Work out value of target function.
            beta = 2 / avg_size_command
            
            result = 0
                
            result += sum([dict_item_appear[item] * dict_item_min_distance[item] for item in range(num_items)])
            
            result += beta * sum([dict_pair_item_appear[pair_items] * dict_pair_items_min_distance[pair_items] for pair_items in possible_pairs])
            
            return (result, )

        # Now we'll consider the cases where the store is still split in our two sections (east and west).
            

        def objective_function_side(individual,
            *,
            num_items,
            dict_commands_freq,
            dict_items_boxes_side,
            distances_matrix_side,
            name_center_stalls,
            side):
            
            """
            Individual: numbers 0 to num_boxes - 1 not necessarily in order (list)
            num_items: int
            dict_commands_freq: Dictionary of the form key: command (tuple), value: associated frequency (int)
            center_stalls_side: int that refers to the column that is at the middle of the subsection of the store
            dict_items_boxes_side: Dictionary of the form key: item (int), value: tuple of associated boxes (tuple)
            distances_matrix_side: Matrix with rows and columns = 0,...,num_boxes - 1 and centre_stall (Pandas df)
                (i,j) Distance between box "i"/stall center and box "j"/stall center
            name_center_stalls: In order to access to its related values in the dataframe (string)
            side: To indicate if it is the west or the east side (string)
            """
            
            # Before anything, if we are in east, we must add num_items to every object of the individual.
            if side == "east":
                individual = [obj + num_items for obj in individual]
            
            # First, we calculate certain figures.
                # Total number of commands attended.
            num_commands = sum(sum(value) for value in dict_commands_freq.values())
            
                # % of appearance of each item.
            dict_item_appear = dict()
            
            for item in range(num_items):
                
                dict_item_appear[item] = sum(sum(value) for key, value in dict_commands_freq.items() if item in key)/num_commands
            
                # Average size of command.
            avg_size_command = sum(dict_item_appear.values())
            
                # % of appearance of each PAIR of items.
            dict_pair_item_appear = dict()
            
            # All possible pairs (i, j) with i<j.
            possible_pairs = list(combinations(range(num_items), 2))
            
            for pair_items in possible_pairs:
                
                dict_pair_item_appear[pair_items] = sum(sum(value) for key, value in dict_commands_freq.items() if pair_items[0] in key and pair_items[1] in key)/num_commands
            
            # Second, we must calculate for each sole item and pair of items the smallest distance possible to get it/them from the stalls_center.
                # Sole item.
            dict_item_min_distance = dict()
            
            for item in range(num_items):
                
                # Get associated box.
                box = dict_items_boxes_side[item][0]
            
                # Get position of associated box (it depends on the individual).
                pos_box = individual.index(box)
                
                if side == "east":
                    pos_box += num_items
                
                # Calculate distance.
                dict_item_min_distance[item] = 2 * distances_matrix_side.loc[name_center_stalls, pos_box]
            
            # Every possible pair.
            dict_pair_items_min_distance = dict()
            
            for pair_items in possible_pairs:
                
                # Get items.
                item_1 = pair_items[0]
                item_2 = pair_items[1]
                
                # Get associated boxes.
                item_1_box = dict_items_boxes_side[item_1][0]
                item_2_box = dict_items_boxes_side[item_2][0]
                
                # Get positions of associated boxes (it depends on the individual).
                pos_item_1_box = individual.index(item_1_box)
                pos_item_2_box = individual.index(item_2_box)
                
                if side == "east":
                    pos_item_1_box += num_items
                    pos_item_2_box += num_items
                    
                
                # Calculate distance.
                dict_pair_items_min_distance[pair_items] = distances_matrix_side.loc[name_center_stalls, pos_item_1_box] + distances_matrix_side.loc[pos_item_1_box, pos_item_2_box] + distances_matrix_side.loc[pos_item_2_box, name_center_stalls]

            
            # Work out value of target function.
            beta = 2 / avg_size_command
            
            result = 0
            
            result += sum([dict_item_appear[item] * dict_item_min_distance[item] for item in range(num_items)])
            
            result += beta * sum([dict_pair_item_appear[pair_items] * dict_pair_items_min_distance[pair_items] for pair_items in possible_pairs])
            
            return (result, )
        
        # Function for the Cycle Crossover.
        def cxCycle(ind1, ind2, first_parent=0):
            """
            Cycle Crossover (CX) for permutations without repetition.
            - ind1, ind2: individuals from DEAP (lists/permutations).
            - first_parent: 0 -> first cycle copied from ind1 (second from ind2, etc.)
                            1 -> first cycle copied from ind2 (second from ind1, etc.)
            Returns (ind1, ind2) modified in-place.
            """
            size = len(ind1)
            # Checks
            if size != len(ind2):
                raise ValueError("Individuals with different sizes.")
            if len(set(ind1)) != size or len(set(ind2)) != size:
                raise ValueError("CX requires permutations without repetition.")
            if set(ind1) != set(ind2):
                raise ValueError("Parents must have same loci.")

            p1 = list(ind1)
            p2 = list(ind2)

            # Map value -> position in P1 to keep cicles in O(1).
            pos_in_p1 = {v: i for i, v in enumerate(p1)}

            used = [False] * size
            take_from_p1 = (first_parent == 0)

            # Empty offspring.
            c1 = [None] * size
            c2 = [None] * size

            for start in range(size):
                if used[start]:
                    continue
                i = start
                cycle_idx = []
                # Continue cycle: i -> value of P2 in i -> position of that value in P1 -> ...
                while not used[i]:
                    used[i] = True
                    cycle_idx.append(i)
                    i = pos_in_p1[p2[i]]

                # Assign alternating cycles. 
                if take_from_p1:
                    for idx in cycle_idx:
                        c1[idx] = p1[idx]
                        c2[idx] = p2[idx]
                else:
                    for idx in cycle_idx:
                        c1[idx] = p2[idx]
                        c2[idx] = p1[idx]

                take_from_p1 = not take_from_p1

            # Write in-place and returns as Deap expects to.
            ind1[:] = c1
            ind2[:] = c2
            return ind1, ind2
                

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        #Individuals are represented by a permutation.
        toolbox = base.Toolbox()

        if section == "complete":
            toolbox.register("permutation", random.sample, range(self.store.num_boxes), self.store.num_boxes)
        else:
            toolbox.register("permutation", random.sample, range(self.store.num_items), self.store.num_items)

        #Structure initializers.
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.permutation)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        if section == "complete":
            toolbox.register(
                "evaluate",
                objective_function_complete,
                num_items = self.store.num_items,                                   
                dict_commands_freq = dict_commands_freq,                 
                dict_items_boxes_complete = self.store.dict_items_boxes_complete,   
                distances_matrix_complete = self.store.distances_matrix_complete,   
                name_center_stalls = self.store.name_center_stalls                  
            )
        

        elif section == "west":
            toolbox.register(
                "evaluate",
                objective_function_side,
                num_items = self.store.num_items,                                   
                dict_commands_freq = dict_commands_freq,                 
                dict_items_boxes_side = self.store.dict_items_boxes_west,   
                distances_matrix_side = self.store.distances_matrix_west,   
                name_center_stalls = self.store.name_center_stalls, 
                side = "west"                  
            )
            
        else:
            toolbox.register(
                "evaluate",
                objective_function_side,
                num_items = self.store.num_items,                                  
                dict_commands_freq = dict_commands_freq,                 
                dict_items_boxes_side = self.store.dict_items_boxes_east,   
                distances_matrix_side = self.store.distances_matrix_east,   
                name_center_stalls = self.store.name_center_stalls, 
                side = "east"                  
            )


        toolbox.register("mate", tools.cxPartialyMatched)
        # toolbox.register("mate", cxCycle, first_parent=0)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0/self.store.num_boxes)
        toolbox.register("select", tools.selTournament, tournsize=2)

        """
        seed = 0
        random.seed(seed)
        """

        pop = toolbox.population(n= self.num_individuals - 1)
        
        # Add here individual that represents the current situation of the store.
        
        current_individual = None
        
        if section == "complete":
            
            current_individual = [self.store.positions_boxes[value] for value in self.store.dict_box_pos_complete.values()]
            
        elif section == "west":
            
            current_individual = [self.store.positions_boxes[value] for value in self.store.dict_box_pos_west.values()]
            
        else:
            
            current_individual = [self.store.positions_boxes[value] - self.store.num_items for value in self.store.dict_box_pos_east.values()]
        
        
        pop.append(creator.Individual(current_individual)) 
        
        
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        """
        stats.register("Avg", np.mean)
        stats.register("Std", np.std)
        stats.register("Min", np.min)
        stats.register("Max", np.max)
        """

        algorithms.eaSimple(pop, toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen, stats=stats,
                            halloffame=hof, verbose=True)
        
        # hof contains the best solution.
        best = hof[0]
        
        # If the genetic was applied onto the east side of the store, we must change the boxes (add self.store.num_items to each box).
        if section == "east":
            
            best = [box + self.store.num_items for box in best]
            
        print("Best solution of the GA: ", best)
        
        print("self.store.positions_boxes before GA: ",self.store.positions_boxes)
            
        # Update self.store.positions_boxes and the matrix that contains all the info about the store.
        if section == "west":
            # columns < screen.
            positions = sorted([pos for pos in self.store.positions_boxes.keys() if pos[1] < self.store.screen])
        elif section == "east":
            # columns >= screen.
            positions = sorted([pos for pos in self.store.positions_boxes.keys() if pos[1] >= self.store.screen])
        else:  # "complete".
            positions = sorted(list(self.store.positions_boxes.keys()))
        
        assert len(best) == len(positions), f"Best ({len(best)}) y positions ({len(positions)}) no cuadran para {section}"
        
        # (Optional Debug)
        # before = [self.store.positions_boxes[p] for p in positions]
        
        # Apply Permutation: position -> box.
        for pos, box in zip(positions, best):
            self.store.positions_boxes[pos] = box
            self.store.matrix[pos[0], pos[1]] = box
        
        # We update the information related to the information for the post-analysis.
        self.store.ga_version += 1
        self.store.current_ga_scope = section
        
        print("self.store.positions_boxes after GA: ",self.store.positions_boxes)

        
    
    # Function that makes every robot eliminate certain information just after the GA has changed the boxes.
    def update_own_info_after_ga(self):
        
        self.dict_orders_movements = {key : [0]*self.store.num_robots for key in self.store.prob_itemsets}
        
        self.updated_info = [1] * self.store.num_robots
        self.updated_info[self.num] = -1
        
        self.optimal_order_boxes = dict()
        
        self.list_routes_before_merging = []
        
        self.command_min_moves = dict()
        
        
    # --------------------------------------------------------------------------------------------------
    
    # Function that defines how the robot works.
    def run(self):
        
        while True:
            
            # If there is already a GA taking place and this robot is affected by it, it must wait. 
            if self.store.ga_in_progress and ((self.num in self.store.ga_participants) or (self.store.ga_scope == "complete")):
                # It this robot hasn't marked yet "Ready", it must do it. 
                ev = self.store.ga_ready_events.get(self.num)
                if ev is not None and not ev.triggered:
                    ev.succeed()
            
                # Wait for the GA to end.
                yield self.store.ga_done_event
            
                # When the GA finishes, the affected robots must update its own information.
                if self.num != self.store.ga_requester:
                    self.update_own_info_after_ga()
            
            # It waits for client to arrive.
            info = yield self.store.waiting_line.get()
                
            # It gets its information.
            id_client = info[0]
            command = info[1]
            print("Robot: ",self.num, ", command: ",command)
            
            # It must request the traffic light for the client, so that he doesn't move until having been served.
            with self.traffic_light_stall.request() as req:
                yield req  
                
                # Indicate that a client is being served. 
                self.store.matrix[self.pos_its_client[0], self.pos_its_client[1]] = 103
                
                # Add elements of the command in dictionary of items to visit.
                self.store.robot_items_not_visited[self.num] = set(command)
            
                # Since the robot has "awaken up" and has a command to attend, it needs to get the required items (taking into account the information it has).
                # Firstly, it must calculate the optimal order (order of BOXES to follow).
                
                # We must consider different scenarios depending on the information available to the robot.
                command_ordered = tuple(sorted(list(command)))
                
                # If it doesn't have any information about optimal routes or it must consider all options because it was an optimal route BEFORE THE MERGING.
                if len(self.optimal_order_boxes) == 0 or (command_ordered in self.list_routes_before_merging and self.env.now > self.time_merge):
                    
                    print("Robot ",self.num, "doesn't have info about optimal_routes or it needs to consider all boxes")
                    optimal_order = self.find_optimal_order(self.calculate_permutations_boxes_visit(command))
                    
                # We must take into account if it has done previously that very command.
                elif command_ordered in self.optimal_order_boxes:
                    
                    print("Robot",self.num," has done this command before and has its info updated")
                    optimal_order = self.optimal_order_boxes[command_ordered]
                    
                    
                # Another possibility is that there's an optimal route which contains the command that we must attend. 
                else:
                    optimal_order = self.find_partial_route(command_ordered)
                    print("Robot ",self.num," Find partial route - Optimal_route: ", optimal_order)
                    
                    # If not, we must see which is the optimal route but taking into account the information we already know of smaller routes.
                    # What we do is 
                        # To take the biggest subset of the command that we already know its best order.
                        # To consider those combinations that include all the elements of the command but 
                        ## taking into account that those elements in the subset are in that very same order.
                    # Then we look for the optimal order but only with those combinations.
                    if optimal_order is None:
                        print("Robot ",self.num, "intelligent combinations")
                        base = self.find_biggest_subset(command_ordered)
                        
                        # If it can base part of its new route on some previous knowledge obtained in find_biggest_subset:
                        if base is not None:
                            print("Base: ",base)
                            intelligent_combinations = self.generate_intelligent_combinations(command, base)
                            optimal_order = self.find_optimal_order(intelligent_combinations)
                        
                        # If not, it must try every possible combination.
                        else:
                            optimal_order = self.find_optimal_order(self.calculate_permutations_boxes_visit(command))
                        
                print("Robot ",self.num," Optimal Order for getting ", command, " is :", optimal_order)
                
                # Add that route to the dictionary of optimal routes. 
                self.optimal_order_boxes[command_ordered] = optimal_order
                
                # If after merging we obtain a route that had to be optimized again, remove it from the list of routes before merging. 
                if self.env.now > self.time_merge and command_ordered in self.list_routes_before_merging:
                    self.list_routes_before_merging.remove(command_ordered)
                
                # Create two lists that will contain the boxes/items that the robots has already visited.
                visited_boxes = []
                visited_items = []
                
                # Start list of occupied_pos.
                occupied_pos = []
                
                # Initialize number of steps to fulfill command.
                number_steps = 0
                
                # Until it doesn't visit all the boxes:
                print("Entra en while while set(visited_items) != set(command)")
                while set(visited_items) != set(command):
                    
                    # Create a boolean to indicate if a robot must recalculate its route to a specific box.
                    boolean_recalculate_route = False
                
                    # Follow steps of the optimal order.
                    for i, box in enumerate(optimal_order):
                        
                        # I write down this because at a certain moment of the simulation, it may find box that was supposed to be found later.
                        if box not in visited_boxes:
                        
                            # It Calculates the shortest path to the box, from its position.
                            box_pos = next((key for key, value in self.store.positions_boxes.items() if value == box), None)
                            route = self.calculate_shortest_path(self.pos, box_pos, box, exclude = occupied_pos)
                            
                            print("Robot ",self.num, ", best route to go from ",self.pos, "to ",box_pos," to get ",box,": ",route)
                            
                            # It may occur that the robot after getting the item from its box can also reach the following object in its route.
                            if len(route) == 0:
                                
                                # It must wait to get the traffic light to move (so that only one robot moves at a time).
                                with self.traffic_light_move.request() as req:
                                    
                                    yield req  
                                
                                    print("Robot ",self.num, ", is already in ",self.pos, " to get ",box, ", len(route): ",len(route))
                                    
                                    # It updates info if needed (with neighboring robots).
                                    self.check_neighbors_update_info()
                                    
                                    # As it found a box, it updates its information and prepares to look for the next box.
                                    print("Robot ", self.num, " found box ", box, " it's looking for at its position: ", box_pos)
                                    visited_boxes.append(box)
                                    related_item = self.store.box_item[box]
                                    visited_items.append(related_item)
                                    self.store.robot_items_not_visited[self.num] = self.store.robot_items_not_visited[self.num] - set([related_item])
                                    print("Robot ",self.num," must still go for ",len(optimal_order[i:]) - 1," boxes, visited_boxes: ",visited_boxes, "while condition: ",set(visited_boxes)==set(command))
                                      
                            # If it has to move (because the length of the route > 0):        
                            else:
                                
                                # Follow the route, step by step:
                                for step in route:
                                    
                                    print(self.env.now, self.num, ", step: ",step, ", box: ", box)
                                    
                                    # It must wait to get the traffic light to move (so that only one robot moves at a time).
                                    with self.traffic_light_move.request() as req:
                                        
                                        yield req  
                                        
                                        # Check if the cell it wants to move in is empty or not.
                                        # print(self.num, step, self.store.matrix[step[0], step[1]])
                                        
                                        if self.store.matrix[step[0], step[1]] == 100:  # Remember that 100 means "empty".
                                        
                                            # Then the robot moves to this position.
                                            
                                            # If last cell wasn't a stoll, put an empty cell.
                                            if not any(np.array_equal(np.array([self.pos[0], self.pos[1]]), value) for value in self.store.positions_stalls.values()):
        
                                                self.store.matrix[self.pos[0], self.pos[1]] = 100  
                                            
                                            # If it was its own stall, put a stall cell (102).
                                            ## It could have been an "else" statement, however, just in case, we check it was its own stall.
                                            elif any(np.array_equal(np.array([self.pos[0], self.pos[1]]), value) for value in self.store.positions_stalls.values()):
                                                
                                                self.store.matrix[self.pos[0], self.pos[1]] = 102
                                            
                                            # We put a 101 (occupied by a robot) in the place where it moves in.
                                            self.store.matrix[step[0], step[1]] = 101 
                                            
                                            # We update the position of the robot.
                                            self.pos = np.array(step)
                                            
                                            # print("Robot ",self.num," has moved to: ",self.pos, ". Proof: ",self.store.matrix[self.pos[0], self.pos[1]])
                                            
                                            # Record current situation with update of the situation of the store.
                                            self.store.positions_robots[self.num] = self.pos
                                            self.graphic()
                                            
                                            # Update number of steps given. 
                                            number_steps = number_steps + 1
                                            
                                            # Restart list of occupied_pos (for the next movement).
                                            occupied_pos = []
                                            
                                            # Before checking boxes next to it, update info if needed (with neighboring robots).
                                            self.check_neighbors_update_info()
                                         
                                            # The place of the box we are looking for is in the list of close boxes' locations. If the robot is there, update all the related information.
                                            if step == route[-1]:
                                                
                                                print("Robot ",self.num, " found box ",box, " it's looking for at its position: ",box_pos)
                                                visited_boxes.append(box)
                                                related_item = self.store.box_item[box]
                                                visited_items.append(related_item)
                                                self.store.robot_items_not_visited[self.num] = self.store.robot_items_not_visited[self.num] - set([related_item])
                                                #print("Robot ",self.num," with boxes",optimal_order,", visited_boxes: ",visited_boxes)
                                                    
                                        # If cell is not empty, recalculate route.        
                                        else:
                                            
                                            # If the step it wants to move in is not in the list of occupied positions, then add it.
                                            if (step[0], step[1]) not in occupied_pos:
                                                occupied_pos.extend([(step[0], step[1])])
                                            print(f"Robot {self.num} found {step} occupied while looking for box {box}. Recalculating route... occupied_pos: ",occupied_pos)
                                            
                                            # Now it will have to recalculate its route to the following box.
                                            boolean_recalculate_route = True
                                            
                                            # Before recalculating route, update information with neighboring robots if needed.
                                            self.check_neighbors_update_info()
                                            
                                            # As it must recalculate its route to the next box:
                                            break
                                            
                                    #print(f"Robot {self.num} waits 0.5 sec until trying to move again")
                                    # Wait until it tries to move again (so that it releases the traffic light).
                                    yield self.env.timeout(0.5 + random.uniform(0, 3))  
                                    
                                    
                                # If the robots must change (or have changed) the order of things to visit: 
                                if boolean_recalculate_route:
                                    
                                    print(f"Robot {self.num} changed the route to get a specific box (inner loop)")
                                    # Wait until it tries to move again (so that it releases the traffic light).
                                    yield self.env.timeout(0.5 + random.uniform(0, 3))  
                                    
                                    break
                                
                                yield self.env.timeout(0.5)  
                                
                            
                    
                # If the robot makes it to this point, then it will have picked all the products for the customer, so it goes back to its stall (initial position).
                # To let other robots take the traffic light to avoid this robot to have it for a long time.
                yield self.env.timeout(0.5)
                 
                print(f"Robot {self.num} starts its way back to stall from ",self.pos, " to ",self.pos_its_stall)
                
                # It must calculate the route to go back to its stall.
                route_back = self.calculate_shortest_path(self.pos, tuple(self.pos_its_stall), None)
                
                # Boolean that indicates if the robot has arrived to its stall or not.
                arrived_to_stall = False
                
                # We restart the occupied_pos list.
                occupied_pos = []
                
                # Until it doesn't arrive to its stall:
                print("Entra en bucle while not arrived_to_stall")
                while not arrived_to_stall:
                    
                    #print(self.num, ", route back: ",route_back)
                    
                    # For each step in its route back to the stall:
                    for step in route_back:
                        
                        #print(self.env.now, self.num, ", step: ",step, " going back")
                        
                        # It asks for the traffic light that enables to move.
                        with self.traffic_light_move.request() as req:
                            yield req
                            
                            # If the next step is empty (= 100):
                            if self.store.matrix[step[0], step[1]] == 100:  
                                # The robot moves to this position (= 101).
                                self.store.matrix[self.pos[0], self.pos[1]] = 100  
                                self.pos = np.array(step)
                                self.store.matrix[self.pos[0], self.pos[1]] = 101 
                                
                                # Create graphic and update situation of the store.
                                self.store.positions_robots[self.num] = self.pos
                                self.graphic()
                                
                                # Update number of steps made to attend the command.
                                number_steps = number_steps + 1
                                
                                # Restart occupied_pos list.
                                occupied_pos = []
                            
                            # If the next step is its own stall:
                            elif np.array_equal(np.array([step[0], step[1]]), self.pos_its_stall): 
                                print(f"Robot {self.num} arrived to its stall, completed client {id_client} order")
                                arrived_to_stall = True
                                
                                # If there is a GA and this robot is affected, we must mark it in the barrier.
                                if self.store.ga_in_progress:
                                    if (self.num in self.store.ga_participants) or (self.store.ga_scope == "complete"):
                                        ev = self.store.ga_ready_events.get(self.num)
                                        if ev is not None and not ev.triggered:
                                            ev.succeed()
                                
                                # Update movement (we don't put that the stall in the matrix is 101).
                                    # Where it was -> 100 (empty).
                                self.store.matrix[self.pos[0], self.pos[1]] = 100  
                                self.pos = np.array(step)
                                
                                # Create graphic and update situation of the store.
                                self.store.positions_robots[self.num] = self.pos
                                self.graphic()
                                
                                # Restart occupied_pos list.
                                occupied_pos = []
                                
                                break
                            
                            # If cell is not empty, recalculate route.
                            else:
                                # Add the occupied cell to the list of occupied positions.
                                if (step[0], step[1]) not in occupied_pos:
                                    occupied_pos.append((step[0], step[1]))
                                    
                                print(f"Robot {self.num} found {step} occupied while going to stall. Recalculating route... occupied_pos: ",occupied_pos)
                                route_back = self.calculate_shortest_path(self.pos, tuple(self.pos_its_stall), None, exclude=occupied_pos)
                                #print("Robot ",self.num, ", NEW best route to go home: ",route_back)
                                if len(route_back) == 0:
                                    # There is no way not: wait and try again with exclude set to nothing (blocking robot may move). 
                                    yield self.env.timeout(0.5 + random.uniform(0, 3))
                                    occupied_pos = []
                                    route_back = self.calculate_shortest_path(self.pos, tuple(self.pos_its_stall), None, exclude=None)
                                    if len(route_back) == 0:
                                        # If there is not way yet, wait a little more and return to "for" loop. 
                                        yield self.env.timeout(0.5 + random.uniform(0, 3))
                                        break  # Tries again in the following iteration of the while not arrived_to_stall.
                                else:
                                    # There was an alternate way -> Break "for" loop to start it.
                                    break
                                
                                break
                        
                        # So that it releases the traffic light.
                        yield self.env.timeout(0.5)  
                        
                
                # If the robot has made it here, it means that it arrived to its stall.
                # The robot updates its own average of number of movements.
                self.dict_orders_movements[command][self.num] = (self.dict_orders_movements[command][self.num] * self.dict_orders_freq[command][self.num] + number_steps)/(self.dict_orders_freq[command][self.num] + 1)
                
                # Update information of the post-analysis.
                self.store.metrics.append({
                    "timestamp": float(self.env.now),
                    "robot_id": int(self.num),
                    "client_id": int(id_client),
                    "ga_version": int(self.store.ga_version),
                    "ga_scope": (self.store.current_ga_scope if self.store.current_ga_scope is not None else
                                 ("pre" if self.env.now < self.time_merge else "post")),  
                    "phase": ("pre_merge" if self.env.now < self.time_merge else "post_merge"),
                    "items_count": int(len(command)),
                    "command_items": tuple(int(x) for x in command),   
                    "steps_per_order": int(number_steps)
                })
                
                
                # It may also update its lowest number of movements for that command.
                if command not in self.command_min_moves.keys() or (command in self.command_min_moves.keys() and number_steps < self.command_min_moves[command]):
                    
                    self.command_min_moves[command] = number_steps
                    print("-----------------------------------------------------------------------------------------------")
                    print("Robot ", self.num, ", updated self.command_min_moves: ", self.command_min_moves)
                    print("-----------------------------------------------------------------------------------------------")
                    
                """
                print("-----------------------------------------------------------------------------------------------")
                print(self.num, ", dict_orders_movements[ command:",command,"][self.num:",self.num,"]: ",self.dict_orders_movements[command][self.num])
                print("-----------------------------------------------------------------------------------------------")
                """
                # The robot updates its own number of commands fulfilled of that specific type of command.
                self.dict_orders_freq[command][self.num] = self.dict_orders_freq[command][self.num] + 1
                """
                print("-----------------------------------------------------------------------------------------------")
                print(self.num, ", dict_orders_freq[ command:",command,"][self.num:",self.num,"]: ",self.dict_orders_freq[command][self.num])
                print("-----------------------------------------------------------------------------------------------")
                """
                # Update update_info list.
                self.updated_info = [0]*self.store.num_robots
                self.updated_info[self.num] = -1          
                
                # So that the client "leaves" (it does so in the next yield).
                self.store.matrix[self.pos_its_client[0], self.pos_its_client[1]] = 100
                
                # Now the GA part takes place, we update the number of commands attended by all robots.
                before = self.total_commands_served
                self.total_commands_served = np.sum([np.array(mylist) for mylist in self.dict_orders_freq.values()])
                
                # If total_commands_served has surpassed the following multiple of module, restart the boolean so that it can generate a new iteration of the GA.
                if (before % self.module_for_genetic) > (self.total_commands_served % self.module_for_genetic):
                    self.store.ga_recent_creation = False
                
                
                # See if work out a new iteration of the GA.
                if random.random() < self.sigmoid(self.total_commands_served % self.module_for_genetic, k=0.3, x0=self.module_for_genetic/2) and not self.store.ga_recent_creation:
                
                    
                    self.store.ga_recent_creation = True
                
                    # Only one robot can execute the GA at a time.
                    with self.store.ga_lock.request() as ga_req:
                        yield ga_req
                        
                        print("-----------------------------------------------------------------------------------------------")
                        print("Robot ", self.num, "wants to work out GA iteration")
                        print("-----------------------------------------------------------------------------------------------")
                
                        # First, we must determine the initial scope/section of the GA. 
                        initial_scope = "complete" if self.env.now >= self.time_merge else ("west" if self.num < (self.store.num_robots/2) else "east")
                
                        # Second, we must warn to the affected robots that a GA is going to take place soon.
                        self.store.ga_in_progress = True
                        self.store.ga_scope = initial_scope
                        self.store.ga_requester = self.num
                        self.store.ga_participants = self.store.robots_in_section(initial_scope)
                
                        # Barriers to the affected robots (so that they wait while the GA takes place).
                        self.store.ga_done_event = self.env.event()
                        self.store.ga_ready_events = {rid: self.env.event() for rid in self.store.ga_participants}
                        # # The robot that is asking for the GA is ready (because it is at its own stall).
                        if not self.store.ga_ready_events[self.num].triggered:
                            self.store.ga_ready_events[self.num].succeed()
                
                        # Third, the robot that will execute the GA must wait for all affected robots to be in their respective stalls.
                        yield simpy.events.AllOf(self.env, list(self.store.ga_ready_events.values()))
                        
                        print("-----------------------------------------------------------------------------------------------")
                        print("Robot ", self.num, "sees that all robots are in their stalls, it works out GA iteration")
                        print("-----------------------------------------------------------------------------------------------")
                
                        # Fourth, if the screen that divided the store has vanished as we waited, we must now consider the complete problem.
                        if (self.store.ga_scope != "complete") and (self.env.now >= self.time_merge):
                            print(f"[{self.env.now}] Robot {self.num}: merge happened while waiting for section GA -> CANCEL attempt and release robots")
                            # Do NOT run GA; free all participants and reset GA coordination objects.
                            # Free whichever robot waiting for being ready in its own stall.
                            for rid, ev in list(self.store.ga_ready_events.items()):
                                if (ev is not None) and (not ev.triggered):
                                    ev.succeed()
                        
                            # Announces ending of the GA (if any robot was waiting at the end barrier).
                            if (self.store.ga_done_event is not None) and (not self.store.ga_done_event.triggered):
                                self.store.ga_done_event.succeed()
                        
                            # Restar coordination of GA.
                            self.store.ga_in_progress = False
                            self.store.ga_scope = None
                            self.store.ga_requester = None
                            self.store.ga_participants = set()
                            self.store.ga_ready_events = {}
                            self.store.ga_recent_creation = False
                        
                            # Releases control and aborts this try (does not execute GA here). 
                            yield self.env.timeout(0)
                            continue
                
                        # Fifth, execute GA.
                        scope_for_ga = self.store.ga_scope
                        if scope_for_ga == "complete":
                            self.genetic_algorithm("complete", self.dict_orders_freq)
                        # 'west' or 'east'.
                        else:
                            self.genetic_algorithm(scope_for_ga, self.dict_orders_freq)  
                
                        # Sixth, the robot that has executed the GA must update its own information. 
                        self.update_own_info_after_ga()
                
                        # Now, we must finish the GA and "wake up" every other robot.
                        self.store.ga_in_progress = False
                        self.store.ga_recent_creation = True
                        self.store.ga_done_event.succeed()
                
                        # Reinitialize the objects that help us to coordinate this process (for the next time; it is optional). 
                        self.store.ga_ready_events = {}
                        self.store.ga_participants = set()
                        self.store.ga_scope = None
                        self.store.ga_requester = None
                
                        # To let other robots move
                        yield self.env.timeout(0)
                
                else:
                    print("Robot", self.num, " has tried GA unsuccessfully")
                    
                    
                yield self.env.timeout(1)


# Class that generates clients following an exponential process.
class Client_Generator(object):
    def __init__(self, env, store, arrival_rate, time_merge):
        self.env = env
        self.store = store
        self.arrival_rate = arrival_rate
        self.time_merge = time_merge
        # For counting clients.
        self.client_id = 1
        
        
    def run(self):
        
        while True:
            # Generate the random arrival time between clients.
            time_between = np.random.exponential(scale=1/self.arrival_rate)
            
            # Wait that time.
            yield self.env.timeout(time_between)
            
            # It is not necessary to differentiate between before and after the merging because the requests will be the same.
            
            # We have the Client ID; we create its command.
            client_command = random.choices(
                population=list(self.store.prob_itemsets.keys()),
                weights=list(self.store.prob_itemsets.values()),
                k=1)[0]
            
            # We put it to wait in the queue.
            yield self.store.waiting_line.put([self.client_id, client_command])
                
            #print(f"{self.env.now}: Client {self.client_id} waiting in queue")
            self.client_id += 1
    
# ------------------------------------------------------------------------------------------------------- EXECUTION OF THE ALGORITHM -----------------------------------------------------------------------------------------------------
#env = simpy.rt.RealtimeEnvironment(factor=0.5)
env = simpy.Environment()
traffic_light_move = simpy.Resource(env, 1)

# If possible, num_boxes >= 8.
    # The total number of different articles will be its half.
num_boxes, num_robots = 12, 10

# Maximum number of items in the order.
max_items_order = 3

if num_boxes % 2 != 0:
    num_boxes = num_boxes + 1

if num_robots % 2 != 0:
    num_robots = num_robots + 1

if max_items_order > num_boxes/2:
    max_items_order = num_boxes/2

# Frequency of commands attending to their length.
length_weights = {1: 0.1, 2: 0.3, 3: 0.6}

# Optional: If the user wants to enter the probabilities of the individual items .
# It could be for instance {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.05, 7: 0.05}.
item_probs = None

if item_probs is None:
    
    # Make sure all possible lengths are included; if not, they are random.
    if list(length_weights.keys()) != list(range(1, max_items_order+1)):
        
        length_weights = {k : random.uniform(0, 1) for k in range(1, max_items_order+1)}
    
    
time_merge = 0

total_time = 1500

# Probability of two parents to mate in GA.
cxpb = 1

# Probability of a member of the offspring to mutate.
mutpb = 0.1

# Number of generations in the GA.
ngen = 100

# Number of individuals in the GA (including the current distribution of boxes).
num_individuals = 50

# Number of total commands (related to the time to execute the GA).
module_for_genetic = 50

# Initialize the store.
store = Store(num_boxes, num_robots, max_items_order, length_weights, item_probs, env)

# Initialize the robots.
dict_robots = dict()
for i in range(num_robots): 
    
    robot = Robot(env, i, store, traffic_light_move, time_merge, cxpb, mutpb, ngen, num_individuals, module_for_genetic)
    dict_robots[i] = robot
    env.process(robot.run())

store.dict_robots = dict_robots
    
# Initialize the client generator (assume that the time between arrivals follows an Exponential Distribution because the arrival process is a Poisson Process).
# Lambda = Arrival rate.
lmbda = 5 
generator = Client_Generator(env, store, lmbda, time_merge)
env.process(generator.run())

env.run(until = total_time)

print("End of simulation")

# Save information for the post-analysis.
df_metrics = pd.DataFrame(store.metrics)
df_metrics.to_csv("./metrics_orders_"+str(datetime.now().strftime("%Y-%m-%d %H_%M_%S"))+".csv", index=False)
print("Saved metrics to metrics_orders.csv with", len(df_metrics), "rows")

if total_time <=200:

    # Create GIF.
    output_path = "./simulacion_screen_"+str(datetime.now().strftime("%Y-%m-%d %H_%M_%S"))+".gif"
    
    images = []
    
    for fig in store.list_graphics:
        # Adjust margings automatically.
        fig.tight_layout()
        # Save figure in a BytesIO object.
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)  # Go back to beginning of buffer.
        # Convert image into format PIL before closing buffer.
        image = Image.open(buf).copy()  # Copy image to avoid problems when closing buffer.
        images.append(image)
        buf.close()  # Close buffer after the copy.
    
    images[0].save(
        output_path, 
        save_all=True, 
        append_images=images[1:], 
        duration=100,  # Duration of each frame in miliseconds.
        loop=0  # 0 for infinte loop.

    )
