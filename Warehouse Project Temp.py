# -*- coding: utf-8 -*-
"""
Carlos Soria Elizalde

Operative Robots in Store

"""

# The purpose of this code is to create a store which works with robots; they fulfill clients' demands, which consist of combinations of products. 
# When a robot receives a client, it calculates the optimal order to visit the boxes, and goes for them
    # It uses previous information to simplify calculations
    # In the beggining, there's a panel which splits the store into two uncommunicated sections or areas. 
    # At a certain moment of the simulation, the panel vanishes, and the robots can go to the entire shop
# When the robot has all the items, it returns to its stall
# If two robots meet, they communicate between themselves the optimal routes they know for getting a command, the number of commands served and the total distance traveled
# When robots finish attending a client, they may create association rules

# First of all, I have to be able to create a logistics centre with hyperparameters to be chosen by the user: 
    
    # Number of different items
    # Number of different robots / stations 
    # In this version, both must be even (because we are going to divide the center in two parts)
    
# Based on that, build a sufficiently large logistics centre with the following structure

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
# To create the demand patterns: 
#    
    # Frequency: Random   
    # Orders: In this version, I must create a function to create the global demmand patterns


import simpy
import numpy as np
import simpy.rt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import io
import random
from itertools import product, combinations, permutations, chain
import heapq
import copy

class DimmensionProblem(Exception):
    pass


class Store(object):
    
    # Constructor: Given a number of articles and a number of robots, we build the store center
    def __init__(self, num_boxes, num_robots, max_items_order, env):
        
        self.num_boxes = num_boxes
        self.num_robots = num_robots
        self.max_items_order = max_items_order
        self.env = env
        
        self.dict_robots = None
        
        # Calculate minimum of squares in the matrix
        
        # The positions of the boxes cannot touch each other horizontal, vertical or diagonally: the distance must be greater than sqrt(2).
        # How do I know if the storage area is large enough to hold all the boxes?
        #
        # A box takes up a 3x3 space if they cannot touch each other even diagonally.
        # So, if we have n boxes, we must have at least nx(3x3) squares of storage space = 9n squares.
        # Also, if we have R robots and stalls, if we want each robot (which occupies 1x1 of space) to have its own stall
        # The length of the centre has to be max(R, sqrt(9n)+1)
        
        self.length_store = max(self.num_robots, int((9*self.num_boxes)**0.5) + 1)
        
        # In this problem, if length_store is odd, make it even
        if self.length_store % 2 != 0:
            self.length_store = self.length_store + 1
        
        # And the height?
        
        # We will leave 3 spaces for customer service area (no boxes in the way) and one extra line for the customers.
        # So the height of the store is 3 + 1 + [sqrt(9n) + 1] = 5 + sqrt(9n)
        
        self.height_store = int((9*self.num_boxes)**0.5) + 5
        
        # Therefore, we initialize the matrix like this (the '100' represents the empty space)
        
        self.matrix = np.zeros((self.height_store, self.length_store)) + 100
        
        self.screen = self.length_store / 2
        
        # We initialize four dictionaries for keeping track of the positions of the robots, the boxes and the stalls and where the clients stand 
        self.positions_robots = dict()
        self.positions_boxes = dict()
        self.positions_stalls = dict()
        self.positions_clients = dict()
        
        
        # Now, I must choose randomly the positions of the boxes and locate the stalls in the most balanced way
        
        # Boxes: As there are initially two sections of the store, we must put half the boxes in the west side and the another half in the east side
            
            # West: They must be in the submatrix(:-5, :self.screen) because we must preserve the last 4 lines for the customer service area and the customer zone
            
                    # And because they must be in the west side
        
        max_row_box = self.height_store - 4
        
        boxes_placed = 0
        
        # We generate all the possible positions (initially they are all available)
        
        feasible_places = list(product(range(max_row_box), range(self.length_store)))
        
        # Until we have not placed every box in the west side
        
        while boxes_placed < self.num_boxes/2:
            
            # In case the assignation is unfeasible
            
            if boxes_placed < self.num_boxes/2 and len(feasible_places) == 0:
                
                print("Dimmension Problem in the west side!")
            
            # We take a random position
            
            pos = random.choice(feasible_places)
            
            # If it is in the west side
            if pos[1] < self.screen:
             
                self.positions_boxes[tuple(pos)] = boxes_placed
                
                self.matrix[pos[0],pos[1]] = boxes_placed
                print("Box ",boxes_placed, ": ",[pos[0],pos[1]])
                
                # We must eliminate those positions that are horizontally/diagonally/vertically next to this position (taking into account boundaries)
                
                # Generate neighbors
                neighbors = self.generate_neighbors(pos, max_row_box - 1, self.length_store - 1)
                
                # Filter the list of pairs
                feasible_places = list([pair for pair in feasible_places if tuple(pair) not in neighbors])
                
                # Update boxes_placed
                boxes_placed = boxes_placed + 1
                
        # East: They must be in the submatrix(:-5, self.screen:) because we must preserve the last 4 lines for the customer service area and the customer zone
        
                # And because they must be in the east side
        
        while boxes_placed < self.num_boxes:
            
            # In case the assignation is unfeasible
            
            if boxes_placed < self.num_boxes and len(feasible_places) == 0:
                
                print("Dimmension Problem in the east side!")
            
            # We take a random position
            
            pos = random.choice(feasible_places)
            
            # If it is in the west side
            if pos[1] >= self.screen:
                
                self.positions_boxes[tuple(pos)] = boxes_placed
                
                self.matrix[pos[0],pos[1]] = boxes_placed
                print("Box ",boxes_placed, ": ",[pos[0],pos[1]])
                
                # We must eliminate those positions that are horizontally/diagonally/vertically next to this position (taking into account boundaries)
                
                # Generate neighbors
                neighbors = self.generate_neighbors(pos, max_row_box - 1, self.length_store - 1)
                
                # Filter the list of pairs
                feasible_places = list([pair for pair in feasible_places if tuple(pair) not in neighbors])
                
                # Update boxes_placed
                boxes_placed = boxes_placed + 1
        
            
        # Once here, the boxes have been successfully placed.
        
        # We create a dict where the key is the number of the item and the value a list with the two boxes (one per side) that have the item
            # If there are 8 boxes (4 articles/items), boxes 0 and 4 will have item 0, boxes 1 and 5 the item 1...
        self.item_boxes = dict()
        for i in range(int(self.num_boxes/2)):
            self.item_boxes[i] = [i, int((self.num_boxes/2)+i)]
        
        # We create a dict where the key is the number of the box and the value its associated item
            # If there are 8 boxes (4 articles/items), box 0 : item 0, box 1: item 1, ... , box 4 : item 0, ...
        self.box_item = dict()
        for i in range(int(self.num_boxes/2)):
            self.box_item[i] = i
            self.box_item[i + int(self.num_boxes/2)] = i
            
        
        # Let's go with the stalls (theoretically it's the same as the normal problem)
        stalls_positions = self.distribute_positions(self.num_robots, self.length_store)
        
        # Add them to the dictionaries (we can not assign yet the robot to its stall, for that we need to initialize the robots)
        for i,stall in enumerate(stalls_positions):
            
            self.positions_stalls[i] = np.array([self.height_store - 2, stall]) 
            
            self.positions_clients[i] = np.array([self.height_store - 1, stall])
        
        # Create a waiting line for the clients
        self.waiting_line = simpy.Store(env)
        
        # Create a list of traffic lights for the stalls, so that the client remains in the stall until it is served
        self.traffic_lights_stalls = [simpy.Resource(self.env, capacity=1) for _ in range(self.num_robots)]
        
        # The last thing to consider is the articles and their chances of being asked (alone or in groups)
        #self.prob_itemsets, self.west_prob_itemsets, self.east_prob_itemsets, self.sum_probs_west, self.sum_probs_east = self.design_article_dict()
        self.prob_itemsets = self.design_article_dict()
        
        # Dictionary of robots and the items they have still to go for in a certain moment of the simulation
        self.robot_items_not_visited = dict()
        
        # Create a list of graphics to show it later
        self.list_graphics= []
        
        # Boolean to stop the simulation
        self.stop_simulation = False
        
        # Define dictionary of colors so that a key is a "general" color, and its value is a list with colors ordered so that the first colors are really different between each other
        self.palette = {
                        "red": ["#DC143C", "#800000", "#FF2400", "#FF007F", "#CB4154"],
                        "green": ["#32CD32", "#228B22", "#808000", "#98FF98", "#50C878"],
                        "blue": ["#4169E1", "#87CEEB", "#008080", "#0047AB"],
                        "purple": ["#8A2BE2", "#E6E6FA", "#DDA0DD", "#9966CC", "#DA70D6"],
                        "brown": ["#D2691E", "#D2B48C", "#A0522D", "#635147", "#F5F5DC"],
                        "orange": ["#FFA500", "#CC5500", "#FFDAB9", "#FBCEB1", "#B7410E"],
                    }
        
        # We create a dictionary of colors and codes
        self.create_colors_dict()
        
        
    def create_colors_dict(self):
        
        
        # 100 - "white" - Nothing
        # 101 - "black" - Robot
        # 102 - "grey" - Stall
        # 103 - "pink" - Client
        # Other numbers and colors are for the different boxes 
        self.colors = {100: 'white', 101: 'black', 102: 'grey', 103: "#FF00FF"}
        
        # Depending on the number of articles, we select different colors 
        self.get_k_elements_by_order()
        
    # Function that gets different colors for the boxes (trying to have the most diversed colors possible)
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
        
    # Function that initializes a robot    
    def initialize_robot(self, num):
        
        pos = self.positions_stalls[num]
            
        self.positions_robots[num] = pos 
        
        self.robot_items_not_visited[num] = []
        
        traffic_light_stall = self.traffic_lights_stalls[num]
        
        return pos, traffic_light_stall
    
    
    # Function to work out the neighbors of a selected position horizontal/vertical/diagonally (1 = distance)
    def generate_neighbors(self, pos, x_max, y_max):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                new_x = pos[0] + dx
                new_y = pos[1] + dy
                if 0 <= new_x <= x_max and 0 <= new_y <= y_max:  # Ensure boundaries
                    neighbors.append((new_x, new_y))
        return neighbors
    
    
    # Function to distribute the stalls in the most balanced way
    def distribute_positions(self, num_robots, length_store):
        # Step size as a float
        step_size = (length_store - 1) / (num_robots - 1)
    
        # Generate positions with rounding
        positions = [round(i * step_size) for i in range(num_robots)]
    
        # Ensure uniqueness and within range
        positions = sorted(set(min(p, length_store - 1) for p in positions))
    
        # Adjust if fewer positions than robots
        while len(positions) < num_robots:
            for i in range(length_store):
                if i not in positions:
                    positions.append(i)
                    if len(positions) == num_robots:
                        break
    
        return sorted(positions)
 
    
    # Function to create the probabilities of each possible command
    def design_article_dict(self):
        
        # Step 1: Generate all possible sets of 1, 2... self.max_items_order articles
        max_articles = min(self.max_items_order, int(self.num_boxes/2))
        all_sets = []
        for r in range(1, max_articles + 1):
            all_sets.extend(combinations(list(range(int(self.num_boxes/2))), r))

        # Step 2: Calculate probabilities for each set
        article_probs = {itemset: random.uniform(0, 1) for itemset in all_sets}

        # Step 3: Normalize probabilities of all sets
        total_set_prob = sum(article_probs.values())
        normalized_set_probs = {k: v / total_set_prob for k, v in article_probs.items()}
        
        return normalized_set_probs


class Robot(object):
    def __init__(self, env, num, store, traffic_light_move, time_merge):
        
        self.env = env
        self.num = num
        self.store = store
        self.traffic_light_move = traffic_light_move
        self.time_merge = time_merge
        
        # We call to the function of the Store class "initialize_robot"
        self.pos, self.traffic_light_stall = self.store.initialize_robot(self.num)
        
        # The position in which the client it serves stands is adding 1 to the row of the pos
        self.pos_its_client = np.array([self.pos[0]+1, self.pos[1]])
        self.pos_its_stall = self.pos
        #print("Robot ",num,", stall position: ",self.pos, ", client pos: ",self.pos_its_client)
        
        # It must keep some information of the things the different robots are doing
            # To keep orders a robot is doing and how many times
            # Key: Command, Value: List where pos j means how many times robot j has completed the command
        self.dict_orders_freq = {key : [0]*self.store.num_robots for key in self.store.prob_itemsets}
        
            # To keep orders and number of movements for complying with
            # Key: Command, Value: List where pos j means the sum of the steps for robot j to complete the command
        self.dict_orders_movements = {key : [0]*self.store.num_robots for key in self.store.prob_itemsets}
        
        # To know which other robots have the latest version of the information of a particular one
            # 0 -> Not Updated
            # 1 -> Updated
            # -1 -> Itself
        self.updated_info = [1] * self.store.num_robots
        self.updated_info[self.num] = -1
        
        # For helping us determining the frequency of calculating association rules
            # The more it approaches to this value or any multiple, more chances of working out association rules
            # If it has recently worked out rules, until it doesn't surpass the following multiple of module, it can't calculate again association rules
        self.module_for_association = 50
        self.recent_creation_rules = False
        self.total_commands_served = 0
        
        
        # Dictionary that contains the optimal order for a certain command
            # Key: Command, Value: Optimal order of boxes, not items 
            # This will be used for avoiding using Dijkstra in certain situations
            # Ex: If we know that the optimal order for command (0,1,2) is (2,1,0), and we have the command (0,1,2,3,4) (no more items exist)
                # Suppose that the screen still exists
                # The commands we must try are (3, 4, 2, 1, 0), (4, 3, 2, 1, 0), (2, 4, 3, 1, 0), (2, 3, 4, 1, 0), 
                # (2, 1, 3, 4, 0), (2, 1, 4, 3, 0), (2, 1, 0, 3, 4), (2, 1, 0, 4, 3)
            # Ex: If we know that the optimal order for command (0,1,2) is (2,1,0), and we have the command (0,1,2,3,4) (no more items exist)
                # Suppose that the screen doesn't exist anymore
                # The commands we must try are 
                    # all mentioned before 
                    # all combinations of 0,1,2 (they must appear in that order) and 8 and 9
                    # all combinations of 5,6,7,8,9
        self.optimal_order_boxes = dict()
        
        # Create also a list with those commands whose routes have been calculated before the merging
        ## This helps us to determine whether an optimal route (associated with the command of the list)
        ## Has been optimised in the "after merge" phase or not
        ### In the beginning, before merging, all commands whose optimal route has been obtained are added
        ### After merging, the robot sees if the command is in here:
            # If it is, we must take into account possible boxes in the other half of the store
            # If it is not, or its route hasn't been calculated, or it has been updated (reliable)
        
        self.list_routes_before_merging = []
   

    # Function to draw the graphic of the situation in a certain moment
    def graphic(self):
        
        # Adjust size of figure
        fig = plt.figure(figsize=(21, 14))  
        
        # Create a grid of 2x2
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])  
        
        # The graphic is the upper half
        ax = fig.add_subplot(gs[0, :])  
        
        # The robots-boxes info is the lower-left quarter
        info_ax = fig.add_subplot(gs[1, 0])  
        
        # Legend in lower right quarter
        legend_ax = fig.add_subplot(gs[1, 1])  
    
        # Hide axis in the lower subplots 
        info_ax.axis('off')
        legend_ax.axis('off')
        
    
        # Draw store situation. For each position: 
        for x in range(self.store.height_store):
            for y in range(self.store.length_store):
                
                # We obtain the value of the cell
                value = self.store.matrix[x, y]
    
                # There is a special case: when a robot is on a stall. In that case, we print the robot with the label on the cell
                if value == 102 and any(np.array_equal(value, np.array([x, y])) for value in self.store.positions_robots.values()):
                    ax.add_patch(plt.Rectangle((y, x), 1, 1, color=self.store.colors[101]))
                    
                    # Get number of robot and put label
                    robot_num = next((k for k, v in self.store.positions_robots.items() if np.array_equal(v, np.array([x, y]))), None)
                    ax.text(y + 0.5, x + 0.5, str(int(robot_num)), color='white', ha='center', va='center', fontsize=12)
                
                # Else if it is a box, take into account the item it has
                elif value < 100:
                    ax.add_patch(plt.Rectangle((y, x), 1, 1, color=self.store.colors[int(value%(self.store.num_boxes/2))]))
                    # If it's box, we put its label on the cell
                    box_text = "Box: "+str(int(value))+", Item: "+str(int(value%(self.store.num_boxes/2)))
                    ax.text(y + 0.5, x + 0.5, box_text, color='black', ha='center', va='center', fontsize=12)
                    
                # Else, print the color represented by the value   
                else:
                    ax.add_patch(plt.Rectangle((y, x), 1, 1, color=self.store.colors[value]))
    
                # If it's a robot, guess which robot it is and put its label on the cell
                if value == 101:
                    robot_num = next((k for k, v in self.store.positions_robots.items() if np.array_equal(v, np.array([x, y]))), None)
                    ax.text(y + 0.5, x + 0.5, str(int(robot_num)), color='white', ha='center', va='center', fontsize=12)
    
        # Set limits and grid 
        ax.set_xlim(0, self.store.length_store)
        ax.set_ylim(self.store.height_store, 0)
        ax.axhline(self.store.height_store - 1, color='black', linewidth=2)  # Horizontal Separation Line
        
        if env.now < time_merge: 
            
            ax.axvline(self.store.screen, color='black', linewidth=2)  # Vertical Separation Line
            
        ax.set_xticks(range(self.store.length_store))
        ax.set_yticks(range(self.store.height_store))
        ax.grid()
        ax.set_title("t = " + str(round(self.env.now, 4)))
    
        # Create information of robots and boxes to visit 
        robot_info = "Robots and Target Items:\n"
        for robot_id, boxes in self.store.robot_items_not_visited.items():
            if len(boxes) == 0:
                robot_info += "Robot "+str(robot_id)+": {}\n"
            else:
                robot_info += f"Robot {robot_id}: {boxes}\n"
    
        # Draw the information in the downleft subplot
        info_ax.text(0.02, 0.5, robot_info, fontsize=18, va='center', ha='left',
                     bbox=dict(boxstyle="round", facecolor="lightgrey", alpha=0.5))
    
        # Create the legend and put it in the downright subplot
        legend_patches = [
            mpatches.Patch(color='white', label='Empty Space'),
            mpatches.Patch(color='black', label='Robot'),
            mpatches.Patch(color='grey', label='Stall'),
            mpatches.Patch(color="#FF00FF", label='Client')
        ]
        legend_ax.legend(handles=legend_patches, loc='center', fontsize=18)
    
        # Save plot in list of plots
        self.store.list_graphics.append(fig)
    
        plt.close(fig)


    # Function to calculate the shortest path between two places using Dijkstra
    ## Excluding blocked cells (boxes, robots, shelves)
    def calculate_shortest_path(self, start, goal, box, exclude=None):
        
        # Function to get the neighbors of a cell (so that the robots can jump into)
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            
            # Not diagonal movements
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            
                nx, ny = x + dx, y + dy
                
                # Check limits and take into account that it can not be either a blocked cell or a cell in the exclusion list
                
                # In this version of the problem, we must differentiate between before and after the merging
                if self.env.now < self.time_merge:
                    # Again, two cases: west and east
                    # West
                    if self.num < (self.store.num_robots / 2):
                        
                        if 0 <= nx < self.store.height_store and 0 <= ny < self.store.screen:
                            
                            # The neighbor value mustn't be a stall (except it's its own), a wall, or another box
                            if (self.store.matrix[nx, ny] in [100, 101, box] or np.array_equal(np.array([nx, ny]), self.pos_its_stall)) and (exclude is None or not any(np.array_equal((nx, ny), arr) for arr in exclude)):
                                neighbors.append((nx, ny))
                    
                    # East
                    else:
                        
                        if 0 <= nx < self.store.height_store and self.store.screen <= ny < self.store.length_store:
                            
                            # The neighbor value mustn't be a stall (except it's its own), a wall, or another box
                            if (self.store.matrix[nx, ny] in [100, 101, box] or np.array_equal(np.array([nx, ny]), self.pos_its_stall)) and (exclude is None or not any(np.array_equal((nx, ny), arr) for arr in exclude)):
                                neighbors.append((nx, ny))
                        
                else:
                    
                    if 0 <= nx < self.store.height_store and 0 <= ny < self.store.length_store:
                        
                        # The neighbor value mustn't be a stall (except it's its own), a wall, or another box
                        if (self.store.matrix[nx, ny] in [100, 101, box] or np.array_equal(np.array([nx, ny]), self.pos_its_stall)) and (exclude is None or not any(np.array_equal((nx, ny), arr) for arr in exclude)):
                            neighbors.append((nx, ny))
                        
            return neighbors

        # Dijkstra
        start = tuple(start)
        goal = tuple(goal)
        #print("start: ",start, ", goal: ", goal)
        
        queue = [(0, start)]  # (cost, position)
        distances = {start: 0}
        previous_nodes = {start: None}
        
        while queue:
            current_distance, current_node = heapq.heappop(queue)
            if current_node == goal:
                break  # Goal is reached
            
            neighbors_current_node = get_neighbors(current_node)
            
            # If a node doesn't have neighbors, it's because it is surrounded by boxes and excluded elements
            if len(neighbors_current_node) == 0:
                neighbors_current_node.append(exclude[0])
            
            for neighbor in neighbors_current_node:
                new_distance = current_distance + 1
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(queue, (new_distance, neighbor))

        # Rebuild the path
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = previous_nodes[node]
        path.reverse()
        #print("Path from ", start," to ", goal, ": ",path)
        # If we are looking for a box, we exclude both the first element since it is its current position and the last one since it's the position of the box itself
        if box is not None:
            return path[1:-1]
        # If we are going back to stall, just exclude first element
        else:
            return path[1:]
    
    # Function that calculates the total distance for a given order of visiting boxes
    def calculate_total_distance(self, order):
        
        total_distance = 0
        current_pos = tuple(self.pos)
        # For each box, calculate the distance between the current box/pos and the following one
        for box in order:
            box_pos = next((key for key, value in self.store.positions_boxes.items() if value == box), None)
            #print(box, box_pos, type(box_pos))
            path = self.calculate_shortest_path(current_pos, box_pos, box)
            total_distance += len(path) - 1
            current_pos = box_pos
        # Work out distance form last visited box to initial position (=stall)
        path_back = self.calculate_shortest_path(current_pos, tuple(self.pos_its_stall), None)
        total_distance += len(path_back) - 1
        #print("Total_distance: ", total_distance)
        return total_distance

    # Function that finds the optimal order of visiting boxes to minimize total distance
    def find_optimal_order(self, possible_client_requests):
        
        best_order = None
        min_distance = float('inf')
        # For each possible order, calculate total distance
        for order in possible_client_requests:
            print("Possible order: ",order, type(order))
            distance = self.calculate_total_distance(order)
            if distance < min_distance:
                min_distance = distance
                best_order = order
        # We only keep the order with the lowest distance traveled
        print("best order: ", best_order)
        return list(best_order)
    
    # Funciton that searches the optimal_order_boxes dictionary for a key that contains all elements of input_tuple,
    ## We iterate in reverse order (we want max number of elements), and return the optimal route based on its associated value.
    def find_partial_route(self, input_tuple):
        
        # Iterate through the dictionary in reverse order
        for key in reversed(self.optimal_order_boxes.keys()):
            # Check if all elements of input_tuple are in the key
            if all(elem in key for elem in input_tuple):
                # Get the value tuple associated with the found key
                value_tuple = self.optimal_order_boxes[key]
                
                # Filter the elements of value_tuple to include only those in input_tuple
                ordered_route = tuple(elem for elem in value_tuple if (elem % int(self.store.num_boxes/2)) in input_tuple)
                
                return ordered_route
        print("No partial_route found")
        
        # If no match is found, return None or an empty tuple
        return None

    # Function that finds key in dict of optimal_order_boxes so that it is the biggest, return order of related boxes
    ## Again in decreasing order
    def find_biggest_subset(self, new_command):
        for length in range(len(new_command), 1, -1):
            guide = next(
                (subset for subset in combinations(new_command, length) if subset in self.optimal_order_boxes),
                None
            )
            if guide:
                return self.optimal_order_boxes[guide]
        
        return None
    
    # Function that creates intelligent permutations to try as possible optimal routes for a certain command
    ## The elements of base must be in the same order in complete
    def generate_intelligent_combinations(self, complete, base):

        # Generate all possible permutations 
        all_permutations = permutations(complete)

        combinations = []

        for combination in all_permutations:
            print("Permutation: ",combination)
            invalid = False
            
            # We must check that the elements of the base are in that order
            for i in range(1, len(base)):
                print("self.store.box_item[base[i-1]]: ",self.store.box_item[base[i-1]], " vs. self.store.box_item[base[i]]: ",self.store.box_item[base[i]])
                if combination.index(self.store.box_item[base[i-1]]) > combination.index(self.store.box_item[base[i]]):
                    invalid = True
                    break
            
            if not invalid:
                combinations.append(combination)

        return combinations
    
    
    # Function that makes robots share information of what they know (optimal_routes, optimal orders and number of movements for fulfilling a command)
    def check_neighbors_update_info(self):
        
        def share_info(robot_num):
            
            other_robot = self.store.dict_robots[robot_num]
            """
            print(self.num,"=self.dict_orders_freq before merging", self.dict_orders_freq)
            print(robot_num,"=other_robot.dict_orders_freq before merging", other_robot.dict_orders_freq)
            print(self.num,"=self.dict_orders_movements before merging", self.dict_orders_movements)
            print(robot_num,"=other_robot.dict_orders_movements before merging", other_robot.dict_orders_movements)
            """
            
            # For each command
            for command in self.dict_orders_freq:
            
                # Convert lists to NumPy arrays once
                freq_self = np.array(self.dict_orders_freq[command])
                freq_other = np.array(other_robot.dict_orders_freq[command])
                
                # Calculate maximum of each pair of elements
                updated_array = np.maximum(freq_self, freq_other)
                
                # Create a mask to determine which elements were taken from self
                mask = freq_self >= freq_other
                
                # Select movements based on the mask
                updated_array_b = np.where(mask, np.array(self.dict_orders_movements[command]), np.array(other_robot.dict_orders_movements[command]))
                
                # Update info of movements for commands
                self.dict_orders_movements[command] = updated_array_b.tolist()
                other_robot.dict_orders_movements[command] = updated_array_b.tolist()
                
                # Update info of number of commands
                self.dict_orders_freq[command] = updated_array.tolist()
                other_robot.dict_orders_freq[command] = updated_array.tolist()
                
                # Now update if necessary the optimal order
                # If both robots have info about an optimal route for a certain command, they don't update
                if command in self.optimal_order_boxes and command not in other_robot.optimal_order_boxes:
                    other_robot.optimal_order_boxes[command] = self.optimal_order_boxes[command]
                elif command in other_robot.optimal_order_boxes and command not in self.optimal_order_boxes:
                    self.optimal_order_boxes[command] = other_robot.optimal_order_boxes[command]
                
            # Update update_info lists of both robots
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
        # After seeing that it is not moving, it checks if there are other robots next to it
        # Not diagonal movements
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            
            # Check limits and see if there's another robot
            
            # In this version of the problem, we must differentiate between before and after the merging
            if self.env.now < self.time_merge:
                # Again, two cases: west and east
                # West
                if self.num < (self.store.num_robots / 2):
                    
                    if 0 <= nx < self.store.height_store and 0 <= ny < self.store.screen and self.store.matrix[nx, ny] == 101:
                        
                        # Detect which robot it is
                        robot_num = next((k for k, v in self.store.positions_robots.items() if np.array_equal(v, np.array([nx, ny]))), None)
                        
                        # If it doesn't have the updated info of both robots
                        if self.updated_info[robot_num] == 0:
                            
                            share_info(robot_num)
                        
                # East
                else:
                    
                    if 0 <= nx < self.store.height_store and self.store.screen <= ny < self.store.length_store and self.store.matrix[nx, ny] == 101:
                        
                        # Detect which robot it is
                        robot_num = next((k for k, v in self.store.positions_robots.items() if np.array_equal(v, np.array([nx, ny]))), None)
                        
                        # If it doesn't have the updated info ob both robots
                        if self.updated_info[robot_num] == 0:
                            
                            share_info(robot_num)
                        
                    
            else:
                
                if 0 <= nx < self.store.height_store and 0 <= ny < self.store.length_store and self.store.matrix[nx, ny] == 101:
                    
                    # Detect which robot it is
                    robot_num = next((k for k, v in self.store.positions_robots.items() if np.array_equal(v, np.array([nx, ny]))), None)
                    
                    # If it doesn't have the updated info ob both robots
                    if self.updated_info[robot_num] == 0:
                        
                        share_info(robot_num)
                        
    # Functions related to Association rules
    
    # Function to create the list of transactions from the dict_orders_freq and the set of different items
    def extractItemsFromList(self):
        list_transactions = []
        c1 = set()
        for key,value in self.dict_orders_freq.items():
            for i in range(sum(value)):
                list_transactions.append(list(key))
            for word in key:
                c1.add(tuple([word]))
        return list_transactions, c1 
    
    
    #  Function that returns the frequent itemsets
    
    def apriori(self, minSupport):
        
        # ------------------------------------------------------------------------------------------
        
        # Function to create the set of itemsets with high support
        
        # globalSupports = dict()
        # itemsets = Ci (C1, C2, ...)
        # transactionList = [[1, 2, 3], [3, 5], [4], ...]
        # minSupport = a number between 0 and 1
        def higherSupports(itemSets, transactionList, minSupport, globalSupports):
            freqItemSets = set()
            localSupports = dict()
            
            # For each itemSet in itemSets and for each transaction in transactionList
            # if the itemSet appears in the transaction, increment the value in 
            # localSupports and globalSupports dictionaries for the corresponding key
            for itemSet in itemSets:
                for transaction in transactionList:
        
                    # My helper
                    complete = True
        
                    # If any item is missing, skip
                    for item in itemSet:
                        if item not in transaction: 
                            complete = False
                    
                    if complete: 
                        #print(itemSet, " is in: ", transaction)
                        
                        if itemSet in globalSupports:
                            globalSupports[itemSet] += 1
                        else:
                            globalSupports[itemSet] = 1
        
                        if itemSet in localSupports:
                            localSupports[itemSet] += 1
                        else:
                            localSupports[itemSet] = 1
        
                    #else:
                        #print(itemSet, " is not in: ", transaction)
                        
            # For each key-value pair in localSupports, calculate the support of the itemSet (key)
            # if the support is greater than or equal to minSupport, add the itemSet to the set of frequent itemSets
            for itemSet in localSupports:
                support = localSupports[itemSet] / len(transactionList)
        
                if support >= minSupport:
                    freqItemSets.add(itemSet)
            
            return freqItemSets
        
            # return Li
        
        # ------------------------------------------------------------------------------------------
        
        # Function to create the set of candidate itemsets of length i+1
        
        # L = Li
        # k = i+1
        
        def union(L, k):
            unionSet = set()
            # For each pair of elements, one from L and the other from its "copy"
            for element in L:
                for elementCopy in L:
                    # If they are not the same
                    if element != elementCopy:
                        # Create the possible ordered tuple without repetitions
                        possibleSet = tuple(sorted(set(element + elementCopy)))
                        # If it also has the desired length
                        if len(possibleSet) <= k:
                            unionSet.add(possibleSet) 
            return unionSet
        
            # return Ci+1
            
        # ------------------------------------------------------------------------------------------
        
        # Function to eliminate those itemsets that have a non-frequent itemset of length i-1
        
        # candidates = Ci
        # prevFrequent = Li-1
        # length = i
        
        def prune(candidates, prevFrequent, length):
            candidatesToReturn = candidates.copy()
            # For each itemSet in candidates, calculate all its (k-1)-subsets
            # If any of those subsets is not in prevFrequent, remove the itemSet 
            # from candidatesToReturn (using the remove function).
            # As soon as we find a subset that is not in prevFrequent, we can skip 
            # to the next itemSet (break).
            for itemSet in candidates:
                lowerKSubsets = combinations(itemSet, length-1)
                for subset in lowerKSubsets:
                    if subset not in prevFrequent:
                        candidatesToReturn.remove(itemSet)
                        break
            return candidatesToReturn
        
        # ------------------------------------------------------------------------------------------
        
        # Initialize the dictionaries to be returned
        globalFrequents = dict()
        globalSupports = dict()
        
        # Calculate the 1-itemsets from the transaction list and extract the frequent ones
        transactionList, C1 = self.extractItemsFromList()
        L1 = higherSupports(C1, transactionList, minSupport, globalSupports)
        
        # Prepare for iteration. Store L1 in currentL and initialize k=2
        currentL = L1
        k = 2
        # Iterate. While currentL contains itemsets, store currentL in globalFrequents,
        # generate the next candidates using union and prune, and update currentL 
        # with the next set of frequent itemsets. Finally, increment k by 1.
        while len(currentL) > 0:
            globalFrequents[k-1] = currentL
    
            C = union(currentL, k)
            C = prune(C, currentL, k)
    
            currentL = higherSupports(C, transactionList, minSupport, globalSupports)
            k += 1
    
        return globalFrequents, globalSupports, transactionList
    
    # --------------------------------------------------------------------------------------------------
    
    # Function to create the valid association rules
    
    # transactionList = [[1, 2, 3], [3, 5], [4], ...]
    # frequentItemSets = globalFrecuentes of before
    # itemSetSupports = globalSoportes of before
    # minConfidence = number between 0/1
    # lift = True/False
    
    def associationRules(self, transactionList, frequentItemSets, itemSetSupports, minConfidence, lift=False):
        rules = []
        # For each itemset, calculate all its possible subsets
        # For each subset S, calculate the confidence of the rule S -> (itemset - S)
        # If the confidence exceeds the minConfidence threshold, add the rule to rules
    
        # frequentItemSets is a dictionary of the form {1: frequent itemsets of size 1, 2: size 2, ...}
        for itemSetLength in frequentItemSets:
            
            itemSets_k = frequentItemSets[itemSetLength]
    
            # For each itemSet:
            for itemSet in itemSets_k:
    
                # Generate all subsets of all possible sizes
                subsets = chain.from_iterable(combinations(itemSet, k) for k in range(1, itemSetLength))
    
                # For each possible subset
                for subset in subsets: 
    
                    # Define the antecedent and the consequent (the consequent must remain ordered and without repeated elements)
                    antecedent = subset
                    consequent = tuple(sorted(set(itemSet) - set(subset)))
    
                    # Calculate confidence
                    supportIS = itemSetSupports[itemSet] / len(transactionList)
                    supportAN = itemSetSupports[antecedent] / len(transactionList)
                    confidence = supportIS / supportAN
    
                    # If confidence exceeds the minimum threshold:
                    if confidence >= minConfidence: 
    
                        # If lift is true, calculate it
                        if lift: 
                            
                            liftValue = supportIS / (supportAN * (itemSetSupports[consequent] / len(transactionList)))
    
                            # Add the rule to rules (antecedent and consequent as sets, per the problem statement)
                            if liftValue >= 1:
                                rules.append([set(antecedent), set(consequent), confidence, liftValue])
                                
                        else:
                            rules.append([set(antecedent), set(consequent), confidence])
                    
        return rules
    
        # return [[setAntecedent, setConsequent, confidence, lift (if True)] of rule 1, [""] of rule 2, ...]
        
    # Function adjusted sigmoid (to work out probability of creating association rules)
    def sigmoid(self, x, k=0.3, x0 = 30):
        return 1 / (1 + np.exp(-k * (x - x0)))
    
    # Function to calculate all the possible permutations if the robot doesn't know any optimal route
    
    def calculate_permutations_boxes_visit(self, command):
        # Differentiate between before/after merging
        if self.env.now < self.time_merge:
            
            # West
            if self.num < (self.store.num_robots/2):
              
                    return list(permutations(command))
                
            # East: Add to each element of command self.store.num_boxes/2 units
            else:
                command_bis = []
                for element in command:
                    command_bis.append(element + int(self.store.num_boxes/2))
                return list(permutations(tuple(command_bis)))
                
        else:
            
            # Firstly, obtain the natural combinations
            total_combinations = list(permutations(command))
            raw = copy.deepcopy(total_combinations)
            # Then, for each element of command, get same combinations but changing that element (representing the box)
            ## with the other one possible
            for element in command:
                     
                for permutation in raw:
                    
                    permutation = list(permutation)
                    
                    for i in range(len(permutation)):
                        
                        if permutation[i] == element:
                            
                            permutation[i] = element + int(self.store.num_boxes/2)
                
                total_combinations.append(tuple(permutation))
            
            return total_combinations
                

    # --------------------------------------------------------------------------------------------------
    
    def run(self):
        
        while True:
            
            # Wait for client to arrive
            info = yield self.store.waiting_line.get()
                
            # Get its info
            id_client = info[0]
            command = info[1]
            print("Robot: ",self.num, ", command: ",command)
            
            with self.traffic_light_stall.request() as req:
                yield req  # So that the client doesn't move until he's served
                
                # Indicate that a client is being served 
                self.store.matrix[self.pos_its_client[0], self.pos_its_client[1]] = 103
                
                # Add elements of command in dictionary of items to visit
                self.store.robot_items_not_visited[self.num] = set(command)
            
                # It has awaken up and has its command, it needs to get the required items taking into account the information it has
                # Firstly, it must calculate the optimal order (order of BOXES to follow)
                
                command_ordered = tuple(sorted(list(command)))
                
                # If it doesn't have any information about optimal routes or it must consider all options
                if len(self.optimal_order_boxes) == 0 or (command_ordered in self.list_routes_before_merging and self.env.now > self.time_merge):
                    print("1098 Robot ",self.num, "doesn't have info about optimal_routes or it needs to consider all boxes")
                    optimal_order = self.find_optimal_order(self.calculate_permutations_boxes_visit(command))
                
                # We must take into account if it has done previously that very command
                elif command_ordered in self.optimal_order_boxes:
                    
                    print("1104 Robot",self.num," has done this command before and has its info updated")
                    optimal_order = self.optimal_order_boxes[command_ordered]
                    
                    
                # Another possibility is that there's an optimal route that part of it is the command we must attend
                else:
                    optimal_order = self.find_partial_route(command_ordered)
                    print("1111 Robot ",self.num," Find partial route - Optimal_route: ", optimal_order)
                    # If not, we must see which is the optimal route but taking into account the information we already know
                    # What we do is to take the biggest subset of the command that we already know its best order and consider those combinations that include
                    ## all the elements of the command but those in the subset are in that order
                    # Then we look for the optimal order but only with those combinations
                    if optimal_order is None:
                        print("1117 Robot ",self.num, "intelligent combinations")
                        base = self.find_biggest_subset(command_ordered)
                        
                        # If it can base part of its new route on some previous knowledge
                        if base is not None:
                            print("Base: ",base)
                            intelligent_combinations = self.generate_intelligent_combinations(command, base)
                            optimal_order = self.find_optimal_order(intelligent_combinations)
                            
                        else:
                            optimal_order = self.find_optimal_order(self.calculate_permutations_boxes_visit(command))
                        
                print("1129: Robot ",self.num," Optimal Order for getting ", command, " is :", optimal_order)
                
                # Add to dictionary of optimal routes
                self.optimal_order_boxes[command_ordered] = optimal_order
                
                # If after merging we obtain a route that had to be optimised again, remove it from the list of routes before merging 
                if self.env.now > self.time_merge and command_ordered in self.list_routes_before_merging:
                    self.list_routes_before_merging.remove(command_ordered)
                
                # Create two lists that will contain the boxes/items that the robots has already visited
                visited_boxes = []
                visited_items = []
                
                # Start list of occupied_pos
                occupied_pos = []
                
                # Initialize number of steps to fulfill command
                number_steps = 0
                
                # Until it doesn't visit all the boxes
                while set(visited_items) != set(command):
                    
                    # Create a boolean to indicate if a robot must recalculate its route to a specific box
                    boolean_recalculate_route = False
                
                    # Follow steps of the optimal order
                    for i,box in enumerate(optimal_order):
                        
                        if box not in visited_boxes:
                        
                            # Calculate shortest path to the box
                            box_pos = next((key for key, value in self.store.positions_boxes.items() if value == box), None)
                            route = self.calculate_shortest_path(self.pos, box_pos, box, exclude = occupied_pos)
                            
                            print("Robot ",self.num, ", best route to go from ",self.pos, "to ",box_pos," to get ",box,": ",route)
                            
                            # It may occur that the robot after getting the item from its box
                            ## Can also reach the following object
                            if len(route) == 0:
                                
                                with self.traffic_light_move.request() as req:
                                    
                                    yield req  # Wait to get the traffic light to move
                                
                                    print("Robot ",self.num, ", is already in ",self.pos, " to get ",box, ", len(route): ",len(route))
                                    
                                    # Update info if needed
                                    self.check_neighbors_update_info()
                                    
                                    print("Robot ",self.num, " found box ",box, " it's looking for at its position: ",box_pos)
                                    visited_boxes.append(box)
                                    related_item = self.store.box_item[box]
                                    visited_items.append(related_item)
                                    self.store.robot_items_not_visited[self.num] = self.store.robot_items_not_visited[self.num] - set([related_item])
                                    print("Robot ",self.num," must still go for ",len(optimal_order[i:]) - 1," boxes, visited_boxes: ",visited_boxes, "while condition: ",set(visited_boxes)==set(command))
                                      
                            # If it has to move        
                            else:
                                
                                # Follow the route
                                for step in route:
                                    
                                    print(self.env.now, self.num, ", step: ",step, ", box: ", box)
                                    
                                    with self.traffic_light_move.request() as req:
                                        
                                        yield req  # Wait to get the traffic light to move
                                        
                                        # Check if the cell it wants to move in is empty or not
                                        # print(self.num, step, self.store.matrix[step[0], step[1]])
                                        
                                        if self.store.matrix[step[0], step[1]] == 100:  # 100 means "empty"
                                        
                                            # Move to this position
                                            
                                            # If last cell wasn't a stoll, put an empty cell
                                            if not any(np.array_equal(np.array([self.pos[0], self.pos[1]]), value) for value in self.store.positions_stalls.values()):
        
                                                self.store.matrix[self.pos[0], self.pos[1]] = 100  # Where it was -> 100 (empty)
                                            
                                            # If it was its own stall, put a stall cell
                                            ## It could have been an "else" statement, however, just in case, we check it was its own stall
                                            elif any(np.array_equal(np.array([self.pos[0], self.pos[1]]), value) for value in self.store.positions_stalls.values()):
                                                
                                                self.store.matrix[self.pos[0], self.pos[1]] = 102  # Where it was -> 102 (stall)
                                                
                                            self.store.matrix[step[0], step[1]] = 101 # 101 -> Robot
                                            
                                            self.pos = np.array(step)
                                            
                                            # print("Robot ",self.num," has moved to: ",self.pos, ". Proof: ",self.store.matrix[self.pos[0], self.pos[1]])
                                            
                                            # Record current situation with update of the situation of the store
                                            self.store.positions_robots[self.num] = self.pos
                                            self.graphic()
                                            
                                            # Update number of steps 
                                            number_steps = number_steps + 1
                                            
                                            # Restart list of occupied_pos
                                            occupied_pos = []
                                            
                                            # Before checking boxes next to it, update info if needed
                                            self.check_neighbors_update_info()
                                         
                                            # The theoretical place of the box we are looking for is in the list of close boxes' locations: 
                                            if step == route[-1]:
                                                
                                                print("Robot ",self.num, " found box ",box, " it's looking for at its position: ",box_pos)
                                                visited_boxes.append(box)
                                                related_item = self.store.box_item[box]
                                                visited_items.append(related_item)
                                                self.store.robot_items_not_visited[self.num] = self.store.robot_items_not_visited[self.num] - set([related_item])
                                                #print("Robot ",self.num," with boxes",optimal_order,", visited_boxes: ",visited_boxes)
                                                    
                                                
                                        else:
                                            # If cell is not empty, recalculate route
                                            if (step[0], step[1]) not in occupied_pos:
                                                occupied_pos.extend([(step[0], step[1])])
                                            print(f"Robot {self.num} found {step} occupied while looking for box {box}. Recalculating route... occupied_pos: ",occupied_pos)
                                            #route = self.calculate_shortest_path(self.pos, box_pos, box, exclude=occupied_pos)
                                            #print("Robot ",self.num, ", NEW best route to go from ",self.pos, "to ",box_pos," to get ",box,": ",route)
                                                
                                            boolean_recalculate_route = True
                                            
                                            # Before recalculating route, update info if needed
                                            self.check_neighbors_update_info()
                                            
                                            break
                                            
                                    #print(f"Robot {self.num} waits 0.5 sec until trying to move again")
                                    yield self.env.timeout(0.5)  # Wait until it tries to move again
                                    
                                    
                                # If the robots have changed the order of things to visit 
                                if boolean_recalculate_route:
                                    
                                    print(f"Robot {self.num} changed the route to get a specific box (inner loop)")
                                        
                                    yield self.env.timeout(0.5)  # Time to move
                                    
                                    break
                                
                                yield self.env.timeout(0.5)  # MAYBE NOT NECESSARY
                                
                            
                    
                #  We have picked all the products for the customer, go back to initial position
                yield self.env.timeout(0.5)
                print(f"Robot {self.num} starts its way back to stall from ",self.pos, " to ",self.pos_its_stall)
                route_back = self.calculate_shortest_path(self.pos, tuple(self.pos_its_stall), None)
                
                arrived_to_stall = False
                
                # start occupied_pos list
                occupied_pos = []
                
                while not arrived_to_stall:
                    
                    #print(self.num, ", route back: ",route_back)
                    
                    for step in route_back:
                        
                        #print(self.env.now, self.num, ", step: ",step, " going back")
                        
                        with self.traffic_light_move.request() as req:
                            yield req
                            
                            # If the next step is empty
                            if self.store.matrix[step[0], step[1]] == 100:  # 100 means "empty"
                                # Move to this position
                                self.store.matrix[self.pos[0], self.pos[1]] = 100  # Where it was -> 100 (empty)
                                self.pos = np.array(step)
                                self.store.matrix[self.pos[0], self.pos[1]] = 101  # 101 -> Robot
                                
                                # Create graphic and update situation of the store
                                self.store.positions_robots[self.num] = self.pos
                                self.graphic()
                                
                                # Update number of steps 
                                number_steps = number_steps + 1
                                
                                # Restart occupied_pos list
                                occupied_pos = []
                            
                            # If next step is its own stall
                            elif np.array_equal(np.array([step[0], step[1]]), self.pos_its_stall): 
                                print(f"Robot {self.num} arrived to its stall, completed client {id_client} order")
                                arrived_to_stall = True
                                
                                # Update movement (we don't put that the stall in the matrix is 101)
                                self.store.matrix[self.pos[0], self.pos[1]] = 100  # Where it was -> 100 (empty)
                                self.pos = np.array(step)
                                
                                # Create graphic and update situation of the store
                                self.store.positions_robots[self.num] = self.pos
                                self.graphic()
                                
                                # Restart occupied_pos list
                                occupied_pos = []
                                
                                break
                                
                            else:
                                # If cell is not empty, recalculate route
                                occupied_pos.extend([(step[0], step[1])])
                                #print(f"Robot {self.num} found {step} occupied while going to stall. Recalculating route... occupied_pos: ",occupied_pos)
                                route_back = self.calculate_shortest_path(self.pos, tuple(self.pos_its_stall), None, exclude=occupied_pos)
                                #print("Robot ",self.num, ", NEW best route to go home: ",route_back)
                                
                                break
                        yield self.env.timeout(1)  
                        
               
                # The robot updates its own number of number of movements
                self.dict_orders_movements[command][self.num] = (self.dict_orders_movements[command][self.num] * self.dict_orders_freq[command][self.num] + number_steps)/(self.dict_orders_freq[command][self.num] + 1)
                """
                print("-----------------------------------------------------------------------------------------------")
                print(self.num, ", dict_orders_movements[ command:",command,"][self.num:",self.num,"]: ",self.dict_orders_movements[command][self.num])
                print("-----------------------------------------------------------------------------------------------")
                """
                # The robot updates its own number of commands fulfilled of that specific type of command
                self.dict_orders_freq[command][self.num] = self.dict_orders_freq[command][self.num] + 1
                """
                print("-----------------------------------------------------------------------------------------------")
                print(self.num, ", dict_orders_freq[ command:",command,"][self.num:",self.num,"]: ",self.dict_orders_freq[command][self.num])
                print("-----------------------------------------------------------------------------------------------")
                """
                # Update update_info list
                self.updated_info = [0]*self.store.num_robots
                self.updated_info[self.num] = -1          
                
                # The client leaves 
                before = self.total_commands_served
                self.total_commands_served = np.sum([np.array(mylist) for mylist in self.dict_orders_freq.values()])
                
                # If total_commands_served has surpassed the following multiple of module, restart the boolean so that it can generate new association rules
                if (before % self.module_for_association) > (self.total_commands_served % self.module_for_association):
                    self.recent_creation_rules = False
                
                self.store.matrix[self.pos_its_client[0], self.pos_its_client[1]] = 100
                
                # See if work out rules of association
                if random.random() < self.sigmoid(self.total_commands_served  % self.module_for_association) and not self.recent_creation_rules:
                    
                    print("Robot ",self.num, "works out association rules")
                    
                    self.recent_creation_rules = True
                    
                    # Work out association rules
                    
                    frequentItemSets, itemSetSupports, transactionList = self.apriori(minSupport = 0.1)
                    
                    print("frequentItemSets: ",frequentItemSets, ", itemSetSupports: ",itemSetSupports)
                    
                    rules = self.associationRules(transactionList, frequentItemSets, itemSetSupports, minConfidence = 0.7, lift = True)
                    
                    print("rules: ",rules)
                    
                    
                yield self.env.timeout(1)



class Client_Generator(object):
    def __init__(self, env, store, arrival_rate, time_merge):
        self.env = env
        self.store = store
        self.arrival_rate = arrival_rate
        self.time_merge = time_merge
        # For counting clients
        self.cliente_id = 1
        
        
    def run(self):
        
        
        while True:
            # Generate the random arrival time between clients
            time_between = np.random.exponential(scale=1/self.arrival_rate)
            
            yield self.env.timeout(time_between)
            
            # It is not necessary to differentiate between before and after the merging because the requests will be the same
            
            # We have the Client ID, we create its command
            client_command = random.choices(
                population=list(self.store.prob_itemsets.keys()),
                weights=list(self.store.prob_itemsets.values()),
                k=1)[0]
            
            # We put it to wait
            yield self.store.waiting_line.put([self.cliente_id, client_command])
                
            #print(f"{self.env.now}: Client {self.cliente_id} waiting in queue")
            self.cliente_id += 1
    
# ------------------------------------------------------------------------------------------------------- EXECUTION OF THE ALGORITHM -----------------------------------------------------------------------------------------------------
#env = simpy.rt.RealtimeEnvironment(factor=0.5)
env = simpy.Environment()
traffic_light_move = simpy.Resource(env, 1)

# If possible, num_boxes >= 8
    # The total number of different articles will be its half
num_boxes, num_robots = 18, 12

# Maximum number of items in the order
max_items_order = 3

if num_boxes % 2 != 0:
    num_boxes = num_boxes + 1

if num_robots % 2 != 0:
    num_robots = num_robots + 1

if max_items_order > num_boxes/2:
    max_items_order = num_boxes/2
    
time_merge = 10

# Initialize the store
store = Store(num_boxes, num_robots, max_items_order, env)

# Initialize the robots
dict_robots = dict()
for i in range(num_robots): 
    robot = Robot(env, i, store, traffic_light_move, time_merge)
    dict_robots[i] = robot
    env.process(robot.run())

store.dict_robots = dict_robots
    
# Initialize the client generator (assume that the time between arrivals follows an Exponential Distribution because the arrival process is a Poisson Process)
# Lambda = Arrival rate
lmbda = 5 
generator = Client_Generator(env, store, lmbda, time_merge)
env.process(generator.run())

env.run(until = 20)

# Crear el GIF
output_path = "./simulacion_screen.gif"

images = []

for fig in store.list_graphics:
    # Adjust margings automatically
    fig.tight_layout()
    # Save figure in a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)  # Go back to beginning of buffer
    # Convert image into format PIL before closing buffer
    image = Image.open(buf).copy()  # Copy image to avoid problems when closing buffer
    images.append(image)
    buf.close()  # Close buffer after the copy

images[0].save(
    output_path, 
    save_all=True, 
    append_images=images[1:], 
    duration=250,  # Duration of each frame in miliseconds 
    loop=0  # 0 for infinte loop 
)


# Perplexity AI -> Para mirar trabajos parecidos al mío, con cuidado: Lo estamos haciendo bien? Y si los hay, hacemos contribución de alguna forma? 
# podc, disc, dsn, srds -> Qué se ha publicado recientemente relacionado con lo mío (combinar con perplexity)
# Revistas: IEEETPDS, Computing (de ACM)
# Mirar cómo se escribe (el estilo)
