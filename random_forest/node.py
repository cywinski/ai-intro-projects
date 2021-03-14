import random
from random_forest.gini_index import gini_index

class Node():
    
    def __init__(self, values, all_classes, num_of_attributes, all_attributes, depth, max_depth, min_size, is_terminal):
        self.values = values
        self.all_classes = all_classes
        self.num_of_attributes = num_of_attributes
        self.all_attributes = all_attributes
        # Use only a random subset of attributes of dataset
        self.possible_split_attributes = random.sample(self.all_attributes, k=self.num_of_attributes) 
        self.grp_count = self.count_grps()
        self.depth = depth
        self.max_depth = max_depth
        self.min_size = min_size
        self.left = None
        self.right = None
        self.gini_index = 1
        self.split_attribute = None
        self.split_value = None
        self.is_terminal = is_terminal
        self.pred_class = self.pred_class()
        if not self.is_terminal:
            self.pick_best_split()


    def count_grps(self):
        grps = [i[-1] for i in self.values]
        count = [grps.count(j) for j in self.all_classes]
        return count
    

    def pred_class(self):
        grps = [i[-1] for i in self.values]
        return max(set(grps), key=grps.count)
    

    def pick_best_split(self):
        tmp_left = None
        tmp_right = None
        for attribute in self.possible_split_attributes:
            for row in range(len(self.values)):
                lower, higher = self.split(attribute, row)
                gini = gini_index([lower, higher], self.all_classes)
                if gini < self.gini_index:
                    self.gini_index = gini
                    self.split_attribute = attribute
                    self.split_value = self.values[row][attribute]
                    tmp_left = lower
                    tmp_right = higher
                    
        self.make_new_node(tmp_left, tmp_right)
        

    def make_new_node(self, left, right):
        if not left or not right or self.gini_index == 0:
            # No more splits possible
            self.is_terminal = True
            return
        
        if len(self.values) <= self.min_size:
            self.is_terminal = True
            return
        
        if self.depth == self.max_depth:
            self.is_terminal = True
            return
        
        self.left = Node(left, self.all_classes, self.num_of_attributes, self.all_attributes, self.depth + 1, self.max_depth, self.min_size, False)
        self.right = Node(right, self.all_classes, self.num_of_attributes, self.all_attributes, self.depth + 1, self.max_depth, self.min_size, False)
        
        
    def split(self, attribute, row):
        lower = list()
        higher = list()
        for i in range(len(self.values)):
            if self.values[i][attribute] < self.values[row][attribute]:
                lower.append(self.values[i])
            else:
                higher.append(self.values[i])
        return lower, higher
   

    def __str__(self):
        return f"Split attribute:{self.split_attribute}, Gini:{self.gini_index}, Samples:{len(self.values)}, Groups:{self.grp_count}, Class:{self.pred_class}, Split value:{self.split_value}, Depth:{self.depth}"