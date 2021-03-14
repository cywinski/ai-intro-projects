import random
from random_forest.decision_tree import DecisionTree

class RandomForest():
    
    def __init__(self, dataset, num_of_trees, max_depth, min_node_size, num_of_attributes):
        self.dataset = dataset
        self.num_of_trees = num_of_trees
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.num_of_attributes = num_of_attributes
        self.all_attributes = [i for i in range(len(self.dataset[0]) - 1)]
        self.trees = list()
        
        for i in range(num_of_trees):
            self.trees.append(self.create_tree())
         
    def bootstrap_data(self):
        return random.choices(self.dataset, k=len(self.dataset))
    
    def create_tree(self):
        data = self.bootstrap_data()
        return DecisionTree(data, self.max_depth, self.min_node_size, self.num_of_attributes, self.all_attributes)
        
    def predict(self, row):
        predictions = dict()
        for tree in self.trees:
            pred = tree.predict(row)
            if pred not in predictions.keys():
                predictions[pred] = 0
            predictions[pred] += 1
        return self.max_predict(predictions)
        
    def max_predict(self, predictions):
        maximal_pred = float('-inf')
        for key in predictions.keys():
            if predictions[key] > float(maximal_pred):
                maximal_pred = key
                
        return maximal_pred