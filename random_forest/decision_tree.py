from random_forest.node import Node
class DecisionTree():
    
    def __init__(self, dataset, max_depth, min_node_size, num_of_attributes, all_attributes):
        self.dataset = dataset
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.num_of_attributes = num_of_attributes
        self.all_attributes = all_attributes
        self.all_classes = list(set([i[-1] for i in dataset]))
        self.root = Node(self.dataset, self.all_classes, self.num_of_attributes, self.all_attributes, 1, max_depth, min_node_size, False)
        
    def visualize(self):
        queue = list()
        queue.append(self.root)
        while queue:
            v = queue.pop(0)
            print(v)
            if v.left is not None:
                queue.append(v.left)
            if v.right is not None:
                queue.append(v.right)
       
                
    def predict(self, row):
        current_node = self.root
        current_predicted_class = None
        while current_node is not None:
            current_predicted_class = current_node.pred_class
            if row[current_node.split_attribute] < current_node.split_value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return current_predicted_class