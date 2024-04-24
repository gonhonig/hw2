import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    S_counts = np.unique(data[:, -1], return_counts=True)[1]
    S = S_counts / data.shape[0]
    gini = 1 - S.dot(S)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    entropy = calc_feature_entropy(data, -1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

def calc_feature_entropy(data, feature):
    S_counts = np.unique(data[:, feature], return_counts=True)[1]
    P = S_counts / data.shape[0]
    return -P.dot(np.log2(P))

class DecisionNode:
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the node
        self.pred = self.calc_node_pred() # the prediction of the node
        self.feature = feature
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.chi = chi
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.impurity = impurity_func(data)
        self.terminal = self.data.shape[1] < 2 or self.depth >= self.max_depth # determines if the node is a leaf
        self.entropy = calc_entropy(data)
        self.feature_importance = 0
        self.n_total = data.shape[0]


    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        values, counts = np.unique(self.data[:, -1], return_counts=True)
        pred = values[np.argmax(counts)]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        self.children.append(node)
        self.children_values.append(val)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        self.feature_importance = (self.data.shape[0] / n_total_sample) * self.goodness_of_split(self.feature)[0]

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        X = self.data
        groups = {v: X[X[:, feature] == v] for v in np.unique(X[:, feature])}
        S = np.array([np.shape(g)[0] for g in groups.values()]) / X.shape[0]

        if self.gain_ratio:
            information_gain = self.entropy - S.dot(np.array(list(map(calc_entropy, groups.values()))))
            split_information = calc_feature_entropy(self.data, feature)
            split_information = split_information if split_information > 0 else 1
            goodness = information_gain / split_information
        else:
            goodness = self.impurity - S.dot(np.array(list(map(self.impurity_func, groups.values()))))

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return goodness, groups

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        if self.terminal:
            return

        best_feature, values = self.find_best_feature()
        self.feature = best_feature
        self.calc_feature_importance(self.n_total)

        if len(values) < 2:
            self.terminal = True
            return

        n_total_samples = self.data.shape[0]

        for value in values:
            child = DecisionNode(data=np.delete(values[value], best_feature, axis=1),
                                 impurity_func=self.impurity_func,
                                 depth=self.depth + 1,
                                 max_depth=self.max_depth,
                                 gain_ratio=self.gain_ratio)
            child.n_total = n_total_samples
            self.add_child(child, value)

        for child in self.children:
            child.split()

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


    def find_best_feature(self):
        features_goodness = {f: self.goodness_of_split(f) for f in range(self.data.shape[1] - 1)}
        best_feature = max(features_goodness, key=lambda key: features_goodness[key][0])

        return best_feature, features_goodness[best_feature][1]

    def log_node(self):
        if self.terminal:
            print("---- built leaf ----")
            print(f"prediction: {self.pred}")
        else:
            print("---- built node ----")
            print(f"feature: {self.feature}")
            print(f"feature importance: {self.feature_importance}")
            print(f"impurity: {self.impurity}")
            print(f"node depth: {self.depth}")
            print(f"no. of features: {self.data.shape[1] - 1}")
            print(f"no. of  children: {self.children.__len__()}")

        print("--------------------\n")


    def get_depth(self):
        if self.terminal or len(self.children) == 0:
            return 1
        else:
            return max(map(DecisionNode.get_depth, self.children)) + 1


class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        self.root = DecisionNode(data=self.data,
                            gain_ratio=self.gain_ratio,
                            depth=0,
                            max_depth=self.max_depth,
                            impurity_func=self.impurity_func)
        self.root.split()

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: a row vector from the dataset. Note that the last element
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        node = self.root

        while not node.terminal:
            feature = node.feature
            node = node.children[node.children_values == instance[feature]]
            instance = np.delete(instance, feature, axis=0)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        n = dataset.shape[0]
        predictions = [(self.predict(row), row[-1]) for row in dataset]
        correct = sum(1 for p in predictions if p[0] == p[1])
        accuracy =  correct / n

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy
        
    def depth(self):
        return self.root.get_depth()

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        root = DecisionTree(data=X_train, impurity_func=calc_entropy, max_depth=max_depth)
        root.build_tree()
        training.append(root.calc_accuracy(X_train))
        validation.append(root.calc_accuracy(X_validation))

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_testing_acc, depth


def count_nodes(node: DecisionNode):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    n_nodes = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    if len(node.children) == 0:
        return 1
    else:
        return sum(map(count_nodes, node.children)) + 1

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






