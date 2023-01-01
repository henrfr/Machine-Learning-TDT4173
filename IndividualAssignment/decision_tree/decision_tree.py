import numpy as np
import pandas as pd
import random
# import graphviz as gv # used for visualizations
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

class DecisionTree:

    def __init__(self, max_branches: int = 100, max_depth: int = 100):
        self.tree = "Tree is not fitted"
        # Tunes if a node with many branches can be chosen
        self.max_branches = max_branches
        # Tunes the depth of the tree
        self.max_depth = max_depth

    class Node:
        """Holds the tree logic. The name of a node corresponds to a column name, and the keys (edges) corresponds to the column entries."""
        def __init__(self, column_name):
            self.name = column_name
            self.children = {}
        
        def add_child(self, feature_class: str, child):
            """The child is a Node."""
            self.children[feature_class] = child

    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        def fit_dt_recursive(X: pd.DataFrame, y: pd.Series, prev_y: pd.Series, column_names: np.ndarray, original_length: int):
            """Fits the decision tree recursively by splitting on the column with maximum information gain.

            Args:
                X (pd.DataFrame): X from the fit method.
                y (pd.Series): y from the fit method.
                prev_y (pd.Series): The previous y in the recursion.
                column_names (np.ndarray<str>): An array holding the column names of X.
                original_length (int): Holds the originial length of the column names of X.

            Returns:
                Node: Represents a decision.
            """
            # If X is empty, choose the most frequent value of it's parent.
            if len(X.index) == 0:
                return self.Node(plurality(prev_y))
            # If all have the same class
            if is_same_class(y):
                return self.Node(y.head(1).values[0])
            # If the recursion reaches maximum depth
            if original_length - len(column_names) > self.max_depth - 1:
                return self.Node(plurality(y))
            # If there are no more columns to split upon
            if len(column_names) == 0:
                return self.Node(plurality(y))    
            
            column_names = X.columns
            split_column = max_information_gain(X, y, column_names, self.max_branches)
            node = self.Node(split_column)
            
            next_column_names = column_names[column_names != split_column]
            for column_class in X[split_column].unique():
                next_X = X[X[split_column] == column_class]
                next_X = next_X.drop(columns=[split_column])
                next_y = y.loc[next_X.index]
                child = fit_dt_recursive(next_X, next_y, y, next_column_names, original_length)
                node.add_child(column_class, child)
            return node

        column_names = X.columns
        self.tree = fit_dt_recursive(X, y, y, column_names, len(column_names))
    
    def predict(self, X: pd.DataFrame): 
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """

        def predict_inner(row):
            tree = self.tree
            while len(tree.children):
                # If there is little training data, and a key from the test data is not found, choose another branch.
                if tree.children.get(str(row[tree.name]), False):
                    tree = tree.children.get(str(row[tree.name]))
                else:
                    key = str(row[tree.name])
                    while key == str(row[tree.name]):
                        key = random.choice(list(tree.children.keys()))
                    tree = tree.children[key]
            return tree.name

        predictions = X.apply(predict_inner, axis = 1)
        return predictions.to_numpy()
    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        node = self.tree
        rules = []
        visited = []
        
        def dfs(node, visited: list, rules: list):
            """A depth-first search finding all possible paths to the leaf nodes

            Args:
                node (Node): The current Node
                visited (list): Keeps track of the current path
                rules (list): Keeps track of all rules.
            """
            visited.append(node.name)
            if not len(node.children):
                rules.append(tuple(visited))
                visited.pop()               
                return
            for key, child in node.children.items():
                if child.name not in visited:
                    visited[-1] = (node.name, key)
                    dfs(child, visited, rules)
            visited.pop()
        dfs(node, visited, rules)

        # Format the rules to comply with provided code
        post_process_rules = []
        for rule in rules:
            post_process_rules.append((list(rule[:-1]), rule[-1]))
        return post_process_rules
    
    # Uses graphviz, uncomment to display trees.
    # def display(self, leaf_class: dict, filename: str = "dt.gv"):

    #     def generate_tree(tree, prev_node, G):
    #         if isinstance(tree, DecisionTree.Node): # This will stop the recursion at the leaf nodes.
    #             for edge, value in tree.children.items():
    #                 if not bool(value.children): # If it is a leaf node, indicate the class
    #                     prev_name = prev_node + value.name + str(edge)
    #                     G.node(name=prev_name, label=f"Class: {value.name}",style='filled', fillcolor=leaf_class[value.name])
    #                     G.edge(prev_node, prev_name, f"Value: {edge}")
    #                     generate_tree(value, prev_name, G)
    #                 elif isinstance(value, DecisionTree.Node): # This block applies to all nodes not being top or bottom nodes.
    #                     prev_name = prev_node + value.name + str(edge)
    #                     G.node(name=prev_name, label=f"Split on: {value.name}")
    #                     G.edge(prev_node, prev_name, f"Value: {edge}")
    #                     generate_tree(value, prev_name, G)

    #     G = gv.Digraph("G", filename=filename)
    #     G.node(name="root", label = f"Split on: {self.tree.name}")
    #     generate_tree(self.tree, "root", G)
    #     G.view()

# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))

def get_distribution(column: pd.Series):


    """Calculates the distribution of a column.

    Args:
        column (pd.Series): A column

    Returns:
        A pd.Series representing the distribution of the column.
    """
    return column.value_counts(normalize=True)

def is_same_class(column: pd.Series):
    """Checks if all rows in column are the same class.

    Args:
        column (pd.Series): A column in the dataset

    Returns:
        bool: True/False
    """
    column = column.to_numpy()
    return (column[0] == column).all()

def plurality(column: pd.Series):
    """Returns the most numerous class in a column.

    Args:
        column (pd.Series): A column in the dataset

    Returns:
        Str: The most common class.
    """
    return column.value_counts().idxmax()

def remaining_information(X: pd.DataFrame, y: pd.Series, column_name: str):
    """Returns the remaining information from a split in column_name.

    Args:
        X (pd.DataFrame): The feature columns
        y (pd.Series): The target column
        column_name (str): The column_name to split upon

    Returns:
        _type_: The remaining entropy of a partitioning.
    """
    # Makes an array containing Dataframes splitted on row value.
    splits = [y for x, y in X.groupby(column_name, as_index=False)]

    # Initializes weights and entropies arrays
    size= len(splits)
    weights = np.empty(shape=size)
    entropies = np.empty(shape=size)

    # Finds weights and entropies for each split
    for i, split in enumerate(splits):
        weights[i] = len(split.index)/len(X.index)
        entropies[i] = entropy(get_distribution(y.loc[split.index]))
    
    # Returns the remaining information
    return np.dot(weights, entropies)

def max_information_gain(X: pd.DataFrame, y: pd.Series, column_names: np.ndarray, max_branches: int):
    """Calculates the split yielding maximum information gain. 

    Args:
        X (pd.DataFrame): The X from the fit method.
        y (pd.Series): The y from the fit method.
        column_names (np.ndarray): A list of remaining columns to split upon.
        max_branches (int): A split is only accepted if it produces less than or equal to max_branches branches.

    Returns:
        str: The best column to split upon.
    """
    best_column = column_names[0]
    max_info_gain = 0
    target_entropy = entropy(get_distribution(y))

    for name in column_names:
        info_gain = target_entropy - remaining_information(X, y, name)
        if info_gain > max_info_gain and len(X[name].unique()) <= max_branches:
            max_info_gain = info_gain
            best_column = name
    return best_column