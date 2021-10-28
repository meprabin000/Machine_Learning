# Prabin Lamichhane
# 1001733599

import numpy as np
import collections 
import math
import sys

# Node of the tree
class Node:
    def __init__(self, tree_id = 1, node_id = -1, feature_id = -1, thresold = -1, distribution = None, gain = 0):
        self.left = None
        self.right = None
        self.tree_id = tree_id
        self.node_id = node_id
        self.feature_id = feature_id
        self.thresold = thresold
        self.gain = gain

def load_data( path ):
    data = np.loadtxt(path, dtype=str)
    x = np.asarray(data[:,0:-1], dtype = np.float64)
    y = data[:,-1]
    return x, y

# return the unique attributes
def get_attributes( data ):
    return np.arange(data[0].shape[1])

def get_distribution( data ):
    class_freq = collections.Counter(data[1])
    total = len(data[1])
    for c in class_freq.keys():
        class_freq[c] = class_freq[c] / total
    return class_freq

def get_max_distr( distribution ):
    return max(distribution, key = distribution.get)

def get_max_prob( data ):
    distr = get_distribution( data )
    return distr[ get_max_distr( distr ) ]

def choose_attribute( data, option ):
    if option == "optimized":
        return choose_attribute_optimized( data )
    elif option == "randomized" or option == "forest3" or option == "forest15":
        return choose_attribute_randomized( data )
    return None

def choose_attribute_optimized( data ):
    max_gain = best_attribute = best_thresold = -1
    attributes = get_attributes( data )
    for A in attributes:
        attribute_values = data[0][:,A]
        L = min(attribute_values)
        M = max(attribute_values)
        for k in range(1, 51):
            thresold = L + k * (M - L) / 51
            gain = information_gain( data, A, thresold )
            if gain > max_gain:
                max_gain = gain
                best_attribute = A
                best_thresold = thresold
    return best_attribute, best_thresold, max_gain

def choose_attribute_randomized( data ):
    max_gain = best_thresold = -1
    A = np.random.choice( get_attributes(data) )
    attribute_values = data[0][:,A]
    L = min(attribute_values)
    M = max(attribute_values)
    for k in range(1, 51):
        thresold = L + k * (M - L) / 51
        gain = information_gain( data, A, thresold )
        if gain > max_gain:
            max_gain = gain
            best_thresold = thresold
    return A, best_thresold, max_gain

def information_gain( data, A, thresold ):
    def calc_entropy( data ):
        sum_d = 0
        distribution = get_distribution( data )
        for item in distribution.values():
            sum_d += (-item) * math.log(item, 2)
        return sum_d
    left_data = filter_data( data, A, thresold, mask = "left" )
    right_data = filter_data( data, A, thresold, mask = "right" )
    H_E = calc_entropy( data )
    H_left = calc_entropy( left_data )
    H_right = calc_entropy( right_data )
    H_left_prob = len(left_data[0]) / len(data[0])
    H_right_prob = 1 - H_left_prob
    return H_E - (H_left_prob * H_left + H_right_prob * H_right)
    

def filter_data( data, feature_id, thresold, mask = None ):
    left_mask = data[0][:,feature_id] < thresold
    right_mask = ~left_mask  
    
    if mask == "left":
        return data[0][left_mask], data[1][left_mask]
    elif mask == "right":
        return data[0][right_mask], data[1][right_mask]
    
    print("Error: mask is none in filter data.\n")
    return None
    
def DTL( data, default, pruning_thru, option, tree_id, index = 1 ):
    if len(data[0]) < pruning_thru: return default
    if not np.any(data[0]): return default
    elif get_max_prob( data ) == 1: return get_distribution( data )
    else:
        best_attribute, best_thresold, max_gain = choose_attribute( data, option )
        
        tree = Node(tree_id = tree_id, gain = max_gain, node_id = index, feature_id = best_attribute, thresold= best_thresold )
        left_data = filter_data(data, best_attribute, best_thresold, mask = "left")
        right_data = filter_data(data, best_attribute, best_thresold, mask = "right")
        dist = get_distribution(data)
        tree.left = DTL( left_data, dist, pruning_thru, option, tree_id = tree_id, index = 2 * index )
        tree.right = DTL( right_data, dist, pruning_thru, option, tree_id = tree_id, index = 2 * index + 1  )
        return tree

def DTL_Toplevel( data, pruning_thru, option, tree_id ):
    default = get_distribution( data )
    return DTL( data, default, pruning_thru, option, tree_id = tree_id )

def calc_accuracy( data, trees ):
    correct_sum = 0
    
    x, y = data
    unique_labels = np.unique( y )
    for ind in range(len(x)):
        new_dict = dict.fromkeys(unique_labels, 0)
        distributions = []
        for root in trees:
            while isinstance(root, Node):
                if( x[ind][root.feature_id] < root.thresold ):
                    root = root.left
                else:
                    root = root.right
            distributions.append( root )
        for key in unique_labels:
            for tree_id in range(len(trees)):
                if key in distributions[tree_id]:
                    new_dict[key] += distributions[tree_id][key]
            new_dict[key] /= len(trees)
        prediction = get_max_distr( new_dict )
        predict_prob = 0
        if( prediction == y[ind] ): 
            correct_sum += 1
            predict_prob = 1
        print('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f' % (ind+1, int(prediction), int(y[ind]), predict_prob ))
    
    return correct_sum / len(data[0])
        
        


def print_tree( tree_id = -1, parent_id = 1, queue = [] ):   
    if queue:
        head = queue.pop(0)
        print("tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f" % (head.tree_id, head.node_id, head.feature_id, head.thresold, head.gain))
        
        if isinstance( head.left, Node ): 
            queue.append( head.left )
        elif head.left is not None:
            queue.append( Node( tree_id = tree_id, node_id = head.node_id * 2))
        
        if isinstance( head.right, Node ): 
            queue.append( head.right )
        elif head.right is not None:
            queue.append( Node( tree_id = tree_id, node_id = head.node_id * 2 + 1))
    
        print_tree(tree_id, parent_id + 1 , queue )
    return
        
def run_decision_tree( data, pruning_thru, option ):
    forest = []
    num_trees = 1
    
    if option == "forest3":
        num_trees = 3
    elif option == "forest15":
        num_trees = 15
        
    for tree_id in range(num_trees):
        tree = DTL_Toplevel(data, pruning_thru, option, tree_id = tree_id + 1)
        forest.append( tree )
        print_tree( tree_id = tree_id+1, queue = [tree] )
    return forest

def decision_tree( training_file, test_file, option, pruning_thru):
    data = load_data(training_file)
    test_data = load_data(test_file)
    forest = run_decision_tree( data, pruning_thru, option)
    accuracy = calc_accuracy( test_data, forest)
    print( 'classification accuracy=%6.4f\n' % accuracy )

def main( ):
    if len(sys.argv) == 5:
        decision_tree(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        print("Not enough arguments")

if __name__ == "__main__":
    main()