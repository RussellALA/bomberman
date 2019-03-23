import numpy as np
from random import shuffle, randint
from time import time, sleep
from collections import deque
import sklearn.ensemble as ensemble
import sklearn.multioutput as multi
#import pickle
import dill
import math
from copy import copy
import importlib

from settings import s, e

class FakePlayer:
    """Helper Class to save the rotations of the game state in"""
    def __init__(self):
        self.game_state = {}
        self.logger = None

class Node:
    """The single components of a tree"""
    id = 0
    def __init__(self):
        self.id = copy(Node.id)
        Node.id += 1

    def __gt__(self, node):
        return self.id > node.id

    def __ge__(self, node):
        return self.id >= node.id

    def __lt__(self, node):
        return self.id < node.id

    def __le__(self, node):
        return self.id <= node.id

    def __eq__(self, node):
        return self.id == node.id

    def __ne__(self, node):
        return self.id != node.id

    def get_labels(self):
        """Gets all the labels of a node by surveying its children.
        This function is necessary, since saving all the data and labels in the nodes directly is way too slow and costly."""
        if hasattr(self, "left") and hasattr(self, "right"):
            return np.append(self.left.get_labels(), self.right.get_labels())
        elif hasattr(self, "labels"):
            return self.labels
        return []

    def get_path(self):
        """Gets the tree path from the root node to this node."""
        path = []
        node = self
        while node.parent is not None:
            path.append(node)
            node = node.parent
        path.append(node)
        return path

    def clear_duplicates(self, n, logger):
        """This function checks if a certain feature configuration in the node appears more than n times.
        If it does, it replaces all of them with n copies of the average of all of these.
        This is meant to reduce the impact of having had games in the past where the same
        feature appeared way too often and "clutters" the node's memory"""
        unique, counts = np.unique(self.data, return_counts=True, axis =0 )
        trim = np.where(counts > n)[0]
        for d in unique[trim]:
            indices = np.all(np.equal(self.data, d), axis = 1)
            to_keep = np.logical_not(indices)
            labels_mean = np.mean(self.labels[indices,:], axis = 0)
            self.data = np.append(self.data[to_keep, :], np.repeat([d], n, axis = 0), axis = 0)
            self.labels = np.append(self.labels[to_keep,:], np.repeat([labels_mean], n, axis = 0), axis=0)
        return len(trim)

class Tree:
    """From the example solution of exercise 4a"""
    def __init__(self):
        self.root = Node()
        self.root.parent = None

    def find_leaf(self, x):
        node = self.root
        while hasattr(node, "feature"):
            j = node.feature
            if x[j] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node





class RegressionTree(Tree):
    def __init__(self, logger, max_depth=5, min_split_node=20, shrinking_factor=0.95, min_samples_leaf=4, max_duplicates=30, loss = 'LS'):
        """Parameters:
        max_depth: limits the maximum depth the tree can have
        min_split_node: each node has to have at least this many datapoints to be split
        shrinking_factor:   a node's response is calculated in part by its parents' responses.
                            The shrinking factor determines how much. See report for details.
        min_samples_leaf:   after a split, each of the resulting nodes has to have
                            at least this many datapoints. If this isn't the case,
                            the split is rejected.
        max_duplicates: The maximum number of datapoints a leaf can hold which have the
                        same features. When calling clean leaves any additional datapoints
                        are averaged out.
        loss: The function to determine the score of a split"""
        super(RegressionTree, self).__init__()
        self.max_depth = max_depth
        self.n_min = min_split_node
        self.shrinking_factor = shrinking_factor
        self.min_samples_leaf = min_samples_leaf
        self.max_duplicates = max_duplicates
        self.loss = loss
        #self.logger = logger

    def c_loss(self, x, y, huber_loss = 20):
        """Determines how good each split is"""
        if self.loss == 1 or self.loss == 'LS':
            return np.sum((x - y) ** 2)/x.shape[0]
        elif self.loss == 2 or self.loss == 'ABS':
            return np.sum(np.abs(x - y))/x.shape[0]
        elif self.loss ==3 or self.loss == 'Huber':
            t = np.sum(np.abs(x - y))/x.shape[0]
            if t <= huber_loss:
                return huber_loss*(t - 0.5 * huber_loss)
            else:
                return 0.5 * np.sum((x - y) ** 2)/x.shape[0]
        else:
            #self.logger.debug(f'Enter valid loss in Regression Tree Initialization: 1 - LS, 2 - ABS, 3 - Huber')
            return 1

    def fit(self, data, labels, logger=None, retrain=False):
        '''
        From the example solution of exercise 4a.
        data: the feature matrix for all digits
        labels: the corresponding ground-truth responses
        n_min: termination criterion (don't split if a node contains fewer instances)
        '''
        N, D = data.shape
        D_try = math.ceil(np.sqrt(D)) # how many features to consider for each split decision

        stack = []

        if retrain:
            # initialize the root node
            self.root.data = data
            self.root.labels = labels

            # put root in stack
            stack = [self.root]
        else:
            # if the tree is already trained, add the new datapoints in the respective
            #leaves and look at them, to see if they need to be split
            stack = self.add_data(data, labels, logger=logger)
        while len(stack):
            node = stack.pop()
            n = node.data.shape[0] # number of instances in present node
            if n >= self.n_min and get_depth(node) < self.max_depth:
                # Call 'make_decision_split_node()' with 'D_try' randomly selected
                # feature indices. This turns 'node' into a split node
                # and returns the two children, which must be placed on the 'stack'.
                perm = np.random.permutation(D)   # permute D indices
                left, right = make_regression_split_node(node, perm[:D_try],logger, self.min_samples_leaf, self.c_loss) #select :D_try of permuted indices
                if left is not None and right is not None:
                    # put children in stack
                    stack.append(left)
                    stack.append(right)
                else:
                    make_regression_leaf_node(node, self.shrinking_factor)
            else:
                # Call 'make_regression_leaf_node()' to turn 'node' into a leaf node.
                make_regression_leaf_node(node, self.shrinking_factor)

    def predict(self, dset):
        pred_set = np.apply_along_axis(self.predict_single, axis=1, arr=dset)
        return pred_set

    def predict_single(self, x):
        leaf = self.find_leaf(x)
        return leaf.response

    def add_data(self, data, labels, logger=None):
        """Finds the leaves that the new training data belongs into,
        adds it to them and returns a list of the leaves that were altered"""
        buckets = np.apply_along_axis(self.find_leaf, axis=1, arr=data)
        changed_leaves = np.unique(buckets)
        for k in range(len(data)):
            x = data[k]
            y = labels[k]
            leaf = buckets[k]
            leaf.data = np.append(leaf.data, [x], axis=0)
            leaf.labels = np.append(leaf.labels, [y], axis=0)
        return changed_leaves.tolist()

    def clean_leaves(self, logger):
        """Cleanes out the duplicates of each leaf"""
        stack = [self.root]
        clean_count = 0
        while(len(stack)):
            node = stack.pop()
            if hasattr(node, "left") and hasattr(node, "right"):
                stack.append(node.left)
                stack.append(node.right)
            elif hasattr(node, "labels"):
                clean_count += node.clear_duplicates(self.max_duplicates, logger)
        return clean_count

    def print_leaves(self):
        """Debugging function. Grants access to all leaves and all splits."""
        stack = [self.root]
        leaves = []
        splits = []
        while len(stack):
            node = stack.pop()
            if hasattr(node, "left") and hasattr(node, "right"):
                splits.append(node.feature)
                stack.append(node.left)
                stack.append(node.right)
            elif hasattr(node, "labels"):
                leaves.append(len(node.labels))
        return leaves

    def freeze(self):
        stack = [self.root]
        while len(stack):
            node = stack.pop()
            if hasattr(node, "left") and hasattr(node, "right"):
                stack.append(node.left)
                stack.append(node.right)
            if hasattr(node, "labels"):
                del node.labels
            if hasattr(node, "data"):
                del node.data
        return

def make_regression_split_node(node, feature_indices, logger, min_split, loss):
    '''
    Taken from example solution to exercise 4a.
    node: the node to be split
    feature_indices: a numpy array of length 'D_try', containing the feature
                     indices to be considered in the present split
    min_split: the minimal number of datapoints in a leaf node
    loss: functor to calculate the loss
    '''
    n, D = node.data.shape

    # find best feature j (among 'feature_indices') and best threshold t for the split
    e_min = 1e100
    j_min, t_min = -1, 0
    for j in feature_indices:
        # remove duplicate features
        dj = np.sort(np.unique(node.data[:,j]))
        # compute candidate thresholds in the middle between consecutive feature values
        tj = 0.5 * (dj[1:] + dj[:-1])
        # each candidate threshold we need to compute Gini impurities of the resulting children node
        if len(np.unique(tj)) >= 1:
            for t in tj:
                left_indices = node.data[:,j] <= t
                nl = np.sum(left_indices)
                ll = node.labels[left_indices, :]
                el = loss(ll, ll.mean())
                nr = n - nl
                right_indices = node.data[:,j] > t
                lr = node.labels[right_indices, :]
                er = loss(lr, lr.mean())
                # choose the the best threshold that minimizes the sum of the losses and fullfills the min_split criteria
                if el + er < e_min and len(ll) > min_split and len(lr) > min_split:
                    e_min = el + er
                    j_min = j
                    t_min = t

    #if no split is found, do nothing
    if j_min == -1:
        return None, None
    # create children
    left = Node()
    right = Node()

    # initialize 'left' and 'right' with the data subsets and labels
    # according to the optimal split found above
    left.data = node.data[node.data[:,j_min] <= t_min, :]
    left.labels = node.labels[node.data[:,j_min] <= t_min, :]
    right.data = node.data[node.data[:,j_min] > t_min, :]
    right.labels = node.labels[node.data[:,j_min] > t_min, :]

    # turn the current 'node' into a split node
    # (store children and split condition)
    node.left = left
    node.right = right
    node.feature = j_min
    node.threshold = t_min
    left.parent = node
    right.parent = node
    #remove the data and labels from parent node to save memory and speed up calculations
    del node.data
    del node.labels
    # return the children (to be placed on the stack)
    return left, right

def make_regression_leaf_node(node, s):
    '''
    node: the node to become a leaf
    s: the shrinking factor
    '''
    #calculate node's response accorting to shrinking factor. See report for details
    path = node.get_path()
    means = [(s)*((1-s)**i)*np.mean(path[i].get_labels(), axis=0) for i in range(len(path))]
    means[-1] = means[-1]/s
    node.response = np.sum(means, axis=0)

def get_depth(node):
    """Calculates the depth of a node in the tree"""
    depth = 0
    while node.parent is not None:
        node = node.parent
        depth += 1
    return depth

def rounder(x):
    """Helper function for consistent rounding behaviour"""
    if (x-int(x) >= 0.5):
        return np.ceil(x)
    else:
        return np.floor(x)

class GradientBooster():
    """Combines a number of estimators (here Regression Trees) progressively make a prediction and refine it."""
    def __init__(self, logger, n_estimators = 60, learning_rate = 0.1, max_depth = [5,12], output_shape = 1, min_split_node=20, deep_first_tree=False, min_samples_leaf=4, max_duplicates=30, loss='LS', shrinking_factor=0.95):
        """Parameters:
        n_estimators: Number of estimators (here Trees) the GradientBooster holds
        learning_rate:  How much each estimator's response contributes to the overall
                        response of the GradientBooster
        max_depth:  The depth limit of the estimators. If it is a list, the depth limit
                    will be scaled from the first to the last value
        output_shape: Which dimensions the output has
        min_split_node: each node in the estimators has to have at least this many datapoints to be split
        deep_first_tree: whether the First tree should have twice the depth or just its normal depth
        min_samples_leaf:   after a split, each of the resulting nodes has to have
                            at least this many datapoints. If this isn't the case,
                            the split is rejected.
        max_duplicates: The maximum number of datapoints a leaf can hold which have the
                        same features. When calling clean leaves any additional datapoints
                        are averaged out.
        loss: The function to determine the score of a split
        shrinking_factor:   a node's response is calculated in part by its parents' responses.
                            The shrinking factor determines how much. See report for details."""
        self.estimators = []
        #The constant estimator is the average of all training labels, used to get
        #a good starting point for the tree's estimation
        self.constant_estimator = np.zeros(output_shape)
        self.n_samples = 0
        self.trained = False
        ramp = np.repeat(5, n_estimators)
        rounder_vec = np.vectorize(rounder)
        #if max_depth is a list, create estimators with a ramping depth, else, keep the depth constant
        if type(max_depth) == list:
            if len(max_depth) == 2:
                ramp = rounder_vec(np.linspace(max_depth[0], max_depth[1], num = n_estimators))
        elif type(max_depth) == int:
            ramp = np.repeat(max_depth, n_estimators)
        #Create the needed estimators
        for i in range(n_estimators):
            if i == 0 and deep_first_tree:
                self.estimators.append(RegressionTree(logger = logger, max_depth=ramp[i]*2, min_split_node=min_split_node, min_samples_leaf=min_samples_leaf, max_duplicates=max_duplicates, loss=loss, shrinking_factor=shrinking_factor))
            else:
                self.estimators.append(RegressionTree(logger = logger, max_depth=ramp[i], min_split_node=min_split_node, min_samples_leaf=min_samples_leaf,  max_duplicates=max_duplicates, loss=loss, shrinking_factor=shrinking_factor))
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        #self.logger = logger
        return

    def fit(self, X, y, logger=None, retrain=False):
        """Parameters:
        X: The training features
        y: The training labels
        retrain: If True, overwrite all training progress"""
        #modify the constant estimator so that it includes the new training labels
        estimation_update = np.expand_dims(np.sum(y, axis=0)/y.shape[0], axis=0)
        if not retrain:
            total_n = self.n_samples + y.shape[0]
            self.constant_estimator = estimation_update * (y.shape[0])/total_n + self.constant_estimator *(self.n_samples)/total_n
            self.n_samples = total_n
        else:
            self.n_samples = y.shape[0]


        current_guess = np.repeat(self.constant_estimator, y.shape[0], axis = 0)
        #bootstrap sampling to present each tree with different data
        bootstrap = np.random.choice(np.arange(X.shape[0]), X.shape[0], replace=True)
        Xtrain = X[bootstrap]
        ytrain = y[bootstrap]
        for i,t in enumerate(self.estimators):
            #error of the estimators' predictions up to this point. A large error
            #is equivalent to a high weight of that sample.
            dist = ytrain - current_guess
            t.fit(Xtrain, dist, logger=logger, retrain=(retrain or not self.trained))
            #bootstrap sampling to present each tree with different data
            bootstrap = np.random.choice(np.arange(X.shape[0]), X.shape[0], replace=True)
            Xtrain = X[bootstrap]
            ytrain = y[bootstrap]
            #predict the new bootstrapped training set to get the training data for the next tree
            current_guess = np.repeat(self.constant_estimator, y.shape[0], axis = 0)
            for h in self.estimators[:i+1]:
                correction = h.predict(Xtrain)
                current_guess += self.learning_rate * correction
        self.trained = True
        return

    def clean_trees(self, logger):
        """Calls the cleaning function of each estimator"""
        clean_count = 0
        for t in self.estimators:
            clean_count += t.clean_leaves(logger)
        return clean_count

    def predict(self, X, logger=None):
        y = np.repeat(self.constant_estimator, X.shape[0], axis=0)
        if self.trained:
            for t in self.estimators:
                c = self.learning_rate * t.predict(X)
                y += c
        return y

    def freeze(self):
        for t in self.estimators:
            t.freeze()
        return

    def add_max_depth(self, n):
        for t in self.estimators:
            t.max_depth += n


class Model():


    def __init__(self, logger, input_shape, Q_bounds, decay=0.95):
        """initialize the model with standard values
        Decay: how much the reward after 1 step in the future still affects the Q-value
        Q_bounds: the maximum Q-value of any state to take into consideration for reward calculation. This prevents divergence."""
        self.forest = GradientBooster(logger = logger, max_depth=[6,11], output_shape=(1,6), min_split_node=16, max_duplicates=50, n_estimators=80, learning_rate=0.05, min_samples_leaf=8, deep_first_tree=False, shrinking_factor=0.98, loss='Huber')
        self.decay = decay
        self.logger = logger
        self.input_shape = input_shape
        self.Q_bounds = Q_bounds
        return

    def fit(self, batch):
        """fit the current estimator based on the expected and
        real rewards of recorded training events"""
        train_err = 0
        #prepare the training data for the GradientBooster
        s = {'S': [], 'Q':[]}
        for data in batch:
            #Get the Q-values of the new states that resulted from the action
            newQ = self.forest.predict(np.asarray([data['new_states']]))
            maxQ = np.amax(newQ, axis=1)
            targetQ = np.asarray([data['oldQ']])
            targetQ = targetQ.reshape(targetQ.shape[0],targetQ.shape[2])
            rewards = np.asarray([data['rewards']])
            n_step = rewards.shape[1]
            #calculate the decaying rewards (rewards received within the n steps)
            decaying_rewards = np.zeros(rewards.shape[0])
            for i in range(n_step):
                decaying_rewards[:] += rewards[:,i] * (self.decay ** i)
            #modify the old Q-value so that the entry for the experienced action fits the rewards received
            targetQ[np.arange(targetQ.shape[0]), np.asarray([data['actions']])] = decaying_rewards + (self.decay**n_step)*np.clip(maxQ, self.Q_bounds[0], self.Q_bounds[1])
            s['Q'].append(targetQ)
            s['S'].append(data['old_states'])
        s['S'] = np.asarray(s['S'])
        s['Q'] = np.asarray(s['Q'])
        #Train the GradientBooster on the prepared dataset
        self.forest.fit(np.asarray(s['S']).reshape(len(s['S']), self.input_shape), np.asarray(s['Q']).reshape(len(s['S']), 6), logger=self.logger)
        self.logger.debug(f"Cleaned {self.forest.clean_trees(self.logger)} duplicate datapoints")
        #Calculate the training error
        Q = self.forest.predict(s['S'].reshape(len(s['S']), self.input_shape))
        train_err += np.sqrt(np.sum((targetQ - Q) ** 2))
        self.logger.debug(f"Training error: {train_err/len(batch)}")
        return

    def estimate(self, state):
        """estimate the gain of an action, based on the current world state"""
        prediction =  self.forest.predict(np.asarray([state.flatten()]))
        return np.argmax(prediction), prediction

    def load_model(self, path = None):
        """setup the weigths and the like of the model"""
        self.logger.debug(f'Loading weights from: {path}')
        if path:
            try:
                self.forest = dill.load(open(path, 'rb'))
            except Exception as er:
                self.logger.debug(f'Couldn\'t load weights from {path}. Error message: {er}')
            #load the weigths from the path here
        else:
            self.logger.debug('Initializing with random weights')

            #else, initialize randomly
        return

    def save_model(self, path = None):
        if path:
            self.logger.debug(f'Writing weights to: {path}')
            #save weights to file
            try:
                dill.dump(self.forest, open(path, 'wb'))
            except Exception as er:
                self.logger.debug(f'Couldn\'t save weights to {path}. Error message: {er}')
        else:
            self.logger.debug('Not saving weights')
        return

    def export_model(self, path = None):
        if path:
            self.logger.debug(f'Writing weights to: {path}')
            #save weights to file
            try:
                f = copy(self.forest)
                f.freeze()
                dill.dump(f, open(path, 'wb'))
            except Exception as er:
                self.logger.debug(f'Couldn\'t save weights to {path}. Error message: {er}')
        else:
            self.logger.debug('Not saving weights')
        return

def getcrates(x, y, i, arena):
    """This function looks 3 tiles in a given direction (i==0: right, i==1: left, i==2: up, i==3: down)
    and checks what the next obstacle is. If it is a crate, returns 1, if it is a wall returns 2.
    If the direction is free up to that point, it looks if there is any kind of obstacle in the next tile,
    which would block the escape in that direction if the agent were to put down a bomb now.
    If that is the case, it also returns 2, else it returns 0."""
    if i == 0:
        for k in arena[x+1:x+4,y]:
            if k == 1:
                return 1
            if k == -1:
                return 2
        if arena[x+4,y]:
            return 2
    if i == 1:
        for k in arena[::-1,:][17-x:17-(x-3), y]:
            if k == 1:
                return 1
            if k == -1:
                return 2
        if arena[x-4,y]:
            return 2
    if i == 2:
        for k in arena[x,y+1:y+4]:
            if k == 1:
                return 1
            if k == -1:
                return 2
        if arena[x,y+4]:
            return 2
    if i == 3:
        for k in arena[:,::-1][x, 17-y:17-y+3]:
            if k == 1:
                return 1
            if k == -1:
                return 2
        if arena[x,y-4]:
            return 2
    return 0

def create_state(self, init=False):
    """Create the state for the model to look at.
    If init is set, it uses default values to just return the shape of the state that the model should expect
    The state consists of the following:
    * 4 tertiary values, telling the agent whether or not the respective
    direction is blocked by a wall (1) an explosion (2) or nothing (0).
    * 4 tertiary values, telling the agent whether a given direction contains a
    crate to be blown up (1) or a wall (2), or nothing (0) within bomb range.
    Refer to getcrates() for more detail.
    * 4 binary values telling the agent whether a direction is a dead end within
    bomb range or not
    * 1 binary value, telling the agent whether or not it has a bomb left
    * 4 pairs of values between -5 and 5 telling the agent about the relative
    position of each bomb. If there aren't 4 bombs on the field, the other values
    are 0, 0
    * A pair of values between -2 and 2 telling the agent about the relative
    position of the nearest coin
    * 3 pairs of values between -6 and 6 telling the agent about the relative position
    of the enemies. If there aren't 3 enemies on the field, the other values are 0, 0
    * 4 binary values telling the agent whether or not there are crates in the respective
    side of the arena left
    """
    if not init:
        coins = self.game_state['coins']
        x, y, _, bombs_left, _ = self.game_state['self']
        arena = self.game_state['arena']
        bombs = copy(self.game_state['bombs'])
        explosions = self.game_state['explosions']
        others = copy(self.game_state['others'])
        for k in range(max(3 - len(others), 0)):
            others.append((x, y, '', 0))
        for l in range(max(4 - len(bombs), 0)):
            bombs.append((x, y, 0))
        enemies = []
        for op in others:
            enemies.append([op[0], op[1]])
        enemies = np.asarray(enemies)
        if not len(coins):
            coins = [[x, y]]
        for k in range(len(bombs)):
            b = bombs[k]
            bombs[k] = np.asarray([b[0], b[1]])
    else:
        coins = [[0, 0]]
        x,y = (0,0)
        arena = np.zeros((17,17))
        bombs = np.zeros((4,2))
        bombs_left = 0
        explosions = np.zeros((17,17))
        enemies = np.zeros((3,2))

    state = np.zeros(13)
    if arena[x+1,y] != 0: state[0] = 1
    if arena[x-1,y] != 0: state[1] = 1
    if arena[x,y+1] != 0: state[2] = 1
    if arena[x,y-1] != 0: state[3] = 1
    if explosions[x+1, y] > 0: state[0] = 2
    if explosions[x-1, y] > 0: state[1] = 2
    if explosions[x, y+1] > 0: state[2] = 2
    if explosions[x, y-1] > 0: state[3] = 2
    for i in range(4):
        state[4 + i] = getcrates(x, y, i, arena)
        state[8 + i] = get_dead_ends(x, y, i, arena)
    state[12] = bombs_left
    nearest_bomb = np.argmin(np.sum(np.abs(np.asarray([x,y]) - np.asarray(bombs)), axis = 1))
    bomb_dist = np.clip(np.asarray(bombs) - np.asarray([x,y]), -5, 5)
    state = np.append(state, bomb_dist)
    nearest_coin = np.argmin(np.sum(np.abs(np.asarray([x,y])-np.asarray(coins)), axis = 1))
    coindist = np.clip(np.asarray(coins[nearest_coin]) - np.asarray([x,y]), -2, 2)
    state = np.append(state, coindist)
    enemy_dist = np.clip(np.asarray([x,y]) - enemies, -6, 6)
    state = np.append(state, enemy_dist)
    crates = [0, 0, 0, 0]
    if 1 in arena[:,y:]:
        crates[3] = 1
    if 1 in arena[x:, :]:
        crates[2] = 1
    if 1 in arena[:,:y]:
        crates[1] = 1
    if 1 in arena[:x, :]:
        crates[0] = 1
    state = np.append(state, crates)

    return state, len(state.flatten())

def choose_rand_action(self, prob, score):
    """Choose a random action, based on the model's current estimation of the actions
    actions with high scores get chosen more likely, while actions with low scores are chosen
    less often."""
    actions = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']
    shifted_score = -self.Q_bounds[0] + score[0]
    if np.random.rand(1) < prob:
        self.next_action = np.random.choice(actions, p = shifted_score/shifted_score.sum())
    return

def get_dead_ends(x, y, i, arena):
    """This function determines whether a given direction (i==0: right, i==1: left, i==2: up, i==3: down)
    is a dead end within the next 5 tiles or not."""
    if i == 0:
        size = 0
        for l, k in enumerate(arena[x+1:x+5, y]):
            if k != 0:
                size = l
                break
        if size == 0:
            return 0
        if 0 in arena[x+1:x+size, y-1] or 0 in arena[x+1:x+size, y+1]:
            return 0
        else:
            return 1
    if i == 1:
        size = 0
        for l, k in enumerate(arena[::-1,:][17-x:17-x+4, y]):
            if k != 0:
                size = l
                break
        if size == 0:
            return 0
        if 0 in arena[::-1,:][17-x:17-x+size, y-1] or 0 in arena[::-1,:][17-x:17-x+size, y+1]:
            return 0
        else:
            return 1
    if i == 2:
        size = 0
        for l, k in enumerate(arena[x, y+1:y+5]):
            if k != 0:
                size = l
                break
        if size == 0:
            return 0
        if 0 in arena[x+1, y+1:y+size] or 0 in arena[x-1, y+1:y+size]:
            return 0
        else:
            return 1
    if i == 3:
        size = 0
        for l, k in enumerate(arena[:,::-1][x, 17-y:17-y+4]):
            if k != 0:
                size = l
                break
        if size == 0:
            return 0
        if 0 in arena[:,::-1][x-1, 17-y:17-y+size] or 0 in arena[:,::-1][x+1, 17-y:17-y+size]:
            return 0
        else:
            return 1

def calculate_reward(self):
    """Calculate the reward for a game step
    The reward looks at:
    Did the agent collect a coin? +500
    Did the agent move closer to a coin? +5
    Did the agent leave its corner for the first time this game? +400
    Did the agent blow up a crate? +200
    Did the agent end up on the same tile as it was in the last 2 moves? -5
    Did the agent kill itself? -800
    Did the agent get killed by another player? -1000
    Did the agent kill another player? +1000
    Did the agent walk into the explosion area of a bomb? -150
    Did the agent escape the explosion area of a bomb? +100
    Did the agent choose an invalid action? -5
    Rejected idea: Did the agent walk into a dead end while being threatend by a bomb?"""
    coins = self.game_state['coins']
    x, y, _, bombs_left, _ = self.game_state['self']
    bombs = self.game_state['bombs']
    arena = self.game_state['arena']
    old_dist = self.coindist
    events = self.events
    past_threat = self.bomb_threat
    repeat = False
    if len(self.past_pos) > 1:
        if np.sum(np.all(np.equal(np.asarray(self.past_pos[-2:]), np.asarray([x,y])), axis = 1)) > 0:
            repeat = True
    try:
        self.trapped = 0
        self.bomb_threat = False
        if len(coins):
            self.coindist = np.min(np.sum(np.abs(np.asarray([x, y]) - np.asarray(coins)), axis = 1))
        if len(bombs):
            dist = np.asarray([x,y]) - np.asarray(bombs)[:,0:2]
            for b in dist:
                if 0 in b:
                    if np.all(np.abs(b) < 5):
                        self.bomb_threat = True
                    d = np.argmax(np.abs(b))
                    bomb = np.asarray([x,y]) - b
                    if d == 0 and b[d] > 0:
                        self.trapped = get_dead_ends(bomb[0], bomb[1], 0, arena)
                    elif d == 0 and b[d] < 0:
                        self.trapped = get_dead_ends(bomb[0], bomb[1], 1, arena)
                    elif d == 1 and b[d] > 0:
                        self.trapped = get_dead_ends(bomb[0], bomb[1], 2, arena)
                    elif d == 1 and b[d] < 0:
                        self.trapped = get_dead_ends(bomb[0], bomb[1], 3, arena)

    except Exception as er:
        self.logger.debug(f"Encountered Exception {er} in reward calculation")
    reward = 0
    #calculate a reward to give for a given action
    if e.COIN_COLLECTED in events:
        reward += 500
    else:
        if old_dist >= 0:
            reward += max(0,5*(old_dist - self.coindist))
    if self.was_in_corner and ((4 <= x <= 12) or (4 <= y <= 12)):
        self.was_in_corner = False
        reward += 800
    if e.CRATE_DESTROYED in events:
        reward += 200
    if repeat:
        reward -= 6
    if e.KILLED_SELF in events:
        reward += 200
    if e.GOT_KILLED in events:
        reward -= 1000
    if e.KILLED_OPPONENT in events:
        reward +=1000
    if self.bomb_threat and not past_threat and not e.BOMB_DROPPED in events:
        reward -= 150
    elif not self.bomb_threat and past_threat:
        reward += 100
    if e.INVALID_ACTION in events:
        reward -= 5
    self.score += reward
    return reward

def create_train_batch(self, i):
    """Bundle the experience of a single action into a dictionary"""
    train_batch = {'old_states': [], 'new_states': [], 'oldQ': [], 'actions': [], 'rewards': []}
    train_batch['old_states'] = self.data['old_states'][i]
    train_batch['new_states'] = self.data['new_states'][i + (self.n_step - 1)*len(self.rotations)]
    train_batch['oldQ'] = self.data['oldQ'][i]
    train_batch['actions'] = self.data['actions'][i]
    train_batch['rewards'] = self.data['rewards'][i:len(self.rotations)*(self.n_step - 1)+i+1:len(self.rotations)]
    return train_batch

def rotate_state(self, i):
    """Rotate the arena and all coordinates by 90 degrees. This can be used to create
    symmetrical training data, ensuring that the agent learns exactly the same moves
    in each orientation of the state"""
    fake_player = FakePlayer()
    fake_player.game_state['arena'] = np.rot90(self.game_state['arena'], i)
    fake_player.game_state['explosions'] = np.rot90(self.game_state['explosions'], i)
    fake_player.game_state['coins'] = copy(self.game_state['coins'])
    fake_player.game_state['bombs'] = copy(self.game_state['bombs'])
    fake_player.game_state['others'] = copy(self.game_state['others'])
    for k in range(len(fake_player.game_state['bombs'])):
        bomb = fake_player.game_state['bombs'][k]
        x, y = bomb[0], bomb[1]
        t = 1
        try:
            t = bomb[2]
        except Exception:
            pass
        newbomb = (x,y,t)
        if i == 1:
            newbomb = (16 - y, x,t)
        elif i == 2:
            newbomb = (16 - x, 16 - y,t)
        elif i == 3:
            newbomb = (y, 16 - x,t)
        fake_player.game_state['bombs'][k] = newbomb

    for k in range(len(fake_player.game_state['others'])):
        x, y, v1, v2, v3 = fake_player.game_state['others'][k]
        newx, newy = copy(x), copy(y)
        if i == 1:
            newx = 16 - y
            newy = x
        elif i == 2:
            newx = 16 - x
            newy = 16 - y
        elif i == 3:
            newx = y
            newy = 16 - x
        fake_player.game_state['others'][k] = (newx, newy, v1, v2, v3)

    for k in range(len(fake_player.game_state['coins'])):
        coin = fake_player.game_state['coins'][k]
        x,y = coin
        newcoin = (x,y)
        if i == 1:
            newcoin = (16 -y, x)
        elif i == 2:
            newcoin = (16 -x, 16 -y)
        elif i == 3:
            newcoin = (y, 16 - x)
        fake_player.game_state['coins'][k] = newcoin
    x, y, v1, v2, v3 = self.game_state['self']
    newx, newy = copy(x), copy(y)
    if i == 1:
        newx = 16 - y
        newy = x
    elif i == 2:
        newx = 16 - x
        newy = 16 - y
    elif i == 3:
        newx = y
        newy = 16 - x
    fake_player.game_state['self'] = (newx, newy, v1, v2, v3)
    fake_player.logger = self.logger
    return fake_player

def rotate_action(action, i):
    """When rotating an experience by 90 degrees, the action also changes"""
    actions = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']
    action_index = actions.index(action)
    if action_index > 3:
        return action_index
    turn_indices = np.asarray([[0, 1, 2, 3], [3, 2, 0, 1], [1, 0, 3, 2], [2, 3, 1, 0]])
    action_index = turn_indices[i, action_index]
    return action_index

def rotate_score(score, i):
    """Move the scores for each action to its new place"""
    turn_indices = np.asarray([[0, 1, 2, 3, 4, 5], [2, 3, 0, 1, 4, 5], [1, 0, 3, 2, 4, 5], [3, 2, 0, 1, 4, 5]])
    return score[:, turn_indices[i]]


def create_training_data(self):
    """Save all experiences into an array"""
    self.data['old_states'].extend(self.old_states)
    self.data['new_states'].extend(self.new_states)
    self.data['oldQ'].extend(self.scores)
    self.data['actions'].extend(self.last_actions)
    self.data['rewards'].extend(np.repeat(self.reward, len(self.rotations)).tolist())

def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    self.Q_bounds = [-1000, 1000]
    self.model = Model(self.logger, create_state(self, init=True)[1], self.Q_bounds)

    path = "final_agent_frozen.sav"

    self.round_count = 0
    self.model.load_model(path)
    self.data = {'old_states': [], 'new_states': [], 'oldQ': [], 'actions': [], 'rewards': []}
    #keep track of all the rewards in an episode
    self.score = 0
    self.coindist = 0
    #let another agent play x rounds
    self.other_agent_round = 0
    self.n_step = 8
    #The way the training data should be rotated. These are the i values for the functions
    #so training on all 4 valid rotations would be [0, 1, 2, 3]
    self.rotations = [0]
    self.train_set = []
    #whether or not the agent is in range of a bomb
    self.bomb_threat = False
    #the tiles an agent has visited in the last few steps
    self.past_pos = []
    #rejected idea: whether or not the agent is in a dead end blocked by a bomb
    self.trapped = 0
    #wether or not the agent hasn't left the corner yet this episode
    self.was_in_corner = True
    #the agent to play the rounds instead of the gb agent
    self.agent_dir = 'simple_agent'
    self.other_agent = importlib.import_module('agent_code.' + self.agent_dir + '.callbacks')
    self.other_agent.setup(self)
    #which round the training should start
    self.train_round = 0

def act(self):
    """Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via self.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    Set the action you wish to perform by assigning the relevant string to
    self.next_action. You can assign to this variable multiple times during
    your computations. If this method takes longer than the time limit specified
    in settings.py, execution is interrupted by the game and the current value
    of self.next_action will be used. The default value is 'WAIT'.
    """
    #prepare the data given to the model to estimate here
    x,y,_,_,_ = self.game_state['self']
    actions = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']
    current_state, _ = create_state(self)
    all_scores = []
    #One idea was to rotate the state in all 4 directions and not only pick the best
    #score from the current rotation but from any rotation. This didn't seem to work
    for i in range(1):
        all_scores.append(rotate_score(self.model.estimate(create_state(rotate_state(self, i))[0])[1], (4 - i)%4).flatten())
    ind = np.unravel_index(np.argmax(all_scores), (4,6))
    score = np.asarray([np.asarray(all_scores)[ind[0], :]])
    action_code = ind[-1]
    self.next_action = actions[action_code]
    if self.other_agent_round > 0:
        self.other_agent.act(self)
    choose_rand_action(self,0.01, score)
    #prepare data for reward calculation
    self.old_states = []
    self.last_actions = []
    self.scores = []
    for i in self.rotations:
        state, _ = create_state(rotate_state(self, i))
        self.old_states.append(state)
        self.last_actions.append(rotate_action(self.next_action, i))
        self.scores.append(rotate_score(score, i))
    self.past_pos.append(np.asarray([x,y]))


def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    if self.game_state["step"] == 1:
        return
    self.new_states = []
    #create the training data from this step's experiences
    for i in self.rotations:
        state, _ = create_state(rotate_state(self, i))
        self.new_states.append(state)
    self.reward = calculate_reward(self)
    create_training_data(self)





def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """

    path = "final_agent.sav"

    self.new_states = []
    #create the training data for the last experience
    for i in self.rotations:
        state, _ = create_state(rotate_state(self, i))
        self.new_states.append(state)
    self.reward = calculate_reward(self)
    create_training_data(self)

    #Extend the data set artificially such that even for the last move, it can look n steps into the future
    #This is important because otherwise the agent wouldn't be able to train on the last few actions
    self.data['new_states'].extend(np.tile(self.data['new_states'][-(len(self.rotations)):], (self.n_step-1,1)).tolist())
    self.data['rewards'].extend(np.repeat(np.zeros(len(self.rotations)), self.n_step-1).flatten().tolist())

    self.logger.debug(f"Reward in episode: {self.score}")
    self.score = 0
    for i in range(len(self.data['old_states'])):
        self.train_set.append(create_train_batch(self, i))
    self.logger.debug(self.train_set[0])

    if self.round_count == 50:
        self.agent_dir = 'simple_agent'
        self.other_agent = importlib.import_module('agent_code.' + self.agent_dir + '.callbacks')
        self.other_agent.setup(self)
    if self.round_count >= self.train_round:
        self.model.fit(np.random.choice(self.train_set, size=len(self.train_set), replace=False))
        self.train_set = []
        self.model.save_model(path)
        #uncomment to create a frozen version (no data/labels) of the model.
        #ONLY do this AFTER saving the regular model.
        #self.model.export_model(path[:-4] + "_frozen.sav")
    self.data = {'old_states': [], 'new_states': [], 'oldQ': [], 'actions': [], 'rewards': []}
    self.other_agent_round -= 1
    self.coindist = -1
    self.past_pos = []
    self.round_count += 1
    self.was_in_corner = True
