
import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque
import tensorflow as tf
import math

from settings import s, e

class Model():


    def __init__(self, logger, decay=0.95, beta = 0.01):
        """initialize the model with standard values"""
        self.inputs = tf.placeholder(shape=[None, 289],dtype=tf.float32)

        self.weights1 = tf.Variable(tf.random_normal([289,289],stddev=0.05))
        self.weights2 = tf.Variable(tf.random_normal([1024,1024],stddev=0.05))
        self.weights3 = tf.Variable(tf.random_normal([1024,1024],stddev=0.05))
        self.weights4 = tf.Variable(tf.random_normal([289,6],stddev=0.05))

        self.bias1 = tf.Variable(tf.random_normal([1024], stddev=0.05))
        self.bias2 = tf.Variable(tf.random_normal([1024], stddev=0.05))
        self.bias3 = tf.Variable(tf.random_normal([1024], stddev=0.05))
        self.bias4 = tf.Variable(tf.random_normal([6], stddev=0.05))


        hidden_out1 = tf.matmul(self.inputs,self.weights1)
        #hidden_out2 = tf.matmul(hidden_out1, self.weights2)
        #hidden_out3 = tf.matmul(hidden_out2, self.weights3)
        self.Qout = tf.matmul(hidden_out1,self.weights4)
        self.prediction = tf.argmax(self.Qout, 1)

        self.nextQ = tf.placeholder(shape=[None,6],dtype=tf.float32)

        loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(0.02, global_step,
                                           100000, 0.95)
        trainer = tf.train.AdamOptimizer(learning_rate)
        self.updateModel = trainer.minimize(loss, global_step=global_step)
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)
        self.decay = decay
        self.saver = tf.train.Saver()
        self.logger = logger
        logger.debug("initialized session")
        return

    def fit(self, batches):
        """fit the current estimator based on the expected and
        real rewards of recorded training events"""
        for data in batches:
            newQ = self.sess.run(self.Qout,feed_dict={self.inputs:np.asarray(data['new_states']).reshape(1, 289)})
            maxQ = np.amax(newQ, axis=1)
            targetQ = np.asarray(data['oldQ'])
            rewards = np.asarray(data['rewards'])
            n_step = rewards.shape[0]
            decaying_rewards = 0
            for i in range(n_step):
                decaying_rewards += rewards[i] * (self.decay ** i)
            targetQ[np.arange(targetQ.shape[0]), np.asarray(data['actions'])] = decaying_rewards + (self.decay**n_step)*maxQ
            self.sess.run([self.updateModel],
                            feed_dict={self.inputs:np.asarray(data['old_states']).reshape(1, 289),
                            self.nextQ:targetQ})
        return

    def estimate(self, state):
        """estimate the gain of an action, based on the current world state"""
        return self.sess.run([self.prediction, self.Qout], feed_dict = {self.inputs: np.expand_dims(state.flatten(), axis=0)})

    def load_model(self, path = None):
        """setup the weigths and the like of the model"""

        if path:
            self.logger.debug(f'Loading weigths from: {path}')
            try:
                self.saver.restore(self.sess, path)
            except Exception as e:
                self.logger.debug(f'Couldn\'t load weights from {path}. Error message: {e}')
            #load the weigths from the path here
        else:
            self.logger.debug('Initializing with random weights')
            #else, initialize randomly
        return

    def save_model(self, path = None):
        if path:
            self.logger.debug(f'Writing weights to: {path}')
            try:
                self.saver.save(self.sess, path)
            except Exception as e:
                self.logger.debug(f'Couldn\'t save weights to {path}. Error message: {e}')
            #save weights to file
        else:
            self.logger.debug('Not saving weights')
        return

def create_state(self):
    x, y, _, bombs_left = self.game_state['self']
    arena = self.game_state['arena']
    coins = self.game_state['coins']
    pos = np.zeros(arena.shape)
    pos[x,y] = 1
    coin_pos = np.zeros(arena.shape)
    coin_pos[np.asarray(coins)] = 1
    self.state = pos - coin_pos
    return

def choose_rand_action(self, prob, count):
    actions = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']
    if np.random.rand(1) < prob:
        self.rand_count = count
    if self.rand_count > 0 or self.rand_round > 0:
        self.rand_count -= 1
        self.next_action = np.random.choice(actions[:-2])
        self.logger.debug(f"Randomly chosen action {self.next_action}")
    return

def calculate_reward(self):
    coins = self.game_state['coins']
    x, y, _, bombs_left = self.game_state['self']
    old_dist = self.coindist
    events = self.events
    self.coindist = np.min(np.sum(np.abs(np.asarray([x, y]) - np.asarray(coins)), axis = 1))
    reward = 0
    #calculate a reward to give for a given action
    if e.COIN_COLLECTED in events:
        reward += 1500
    else:
        reward += min(145,150*(old_dist - self.coindist))
    if e.KILLED_SELF in events:
        reward -= 200
    if e.WAITED in events:
        reward -= 1000
    if e.INVALID_ACTION in events:
        reward -= 1000
    self.score += reward
    return reward

def create_train_batch(self, i):
    train_batch = {'old_states': [], 'new_states': [], 'oldQ': [], 'actions': [], 'rewards': []}
    train_batch['old_states'] = self.data['old_states'][i]
    train_batch['new_states'] = self.data['new_states'][i+self.n_step]
    train_batch['oldQ'] = self.data['oldQ'][i]
    train_batch['actions'] = self.data['actions'][i]
    train_batch['rewards'] = self.data['rewards'][i:self.n_step+i]
    return train_batch

def create_training_data(self, reward, old_state):
    actions = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']
    self.data['old_states'].append(old_state)
    self.data['new_states'].append(self.state)
    self.data['oldQ'].append(self.scores)
    self.data['actions'].append(actions.index(self.last_action))
    self.data['rewards'].append(reward)


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
    self.n_step = 5
    self.model = Model(self.logger)
    #self.model.load_model('/home/russell/bomberman_ai/agent_code/nn_agent/weights.sav')
    self.data = {'old_states': [], 'new_states': [], 'oldQ': [], 'actions': [], 'rewards': []}
    self.score = 0
    self.rand_count = 0
    self.coindist = 0
    self.rand_round = 0
    self.train_set =[]



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
    # Gather information about the game state
    #prepare the data given to the model to estimate here
    actions = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']
    create_state(self)

    action_code, self.scores = self.model.estimate(self.state)
    self.logger.debug(f"choosing action {action_code[0]} with scores {self.scores[0]}")
    self.next_action = actions[action_code[0]]
    choose_rand_action(self, 0.05, 1)
    self.last_action = self.next_action

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
    old_state = self.state
    create_state(self)
    reward = calculate_reward(self)
    create_training_data(self, reward, old_state)


def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')
    old_state = self.state
    create_state(self)
    reward = calculate_reward(self)
    create_training_data(self, reward, old_state)

    self.logger.debug(f"Reward in episode: {self.score}")
    self.score = 0

    for i in range(len(self.data['old_states']) - self.n_step):
        self.train_set.append(create_train_batch(self, i))

    self.model.fit(np.random.choice(self.train_set, size=(math.ceil(self.game_state['step']/2)), replace=False))

    self.data = {'old_states': [], 'new_states': [], 'oldQ': [], 'actions': [], 'rewards': []}
    self.model.save_model('/home/russell/bomberman_ai/agent_code/nn_agent/weights.sav')
    self.rand_round -= 1
    self.rand_count = 0
