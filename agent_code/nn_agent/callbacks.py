import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.initializers import he_uniform
from keras.optimizers import RMSprop
from keras.layers import Dense, Activation, LeakyReLU, Dropout

from settings import s, e

class Model():

    def __init__(self, logger, decay=0.95, beta = 0.01):
        """initialize the model with standard values"""
        self.n_input = [1, 6]
        self.n_hidden = [10,50,10]
        self.n_classes = 6

        self.inputs = tf.placeholder(shape=self.n_input, dtype=tf.float32)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        K.set_session(self.sess)

        self.model = Sequential()
        self.model.add(Dense(self.n_hidden[0], input_dim=self.n_input[1], activation='sigmoid'))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.n_hidden[1], input_dim=self.n_hidden[0], use_bias=True, kernel_initializer= "he_uniform", bias_initializer= "zeros" ))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.n_hidden[2], input_dim=self.n_hidden[1], use_bias=True, kernel_initializer= "he_uniform", bias_initializer= "zeros" ))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.n_classes, input_dim=self.n_hidden[2], use_bias=True, kernel_initializer="he_uniform", bias_initializer="zeros"))
        rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=rms, loss='mse')

        self.nextQ = tf.placeholder(shape=[None, self.n_classes], dtype=tf.float32)
        self.Qout = self.model(self.inputs)
        self.prediction = tf.argmax(self.Qout, 1)

        self.nextQ = tf.placeholder(shape=[None, self.n_classes], dtype=tf.float32)
        self.logger = logger
        self.load_model('weights.h5')
        self.decay = decay
        self.saver = tf.train.Saver()
        logger.debug("Initialize session")
        return

    def fit(self, batches):
        """fit the current estimator based on the expected and
        real rewards of recorded training events"""
        for data in batches:
            newQ = self.sess.run(self.Qout, feed_dict={self.inputs: np.asarray(data['new_states']).reshape(self.n_input[0], self.n_input[1])})
            maxQ = np.amax(newQ, axis=1)
            targetQ = np.asarray(data['oldQ'])
            rewards = np.asarray(data['rewards'])
            n_step = rewards.shape[0]
            decaying_rewards = 0
            for i in range(n_step):
                decaying_rewards += rewards[i] * (self.decay ** i)
            targetQ[np.arange(targetQ.shape[0]), np.asarray(data['actions'])] = decaying_rewards + (
                        self.decay ** n_step) * np.clip(maxQ, -10000, 10000)

            self.model.fit(np.asarray(data['old_states']).reshape(self.n_input[0], self.n_input[1]), targetQ)
        return

    def estimate(self, state, keep = 0.5):
        """estimate the gain of an action, based on the current world state"""
        return self.sess.run([self.prediction, self.Qout], feed_dict = {self.inputs: np.expand_dims(state.flatten(), axis=0)})

    def load_model(self, path):
        """setup the weigths and the like of the model"""

        if path:
            self.logger.debug(f'Loading weigths')
            try:
                self.model.load_weights(path)
            except Exception as e:
                self.logger.debug(f'Couldn\'t load weights. Error message: {e}')
        else:
            self.logger.debug('Initializing with random weights')
        return

    def save_model(self, path = None):
        if path:
            self.logger.debug(f'Writing weights to: {path}')
            try:
                self.model.save_weights(path)
            except Exception as e:
                self.logger.debug(f'Couldn\'t save weights to {path}. Error message: {e}')
        else:
            self.logger.debug('Not saving weights')
        return

def create_state(self):
    coins = self.game_state['coins']
    x,y,_,_ = self.game_state['self']
    arena = self.game_state['arena']
    explosions = self.game_state['explosions']
    if not len(coins):
        coins = [[x,y]]

    self.state = np.zeros(4)

    if arena[x + 1, y] != 0: self.state[0] = 1
    if arena[x - 1, y] != 0: self.state[1] = 1
    if arena[x, y + 1] != 0: self.state[2] = 1
    if arena[x, y - 1] != 0: self.state[3] = 1
    if explosions[x + 1, y] > 0: self.state[0] = 2
    if explosions[x - 1, y] > 0: self.state[1] = 2
    if explosions[x, y + 1] > 0: self.state[2] = 2
    if explosions[x, y - 1] > 0: self.state[3] = 2

    nearest_coin = np.argmin(np.sum(np.abs(np.asarray([x, y]) - np.asarray(coins)), axis=1))
    coindist = np.clip(np.asarray(coins[nearest_coin]) - np.asarray([x, y]), -2, 2)
    self.state = np.append(self.state, coindist)
    return

def choose_rand_action(self, count):
    self.epsilon = self.epsilon*0.995
    actions = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']
    if np.random.rand(1) < self.epsilon:
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
    reward = -1

    #calculate a reward for a given action
    if e.COIN_COLLECTED in events:
        reward += 1000
    if e.KILLED_SELF in events:
        reward -= 20000
    if e.WAITED in events:
        reward -= 10
    if e.INVALID_ACTION in events:
        reward -= 1000
    if self.coindist < old_dist:
        reward += 100
    if self.coindist > old_dist:
        reward -= 10
    self.score += reward
    return reward

def create_train_batch(self, i):
    train_batch = {'old_states': [], 'new_states': [], 'oldQ': [], 'actions': [], 'rewards': []}
    train_batch['old_states'] = self.data['old_states'][i]
    train_batch['new_states'] = self.data['new_states'][i+self.n_step-1]
    train_batch['oldQ'] = self.data['oldQ'][i]
    train_batch['actions'] = self.data['actions'][i]
    train_batch['rewards'] = self.data['rewards'][i:(self.n_step-1+i)]
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
    self.n_step = 7
    self.model = Model(self.logger)
    self.data = {'old_states': [], 'new_states': [], 'oldQ': [], 'actions': [], 'rewards': []}
    self.score = 0
    self.rand_count = 0
    self.coindist = 0
    self.rand_round = 0
    self.train_set =[]
    self.state = np.zeros(6)
    self.epsilon = 0.1

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
    actions = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']
    create_state(self)

    action_code, self.scores = self.model.estimate(self.state)
    self.logger.debug(f"choosing action {action_code[0]} with scores {self.scores[0]}")
    self.next_action = actions[action_code[0]]
    choose_rand_action(self, 1)
    self.last_action = self.next_action

def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    self.logger.debug(f"Game state step: {self.game_state['step']}")
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

    for i in range(len(self.data['old_states']) - (self.n_step-1)):
        self.train_set.append(create_train_batch(self, i))

    self.model.fit(self.train_set)

    self.logger.debug(f'Printing training batch:')
    self.logger.debug(f'chosen action: {self.data["actions"][4]}')
    self.logger.debug(f'rewards of n following steps: {self.data["rewards"][-4:]}')

    self.data = {'old_states': [], 'new_states': [], 'oldQ': [], 'actions': [], 'rewards': []}
    self.model.save_model('weights.h5')
    self.rand_round -= 1
    self.rand_count = 0
