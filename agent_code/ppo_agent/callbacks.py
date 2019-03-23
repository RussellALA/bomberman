"""
A simple version of Proximal Policy Optimization (PPO) using single thread.
Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

from settings import s, e


EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 315, 6
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


class PPO(object):

    def __init__(self, agent):
        self.sess = tf.Session()
        self.agent = agent
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)


        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):

        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]



# for ep in range(EP_MAX):
#     # s = env.reset()
    
#     buffer_s, buffer_a, buffer_r = [], [], []
#     ep_r = 0
#     for t in range(EP_LEN):    # in one episode
#         # env.render()
#         a = ppo.choose_action(s)
#         # s_, r, done, _ = env.step(a)
#         buffer_s.append(s)
#         buffer_a.append(a)
#         buffer_r.append((r+8)/8)    # normalize reward, find to be useful
#         s = s_
#         ep_r += r

#         # update ppo
#         if (t+1) % BATCH == 0 or t == EP_LEN-1:
#             v_s_ = ppo.get_v(s_)
#             discounted_r = []
#             for r in buffer_r[::-1]:
#                 v_s_ = r + GAMMA * v_s_
#                 discounted_r.append(v_s_)
#             discounted_r.reverse()

#             bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
#             buffer_s, buffer_a, buffer_r = [], [], []
#             ppo.update(bs, ba, br)
#     if ep == 0: all_ep_r.append(ep_r)
#     else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
#     print(
#         'Ep: %i' % ep,
#         "|Ep_r: %i" % ep_r,
#         ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
#     )





def setup(self):
    # env = gym.make('Pendulum-v0').unwrapped
    self.ppo = PPO(self)
    self.all_ep_r = []
    self.buffer_s, self.buffer_a, self.buffer_r = [], [], []
    self.ep_r = 0
    self.t = 0
    self.ep = 0

def end_of_episode(self):
    # self.model.d = True
    #self.logger.debug("entering end of episode " + str(self.model.epoch_step_num))

    self.buffer_s, self.buffer_a, self.buffer_r = [], [], []
    self.ep_r = 0
    self.ep += 1
    self.t = 0

def act(self):    
    self.logger.debug("reached act function " + str(self.t))

    # env.render()
    # s_, r, done, _ = env.step(a)
    self.s_ = gen_input_vect(self)
    self.logger.debug(self.ppo.choose_action(self.s_))
    self.next_action = s.actions[int(np.max(self.ppo.choose_action(self.s_)))]
    self.logger.debug("next action is " + str(self.next_action))

    if self.t != 0:
        self.a = self.ppo.choose_action(s)
        self.r = custom_calculate_reward(self)
        self.done = False

        self.buffer_s.append(self.s)
        self.buffer_a.append(self.a)
        self.buffer_r.append((self.r+8)/8)    # normalize reward, find to be useful
        self.ep_r += self.r

    self.s = self.s_

    # update ppo
    if (self.t+1) % BATCH == 0 or self.t == EP_LEN-1:
        v_s_ = self.ppo.get_v(self.s_)
        discounted_r = []
        for r in self.buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()

        bs, ba, br = np.vstack(self.buffer_s), np.vstack(self.buffer_a), np.array(discounted_r)[:, np.newaxis]
        self.buffer_s, self.buffer_a, self.buffer_r = [], [], []
        self.ppo.update(bs, ba, br)

    self.t+= 1
    self.logger.debug("reached end of act function")


def reward_update(self):
    self.logger.debug("reached reward_update function")

    pass
    # Save model
    # self.model.set_trainable = True
    # self.model.r = custom_calculate_reward(agent = self)
    # self.model.ep_ret += self.model.r

    #self.logger.info("reward is " +  str(self.model.ep_ret))
    
def end_of_epoch(self):
    self.logger.debug("reached end_of_epoch function")

    if self.ep == 0: self.all_ep_r.append(self.ep_r)
    else: self.all_ep_r.append(self.all_ep_r[-1]*0.9 + self.ep_r*0.1)
    self.logger.debug(
        'Ep: %i' % self.ep,
        "|Ep_r: %i" % self.ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )

    # after all epochs
    if self.ep == EP_MAX:
        self.logger.debug(str(self.all_ep_r))
        # plt.plot(np.arange(len(self.all_ep_r)), self.all_ep_r)
        # plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()

def custom_calculate_reward(agent):
    agent.logger.debug("reached custom_calculate_reward function")

    #coins = agent.game_state['coins']
    #x, y, _, bombs_left = agent.game_state['self']
    #old_dist = self.coindist
    events = agent.events
    #agent.logger.debug(events)
    #self.coindist = np.min(np.sum(np.abs(np.asarray([x, y]) - np.asarray(coins)), axis = 1))
    reward = 0
    if e.COIN_COLLECTED in events:
        reward += 2
    if e.KILLED_SELF in events:
        reward += -4
        agent.logger.debug("killed myself")

    if e.WAITED in events:
        reward += 0   
    if e.INVALID_ACTION in events:
        reward += 0
    if e.KILLED_OPPONENT in events:
        reward += 10
        agent.logger.debug("killed opponent")
    if e.CRATE_DESTROYED in events:
        reward += 1
        agent.logger.debug("crate destroyed")
    return reward

def gen_input_vect(self):
    self.logger.debug("reached gen_input_vect function")

    input_vect = []
    coins = self.game_state['coins']
    coins_list = np.ones(18) * (-20)
    x, y, _, bombs_left, _ = self.game_state['self']

    explosions = self.game_state['explosions']
    explosion_fields = [0, 0, 0, 0]
    explosion_fields[0] = explosions[x-1][y]
    explosion_fields[1] = explosions[x+1][y]
    explosion_fields[2] = explosions[x][y-1]
    explosion_fields[3] = explosions[x][y+1]

    #others should obtain shape (6)
    others_list = np.ones(6) * 20
    others = self.game_state['others']
    for i in range(len(others)):
        others_list[i*2] = others[i][0] -x
        others_list[i*2 +1] = others[i][1] -y

    arena = list(self.game_state['arena'])
    #self.logger.debug(arena)
    neighbour_fields = [0, 0, 0, 0]
    neighbour_fields[0] = arena[x-1][y]
    neighbour_fields[1] = arena[x+1][y]
    neighbour_fields[2] = arena[x][y-1]
    neighbour_fields[3] = arena[x][y+1]

    #self.logger.debug("self neighbouring fields x-1, x+1, y-1, y+1 " + str(neighbour_fields))

    # coins_list = []
    coins = [list(c) for c in coins]
    closest_coin = (20,20)
    for i in range(len(coins)):
        coins_list[i*2] = coins[i][0]-x
        coins_list[i*2+1] = coins[i][1]-y
        if ((np.abs(coins[i][0] -x ) + np.abs(coins[i][1] - y)) < np.sum(np.abs(closest_coin))):
            closest_coin = (coins[i][0] -x, coins[i][1]-y)
    arena = list(self.game_state['arena'].flatten())


    bombs = self.game_state['bombs']
    bombs_list = np.ones(8) * (-20)
    bombs = [list(b) for b in bombs]
    closest_bomb = (20,20)
    for i in range(len(bombs)):
        bombs_list[i*2] = bombs[i][0]-x
        bombs_list[i*2+1] = bombs[i][1]-y
    
    in_bomb_radius_fields = [0,0,0,0, 0]
    #is player in bomb radius:
    # bombs_tuples = [(i) for i in range(0,2,2)]
    bombs_tuples = [(bombs_list[i], bombs_list[i+1]) for i in range(0,len(bombs_list),2)]
    if (0,0) in bombs_tuples:
        in_bomb_radius_fields[4] = 1
    for i in range(1, s.bomb_power +1):
        pos_i = i
        if(0, pos_i) in bombs_tuples :
            in_bomb_radius_fields[0] = 1
        pos_i = -i
        if(0,pos_i) in bombs_tuples:
            in_bomb_radius_fields[1] = 1
        pos_i = i
        if(pos_i,0) in bombs_tuples:
            in_bomb_radius_fields[2] = 1
        pos_i = -i
        if(pos_i, 0) in bombs_tuples:
            in_bomb_radius_fields[3] = 1
    # self.logger.debug("in bomb radius is " + str(in_bomb_radius_fields))
    # self.logger.debug(" x,y are " + str((x,y)))

    for i in range(len(coins)):
        coins_list[i*2] = coins[i][0]-x
        coins_list[i*2+1] = coins[i][1]-y
        if ((np.abs(coins[i][0] -x ) + np.abs(coins[i][1] - y)) < np.sum(np.abs(closest_coin))):
            closest_coin = (coins[i][0] -x, coins[i][1]-y)
    arena = list(self.game_state['arena'].flatten())
    #input_vect += list(coins_list)
    input_vect += arena
    input_vect += explosion_fields
    input_vect += list(bombs_list)
    input_vect += list(closest_coin)
    input_vect += in_bomb_radius_fields
    input_vect += neighbour_fields
    #input_vect += list(others_list)
    input_vect += [x, y]
    input_vect += [bombs_left]
    #if self.epoch_step_num ==0:
    #self.logger.debug("bombs list " + str(bombs_list))
    #self.logger.debug("explosion fields " + str(explosion_fields))
    return np.array(input_vect)
