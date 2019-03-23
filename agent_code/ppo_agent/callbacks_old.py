
import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import os.path as osp
import tensorflow as tf
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph

#create discrete space 
from gym.spaces import Discrete, Box
from gym import spaces
from settings import s, e

from spinup.utils.run_utils import setup_logger_kwargs


eps = 0.0000000001

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma, lam):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, ), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]



class Model():

    def gen_input_vect(self, agent):
        input_vect = []
        coins = agent.game_state['coins']
        coins_list = np.ones(18) * (-20)
        

        x, y, _, bombs_left, _ = agent.game_state['self']

        explosions = agent.game_state['explosions']
        explosion_fields = [0, 0, 0, 0]
        explosion_fields[0] = explosions[x-1][y]
        explosion_fields[1] = explosions[x+1][y]
        explosion_fields[2] = explosions[x][y-1]
        explosion_fields[3] = explosions[x][y+1]

        #others should obtain shape (6)
        others_list = np.ones(6) * 20
        others = agent.game_state['others']
        for i in range(len(others)):
            others_list[i*2] = others[i][0] -x
            others_list[i*2 +1] = others[i][1] -y

        arena = list(agent.game_state['arena'])
        #agent.logger.debug(arena)
        neighbour_fields = [0, 0, 0, 0]
        neighbour_fields[0] = arena[x-1][y]
        neighbour_fields[1] = arena[x+1][y]
        neighbour_fields[2] = arena[x][y-1]
        neighbour_fields[3] = arena[x][y+1]

        #agent.logger.debug("agent neighbouring fields x-1, x+1, y-1, y+1 " + str(neighbour_fields))

        # coins_list = []
        coins = [list(c) for c in coins]
        closest_coin = (20,20)
        for i in range(len(coins)):
            coins_list[i*2] = coins[i][0]-x
            coins_list[i*2+1] = coins[i][1]-y
            if ((np.abs(coins[i][0] -x ) + np.abs(coins[i][1] - y)) < np.sum(np.abs(closest_coin))):
                closest_coin = (coins[i][0] -x, coins[i][1]-y)
        arena = list(agent.game_state['arena'].flatten())


        bombs = agent.game_state['bombs']
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
        # agent.logger.debug("in bomb radius is " + str(in_bomb_radius_fields))
        # agent.logger.debug(" x,y are " + str((x,y)))

        for i in range(len(coins)):
            coins_list[i*2] = coins[i][0]-x
            coins_list[i*2+1] = coins[i][1]-y
            if ((np.abs(coins[i][0] -x ) + np.abs(coins[i][1] - y)) < np.sum(np.abs(closest_coin))):
                closest_coin = (coins[i][0] -x, coins[i][1]-y)
        arena = list(agent.game_state['arena'].flatten())
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
        #agent.logger.debug("bombs list " + str(bombs_list))
        #agent.logger.debug("explosion fields " + str(explosion_fields))
        return np.array(input_vect)


    def __init__(self, agent):

        self.set_trainable = False
        self.bomberman_action_space = spaces.Discrete(6)
        self.bomberman_observation_space = spaces.Box(low=-20, high=20, shape=(315,), dtype=np.uint8)
        #agent.logger.info("act dim is " + str(self.bomberman_action_space.shape) + str(self.bomberman_observation_space.shape))
        self.obs_dim = self.bomberman_observation_space.shape[0]
        self.act_dim = 1
        self.start_time = time.time()

        self.r = 0
        self.ep_ret = 0
        self.ep_len = 0
        self.d = False

        self.epoch =0
        self.epoch_step_num = 0

        self.hid = 128
        self.l = 2
        #self.epochs=50
        self.steps_per_epoch=800
        self.seed = 0
        self.gamma=0.95
        self.cpu = 4
        self.exp_name = "ppo"
        self.logger_kwargs = setup_logger_kwargs(self.exp_name, self.seed)
        self.actor_critic= core.mlp_actor_critic 

    
        self.ac_kwargs= {'hidden_sizes': [self.hid]*self.l, 'action_space': None}
        
        # #epochs will be ignored - set value at settings.py
        self.clip_ratio=0.2
        self.pi_lr=3e-4
        self.vf_lr=1e-3
        self.train_pi_iters=80
        self.train_v_iters=80
        self.lam=0.93
        self.max_ep_len=200
        self.target_kl=0.01
        self.logger_kwargs=dict()
        self.save_freq=10

        self.logger = EpochLogger(**self.logger_kwargs)
        #locals returns dict with locals - every item except 'agent' should be saved, in this case only 'self'
        self.logger.save_config(locals()['self'])

        self.seed += 10000 * proc_id()
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

        # Share information about action space with policy architecture
        self.ac_kwargs['action_space'] = self.bomberman_action_space

        # Inputs to computation graph
        self.x_ph, self.a_ph = core.placeholders_from_spaces(self.bomberman_observation_space , self.bomberman_action_space)
        self.adv_ph, self.ret_ph, self.logp_old_ph = core.placeholders(None, None, None)

        # Main outputs from computation graph
        self.pi, self.logp, self.logp_pi, self.v = self.actor_critic(self.x_ph, self.a_ph, **self.ac_kwargs)

        # Need all placeholders in *this* order later (to zip with data from buffer)
        self.all_phs = [self.x_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph]

        # Every step, get: action, value, and logprob
        self.get_action_ops = [self.pi, self.v, self.logp_pi]

        # Experience buffer
        self.local_steps_per_epoch = int(self.steps_per_epoch / num_procs())
        agent.logger.debug("local_steps_per_epoch is " + str(self.local_steps_per_epoch))
        agent.logger.debug("local_steps_per_epoch is " + str(self.steps_per_epoch))
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, self.local_steps_per_epoch, self.gamma, self.lam)

        # Count variables
        self.var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%self.var_counts)
        agent.logger.info('\nNumber of parameters: \t pi: %d, \t v: %d\n'%self.var_counts)

        # PPO objectives
        self.ratio = tf.exp(self.logp - self.logp_old_ph)          # pi(a|s) / pi_old(a|s)
        self.min_adv = tf.where(self.adv_ph>0, (1+self.clip_ratio)*self.adv_ph, (1-self.clip_ratio)*self.adv_ph)


        self.pi_loss = -tf.reduce_mean(tf.minimum(self.ratio * self.adv_ph, self.min_adv))
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

        # Info (useful to watch during learning)
        self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp)      # a sample estimate for KL-divergence, easy to compute
        self.approx_ent = tf.reduce_mean(-self.logp)                  # a sample estimate for entropy, also easy to compute
        self.clipped = tf.logical_or(self.ratio > (1+self.clip_ratio), self.ratio < (1-self.clip_ratio))
        self.clipfrac = tf.reduce_mean(tf.cast(self.clipped, tf.float32))

        # Optimizers
        self.train_pi = MpiAdamOptimizer(learning_rate=self.pi_lr).minimize(self.pi_loss)
        self.train_v = MpiAdamOptimizer(learning_rate=self.vf_lr).minimize(self.v_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Sync params across processes
        self.sess.run(sync_all_params())
    
        # Setup model saving
        self.logger.setup_tf_saver(self.sess, inputs={'x': self.x_ph}, outputs={'pi': self.pi, 'v': self.v})

        if(False and not agent.train_flag):
            agent.logger.debug("loading model from file")
            fpath = "pathname"
            self.get_action = load_policy(fpath)
            
        

    def update(self, agent):
        self.inputs = {k:v for k,v in zip(self.all_phs, self.buf.get())}
        self.pi_l_old, self.v_l_old, self.ent = self.sess.run([self.pi_loss, self.v_loss, self.approx_ent], feed_dict=self.inputs)

        # Training
        for i in range(self.train_pi_iters):
            _, self.kl = self.sess.run([self.train_pi, self.approx_kl], feed_dict=self.inputs)
            self.kl = mpi_avg(self.kl)
            agent.logger.debug("taking training step i: " + str(i))
            if self.kl > 1.5 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
        self.logger.store(StopIter=i)
        for _ in range(self.train_v_iters):
            self.sess.run(self.train_v, feed_dict=self.inputs)

        # Log changes from update
        self.pi_l_new, self.v_l_new, self.kl, self.cf = self.sess.run([self.pi_loss, self.v_loss, self.approx_kl, self.clipfrac], feed_dict=self.inputs)
        self.logger.store(LossPi=self.pi_l_old, LossV=self.v_l_old, 
                    KL=self.kl, Entropy=self.ent, ClipFrac=self.cf,
                    DeltaLossPi=(self.pi_l_new - self.pi_l_old),
                    DeltaLossV=(self.v_l_new - self.v_l_old))

        self.o, self.r, self.d, self.ep_ret, self.ep_len = None, 0, False, 0, 0

        self.epoch += 1

    def load_policy(fpath, itr='last', deterministic=True):

        # handle which epoch to load from
        if itr=='last':
            saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
            itr = '%d'%max(saves) if len(saves) > 0 else ''
        else:
            itr = '%d'%itr

        # load the things!
        sess = tf.Session()
        model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

        # get the correct op for executing actions
        if deterministic and 'mu' in model.keys():
            # 'deterministic' is only a valid option for SAC policies
            print('Using deterministic action op.')
            action_op = model['mu']
        else:
            print('Using default action op.')
            action_op = model['pi']

        # make function for producing an action given a single state
        get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

        # try to load environment from save
        # (sometimes this will fail because the environment could not be pickled)
        try:
            state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
            env = state['env']
        except:
            env = None

        return env, get_action

def end_of_episode(self):
    self.model.d = True

    #self.logger.debug("entering end of episode " + str(self.model.epoch_step_num))

    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """


def setup(self):
    self.model = Model(self)

def act(self):
    self.model.epoch_step_num +=1
    o = self.model.gen_input_vect(self)
    
    a, v_t, logp_t = self.model.sess.run(self.model.get_action_ops, feed_dict={self.model.x_ph: o.reshape(1,-1)})

    #tic = time.time()
    self.model.buf.store(o, a, self.model.r, v_t, logp_t)
    self.model.logger.store(VVals=v_t)

    #toc = time.time()
    #self.logger.debug("time taken for storing buffer " + str(toc-tic))

    
    #self.model.r = custom_calculate_reward(agent = self)
    #self.model.ep_ret += self.model.r
    self.model.ep_len += 1

    #self.model.t = self.model.epoch_step_num
    # if self.next_action == 'BOMB':
    #     self.next_action = 'WAIT'
    # self.logger.info("a is now " + str(a))
    #o, r, d, _ = env.step(a[0])
    self.next_action = s.actions[a[0]]


    if (self.model.epoch_step_num >= self.model.steps_per_epoch ):
        end_of_epoch(self)
        #self.logger.debug("end of epoch called")
    
    terminal = self.model.d or (self.model.ep_len == self.model.max_ep_len)
    if terminal or (self.model.epoch_step_num==self.model.local_steps_per_epoch-1):
        #self.logger.debug("entering act -epoch step num is now " + str(self.model.epoch_step_num))

        if not(terminal):
            print('Warning: trajectory cut off by epoch at %d steps.'%self.model.ep_len)
        # if trajectory didn't reach terminal state, bootstrap value target
        last_val = self.model.r if self.model.d else self.model.sess.run(self.model.v, feed_dict={self.model.x_ph: o.reshape(1,-1)})
        self.model.buf.finish_path(last_val)
        if terminal:
            # only save EpRet / EpLen if trajectory finished
            self.model.logger.store(EpRet=self.model.ep_ret, EpLen=self.model.ep_len)
        self.logger.debug("ep_ret is " + str(self.model.ep_ret))
        
        o, self.model.r, self.model.d, self.model.ep_ret, self.model.ep_len = self.model.gen_input_vect(self), 0, False, 0, 0


    #model.d can be set to True by end_of_episode()
    if self.model.d == True:
        self.model.d = False


def reward_update(self):
    # Save model
    self.model.set_trainable = True
    self.model.r = custom_calculate_reward(agent = self)
    self.model.ep_ret += self.model.r

    #self.logger.info("reward is " +  str(self.model.ep_ret))
    
def end_of_epoch(self):
    self.logger.debug("entering end of epoch " + str(self.model.epoch_step_num))
    #this block will be executed as often as the outer for loop (loop over epochs) from vpg orig
    # if (self.model.epoch % self.model.save_freq == 0) or (self.model.epoch == self.model.epochs-1):
    if (self.model.epoch % self.model.save_freq == 0):
        o = self.model.gen_input_vect(self)

        self.model.logger.save_state({'env': o}, None)
    if(self.model.set_trainable):
        self.model.update(self)
    self.model.epoch_step_num = 0
    
    #Log info about epoch
    self.model.logger.log_tabular('Epoch', self.model.epoch)
    self.model.logger.log_tabular('EpRet', with_min_and_max=True)
    self.model.logger.log_tabular('EpLen', average_only=True)
    self.model.logger.log_tabular('VVals', with_min_and_max=True)
    self.model.logger.log_tabular('TotalEnvInteracts', (self.model.epoch+1)*self.model.steps_per_epoch)
    self.model.logger.log_tabular('LossPi', average_only=True)
    self.model.logger.log_tabular('LossV', average_only=True)
    self.model.logger.log_tabular('DeltaLossPi', average_only=True)
    self.model.logger.log_tabular('DeltaLossV', average_only=True)
    self.model.logger.log_tabular('Entropy', average_only=True)
    self.model.logger.log_tabular('KL', average_only=True)
    self.model.logger.log_tabular('ClipFrac', average_only=True)
    self.model.logger.log_tabular('StopIter', average_only=True)
    self.model.logger.log_tabular('Time', time.time()-self.model.start_time)
    self.model.logger.dump_tabular()

def custom_calculate_reward(agent):
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
