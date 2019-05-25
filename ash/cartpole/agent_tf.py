"""agent.py
Define how the agent act here
"""
import random
import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np
from .model_tf import DQN, DQN_Linear, Brain
from .utils import Transition, ReplayMemory

class DqnAgent():
    def __init__(self,
                 input_shape=(128, 128, 3),
                 action_space=2, 
                 soft_update_ratio=0.01,
                 learning_rate=1e-4,
                 update_every=10,
                 gamma=0.99,
                 memory_capacity=10000,
                 batch_size=128,
                 conv_mode=False
                 ):
        self.action_space = action_space
        self.soft_update_ratio = soft_update_ratio
        self.gamma = gamma
        self.memory = ReplayMemory(int(memory_capacity))
        self.batch_size = batch_size
        self.update_every = update_every
        self.conv_mode = conv_mode
        K.clear_session()
        print(conv_mode)
        if conv_mode:
            self.policy_net = DQN(action_space) # action giver
            self.target_net = DQN(action_space) # action learner
        else:
            self.policy_net = DQN_Linear(action_space)  # action giver
            self.target_net = DQN_Linear(action_space)  # action learner

        self.policy_net.build(input_shape=(1,)+input_shape)
        self.target_net.build(input_shape=(1,)+input_shape)
        self._init_weights_copy() # sync weights at begining
        
        self.brain = Brain(policy_net=self.policy_net, 
                           target_net=self.target_net, 
                           gamma=gamma)
        #self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)
        self.step_done = 0
    
    def act(self, state, eps):
        """ Decide to random act or follow policy"""
        dice = random.random()
        if (dice < eps):
            # Random act
            action = np.random.choice(self.action_space)
        else:
            # Follow policy
            #print(state.shape)
            if self.conv_mode:
                action = self.policy_net.predict(state[np.newaxis, :,:,:])
            else:
                action = self.policy_net.predict(state[np.newaxis, :])
            action = np.argmax(action, axis=1)[0]
            #print("The action: {}".format(action))
        
        return action
    
    def step(self, state, action, next_state, reward, is_done):
        state = state.astype('float32')
        next_state = next_state.astype('float32')
        self.memory.push(state, action, next_state, reward, is_done)
        if len(self.memory) < self.batch_size:
            return False
        experience = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*experience))
        batch = [np.array(batch.state, np.float32),
                 np.array(batch.action, dtype=np.float32),
                 np.array(batch.next_state, dtype=np.float32),
                 np.array(batch.reward, dtype=np.float32),
                 np.array(batch.done, dtype=np.float32)]

        target, estimate = self.brain(batch)
        loss = self.compute_loss(target, estimate)

        grads = tf.gradients(loss, self.brain.trainable_variables)
        _ = self.optimizer.apply_gradients(zip(grads, self.brain.trainable_variables))

        if (self.step_done % self.update_every) == 0:
            self._soft_update()
        return True
    
    def _soft_update(self):
        for pl,tl in zip(self.policy_net.layers, self.target_net.layers):
            pl.set_weights([wp*self.soft_update_ratio+tp*(1.-self.soft_update_ratio) \
                            for wp,tp in zip(pl.get_weights(), tl.get_weights())])
        
    def _init_weights_copy(self):
        for pl,tl in zip(self.policy_net.layers, self.target_net.layers):
            pl.set_weights([tp for wp,tp in zip(pl.get_weights(), tl.get_weights())])

        
    @staticmethod
    def compute_loss(y_true, y_pred):
        #return K.mean(K.square(y_true-y_pred))
        return tf.keras.losses.MSE(y_true, y_pred)
        
