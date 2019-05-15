"""model..py
Define model structure
"""
import tensorflow as tf
from tensorflow.python.keras import layers, models

class DQN(models.Model):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.conv1 = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same")
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.conv2 = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.conv3 = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")
        self.bn3 = layers.BatchNormalization(axis=-1)
        self.flat = layers.Flatten()

        self.head = layers.Dense(units=outputs)

    def call(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = tf.nn.relu(self.bn1(self.conv1(x)))
        x = tf.nn.relu(self.bn2(self.conv2(x)))
        x = tf.nn.relu(self.bn3(self.conv3(x)))
        x = self.flat(x)

        #return tf.nn.softmax(self.head(x))
        return self.head(x)

class Brain(models.Model):
    def __init__(self, policy_net, target_net, gamma):
        super(Brain, self).__init__()
        self.policy_net = policy_net
        self.target_net = target_net
        #self.loss_layer = LossLayer(name="TDerror")
        self.gamma = gamma
        self.policy_net.trainable = False
        
    def call(self, x):
        state, action, next_state, reward, done = x
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        
        bz = K.shape(state)[0]
        q_target_next = tf.gather_nd(self.target_net(next_state), 
                                     tf.stack((tf.range(bz), 
                                              tf.cast(tf.argmax(self.policy_net(state),axis=1), dtype=tf.int32)), axis=1))
        target = reward + (self.gamma * q_target_next * (1. - done))
        estimate = tf.gather_nd(self.policy_net(state),
                                tf.stack((tf.range(bz), 
                                          tf.cast(action, dtype=tf.int32) ), axis=1))
        #loss = self.loss_layer([target, estimate])
        """
        loss = tf.keras.losses.MSE(target, estimate)
        self.add_loss(loss)
        """
        return target, estimate
    
class LossLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(LossLayer, self).__init__(**kwargs)
        
    def call(self, inputs):
        loss = K.square(K.mean(inputs[0]-inputs[1]))
        self.add_loss(loss)
        return inputs    
    
if __name__ == "__main__":
    import numpy as np
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    model = DQN(2)

    array = np.random.random((256, 256, 3))
    out = model(array[np.newaxis, :,:,:])
    model.summary()
