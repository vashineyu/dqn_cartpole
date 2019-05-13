"""model..py
Define model structure
"""
import tensorflow as tf
from tensorflow.python.keras import layers, models

class DQN(models.Model):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=2, padding="same")
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.conv2 = layers.Conv2D(filters=32, kernel_size=5, strides=2, padding="same")
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.conv3 = layers.Conv2D(filters=32, kernel_size=5, strides=2, padding="same")
        self.bn3 = layers.BatchNormalization(axis=-1)
        self.flat = layers.Flatten()

        self.head = layers.Dense(units=outputs)

    def call(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = tf.nn.relu(self.bn1(self.conv1(x)))
        x = tf.nn.relu(self.bn2(self.conv2(x)))
        x = tf.nn.relu(self.bn3(self.conv3(x)))
        x = self.flat(x)

        return tf.nn.softmax(self.head(x))

if __name__ == "__main__":
    import numpy as np
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    model = DQN(2)

    array = np.random.random((256, 256, 3))
    out = model(array[np.newaxis, :,:,:])
    model.summary()
