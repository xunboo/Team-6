from utils import *
import tensorflow as tf
import random
import pdb
import os

layers = tf.contrib.layers
BATCH_SIZE = 16

class BaselineModel():
    def __init__(self):
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.build_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    # from https://github.com/dalgu90/resnet-18-tensorflow/blob/master/resnet.py
    def _residual_block(self, x, input_q=None, output_q=None, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = layers.conv2d(x, filters=num_channel, kernel_size=3, padding='same')
            x = layers.batch_normalization(x)
            x = tf.nn.relu(x)

            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name='conv_2')
            x = self._bn(x, name='bn_2')

            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x

    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None,32,32,1])
        self.y = tf.placeholder(tf.float32, shape=[None,32,32,3])
        x_float = self.x
        y_float = self.y
        #x_float = tf.cast(self.x, tf.float32)
        #y_float = tf.cast(self.y, tf.float32)

        with tf.name_scope("conv"):
            e1 = layers.conv2d(inputs=x_float, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu, padding='same', stride=2)
            e2 = layers.conv2d(inputs=e1, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu, padding='same', stride=2)
            e3 = layers.conv2d(inputs=e2, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu, padding='same', stride=2)
            e4 = layers.conv2d(inputs=e3, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu, padding='same', stride=2)
            e5 = layers.conv2d(inputs=e4, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu, padding='same', stride=2)

            g1 = layers.conv2d_transpose(inputs=e5, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu, padding='same', stride=2)
            g2 = layers.conv2d_transpose(inputs=g1, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu, padding='same', stride=2)
            g3 = layers.conv2d_transpose(inputs=g2, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu, padding='same', stride=2)
            g4 = layers.conv2d_transpose(inputs=g3, num_outputs=32, kernel_size=3, activation_fn=tf.nn.relu, padding='same', stride=2)
            g5 = layers.conv2d_transpose(inputs=g4, num_outputs=3, kernel_size=3, activation_fn=tf.nn.relu, padding='same', stride=2)

            '''
            h4 = layers.conv2d(inputs=h3, num_outputs=100, kernel_size=3, activation_fn=tf.nn.relu, padding='same')
            h5 = layers.conv2d(inputs=h4, num_outputs=60, kernel_size=3, activation_fn=tf.nn.relu, padding='same')
            h6 = layers.conv2d(inputs=h5, num_outputs=20, kernel_size=3, activation_fn=tf.nn.relu, padding='same')
            '''
            self.output = layers.conv2d(inputs=g5, num_outputs=3, kernel_size=3, activation_fn=None, padding='same')
            self.output_image = tf.cast(self.output, tf.int32)

            self.loss = tf.nn.l2_loss(self.output - y_float)  / tf.cast(tf.size(self.output), tf.float32)
            self.train_step = layers.optimize_loss(
                                self.loss,
                                global_step=self.global_step,
                                learning_rate=0.0001,
                                optimizer=tf.train.MomentumOptimizer(0.00001,0.9))

    def train(self):
        data = load_cifar()
        output_dir = 'outputs'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for ep in range(1000):
            loss_sum = 0
            for i in range(100):
                input_feed = {}
                batch_data = data[random.sample(range(data.shape[0]), BATCH_SIZE)]
                #batch_data = data[range(BATCH_SIZE)]
                bw_data = np.stack([np.expand_dims(convert_bw(batch_data[i]), 3) for i in range(BATCH_SIZE)])

                input_feed[self.x] = bw_data
                input_feed[self.y] = batch_data
                loss, _ = self.sess.run([self.loss, self.train_step], input_feed)
                loss_sum += loss
            print('ep:\t', ep, '\tloss:\t', loss_sum)

            # inference
            input_feed = {}
            bw_data = np.stack([np.expand_dims(convert_bw(data[i]), 3) for i in range(10)])
            input_feed[self.x] = bw_data
            input_feed[self.y] = data[range(10)]
            images = self.sess.run([self.output], input_feed)[0]
            for i in range(10):
                save_arr(images[i], 'outputs/output_' + str(ep) + '_' + str(i) + '.png')

if __name__ == '__main__':
    model = BaselineModel()
    model.train()
