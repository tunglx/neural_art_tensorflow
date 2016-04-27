import tensorflow as tf
import numpy as np

# Define equation to compute models
class Network(object):
    def __init__(self, input, params_path):                
        self.params = np.load(params_path).item()
        self.vars = []
        self.vardict = {}
        self.batch_size = int(input.get_shape()[0])
        self.add_('input', input)
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def get_unique_name_(self, prefix):        
        id = sum(t.startswith(prefix) for t,_ in self.vars)+1
        return '%s_%d'%(prefix, id)

    def add_(self, name, var):
        self.vars.append((name, var))
        self.vardict[name] = var

    def get_output(self):
        return self.vars[-1][1]

    def conv(self, h, w, c_i, c_o, stride=1, name=None):
        name = name or self.get_unique_name_('conv')
        with tf.variable_scope(name) as scope:
            weights = self.params[name][0].astype(np.float32)
            conv = tf.nn.conv2d(self.get_output(), weights, [stride]*4, padding='SAME')
            if len(self.params[name]) > 1:
                biases = self.params[name][1].astype(np.float32)
                bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
                relu = tf.nn.relu(bias, name=scope.name)
            else:
                relu = tf.nn.relu(conv, name=scope.name)            
            self.add_(name, relu)
        return self

    def conv_alexnet(self, h, w, c_i, c_o, stride=4, name=None):
        name = name or self.get_unique_name_('conv')
        with tf.variable_scope(name) as scope:
            weights = self.params[name][0].astype(np.float32)
            conv = tf.nn.conv2d(self.get_output(), weights, [stride]*4, padding='SAME')
            if len(self.params[name]) > 1:
                biases = self.params[name][1].astype(np.float32)
                bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
                relu = tf.nn.relu(bias, name=scope.name)
            else:
                relu = tf.nn.relu(conv, name=scope.name)            
            self.add_(name, relu)
        return self

    def pool(self, size=2, stride=2, name=None):
        name = name or self.get_unique_name_('pool')
        # pool = tf.nn.avg_pool(self.get_output(),
        pool = tf.nn.max_pool(self.get_output(),
                              ksize=[1, size, size, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME',
                              name=name)
        self.add_(name, pool)
        return self


# Define convolutional models

class VGG16(Network):
    alpha = [0, 0, 0, 1, 1]
    beta  = [1, 1, 1, 1, 1]
    def setup(self):
        (self.conv(3, 3,   3,  64, name='conv1_1')
             .conv(3, 3,  64,  64, name='conv1_2')
             .pool()
             .conv(3, 3,  64, 128, name='conv2_1')
             .conv(3, 3, 128, 128, name='conv2_2')
             .pool()
             .conv(3, 3, 128, 256, name='conv3_1')
             .conv(3, 3, 256, 256, name='conv3_2')
             .conv(3, 3, 256, 256, name='conv3_3')
             .pool()
             .conv(3, 3, 256, 512, name='conv4_1')
             .conv(3, 3, 512, 512, name='conv4_2')
             .conv(3, 3, 512, 512, name='conv4_3')
             .pool()
             .conv(3, 3, 512, 512, name='conv5_1')
             .conv(3, 3, 512, 512, name='conv5_2')
             .conv(3, 3, 512, 512, name='conv5_3')
             .pool())

    def y(self):
        return [self.vardict['conv1_2'], self.vardict['conv2_2'], self.vardict['conv3_3']]

class I2V(Network):
    alpha = [0,0,1,1,10]
    beta  = [0.1,1,10,10,100]
    def setup(self):
        (self.conv(3, 3,   3,  64, name='conv1_1')
             .pool()
             .conv(3, 3,  64, 128, name='conv2_1')
             .pool()
             .conv(3, 3, 128, 256, name='conv3_1')
             .conv(3, 3, 256, 256, name='conv3_2')
             .pool()
             .conv(3, 3, 256, 512, name='conv4_1')
             .conv(3, 3, 512, 512, name='conv4_2')
             .pool()
             .conv(3, 3, 512, 512, name='conv5_1')
             .conv(3, 3, 512, 512, name='conv5_2')
             .pool())

    def y(self):
        return [self.vardict['conv1_1'], self.vardict['conv2_1'], self.vardict['conv3_2'], self.vardict['conv4_2'], self.vardict['conv5_2']]

class Alexnet(Network):
    alpha = [0,0,1,1,10]
    beta  = [0.1,1,10,10,100]
    def setup(self):
        (self.conv_alexnet(11, 11, 3, 96, name='conv1')
             .pool()
             .conv_alexnet(5, 5,  96, 256, name='conv2')
             .pool()
             .conv_alexnet(3, 3, 256, 384, name='conv3')
             .conv_alexnet(3, 3, 384, 384, name='conv4')
             .conv_alexnet(3, 3, 384, 256, name='conv5')
             .pool())

    def y(self):
        return [self.vardict['conv1'], self.vardict['conv2'], self.vardict['conv3'], self.vardict['conv4'], self.vardict['conv5']]