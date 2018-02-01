import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels
batch_size = 256
g_dim = 100

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding = 'SAME')
def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 2, 2, 1], padding = 'SAME')

class layer:
    def __init__(self, in_size, out_size):
        self.W = tf.Variable(tf.random_normal([in_size, out_size], mean=0.0, stddev=0.1))
        self.b = tf.Variable(tf.random_normal([1, out_size], mean=0.0, stddev=0.1))
    def output(self, inputs, activation_function=None):
        if activation_function == None:
            return tf.matmul(inputs, self.W) + self.b
        else :
            return activation_function(tf.matmul(inputs, self.W) + self.b)

def weight_variable(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial)

layer_e = layer(4*4*64, g_dim)
layer_d = layer(g_dim, 4*4*64)

weights = {
    "W_e_conv1" : weight_variable([3,3, 1,16]),
    "W_e_conv2" : weight_variable([3,3,16,32]),
    "W_e_conv3" : weight_variable([3,3,32,64]),
    "W_d_conv1" : weight_variable([3,3,32,64]),
    "W_d_conv2" : weight_variable([3,3,16,32]),
    "W_d_conv3" : weight_variable([3,3, 1,16])
}

biases = {
    "b_e_conv1" : bias_variable([16]),   
    "b_e_conv2" : bias_variable([32]),
    "b_e_conv3" : bias_variable([64]),
    "b_d_conv1" : bias_variable([32]),
    "b_d_conv2" : bias_variable([16]),
    "b_d_conv3" : bias_variable([1])
}


#var_d = [weights["W_e_conv1"], weights["W_e_conv2"], weights["W_d_conv1"], weights["W_d_conv2"], biases["b_e_conv1"], biases["b_e_conv2"], biases["b_d_conv1"], biases["b_d_conv2"]]
#var_g = [weights["w_g1"], weights["w_g2"],weights["w_g3"], weights["w_g4"], biases["b_g1"], biases["b_g2"],biases["b_g3"], biases["b_g4"]]

var_d = [weights[w] for w in weights]
var_g = [biases[b] for b in biases]

def encoder(x):
    x_origin = tf.reshape(x, [-1,28,28,1])      #28x28x1
    h_e_conv1 = tf.nn.relu(tf.add(conv2d(x_origin, weights["W_e_conv1"]), biases["b_e_conv1"]))     #14x14x16
    h_e_conv2 = tf.nn.relu(tf.add(conv2d(h_e_conv1, weights["W_e_conv2"]), biases["b_e_conv2"]))    #7x7x32
    h_e_conv3 = tf.nn.relu(tf.add(conv2d(h_e_conv2, weights["W_e_conv3"]), biases["b_e_conv3"]))    #4x4x64
    h_e_conv3_reshape = tf.reshape(h_e_conv3, [-1,4*4*64])
    h_e_layer = layer_e.output(h_e_conv3_reshape, tf.nn.relu)
    return h_e_layer
    
def decoder(z):
    h_d_layer = layer_d.output(z, tf.nn.relu)
    h_d_layer_reshape = tf.reshape(h_d_layer, [-1,4,4,64])
    
    output_shape_d_conv1 = tf.stack([tf.shape(z)[0], 7, 7, 32])
    h_d_conv1 = tf.nn.relu(deconv2d(h_d_layer_reshape, weights["W_d_conv1"], output_shape_d_conv1)+biases["b_d_conv1"])

    output_shape_d_conv1 = tf.stack([tf.shape(z)[0], 14, 14, 16])
    h_d_conv2 = tf.nn.relu(deconv2d(h_d_conv1, weights["W_d_conv2"], output_shape_d_conv1)+biases["b_d_conv2"])

    output_shape_d_conv2 = tf.stack([tf.shape(z)[0], 28, 28, 1])
    h_d_conv3 = tf.nn.relu(deconv2d(h_d_conv2, weights["W_d_conv3"], output_shape_d_conv2)+biases["b_d_conv3"])
    return h_d_conv3

def generator(z):
    return decoder(z)

def discriminator(x):
    return decoder(encoder(x))

x_d = tf.placeholder(tf.float32, shape = [None, 784])
x_g = tf.placeholder(tf.float32, shape = [None, g_dim])

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def loss(x):
    return tf.reduce_mean(tf.pow(tf.reshape(x, [-1, 784]) - tf.reshape(discriminator(x), [-1, 784]), 2))
#    return tf.pow(tf.reshape(x, [-1, 784]) - tf.reshape(discriminator(x), [-1, 784]), 2)
#     return tf.pow(x-discriminator(x), 2)

gamma = 0.5

k_t = 0

d_loss = tf.reduce_mean(loss(x_d))
g_loss = tf.reduce_mean(loss(generator(x_g)))

g_sample = generator(x_g)

M_global = tf.reduce_mean(loss(x_d) + tf.abs(gamma*loss(x_d) - loss(generator(x_g))))

d_optimizer = tf.train.AdamOptimizer(0.00001).minimize(d_loss, var_list= var_d)
g_optimizer = tf.train.AdamOptimizer(0.00001).minimize(g_loss, var_list= var_g)

balancer = gamma*loss(x_d) - loss(generator(x_g))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(10001):
    batch_x = mnist.train.next_batch(batch_size)[0]
    sess.run(d_optimizer, feed_dict={x_d: batch_x, x_g: sample_Z(batch_size, g_dim)})
    sess.run(g_optimizer, feed_dict={x_g: sample_Z(batch_size, g_dim)})
    k_t = k_t + 0.001*(sess.run(balancer, feed_dict={x_d: batch_x, x_g: sample_Z(batch_size, g_dim)}))
    if step%1000==0:
        d_loss_train, g_loss_train, M_global_train = sess.run([d_loss, g_loss, M_global], feed_dict=
                            {x_d: batch_x, x_g: sample_Z(batch_size, g_dim)})
        print 'step:', step, ' d-loss:', d_loss_train, ' g-loss:', g_loss_train, 'k_t:', k_t,"M_global:", M_global_train
  

zz = sample_Z(batch_size, g_dim)
gg = sess.run(g_sample, feed_dict = {x_g: zz})
gg_pic = np.array([np.reshape(m,(28,28)) for m in gg])
fig, ax = plt.subplots(nrows=3, ncols=3)
for i,row in enumerate(ax):
    for j,col in enumerate(row):
        ax[i][j].imshow(gg_pic[i*3+j], cmap='gray')
