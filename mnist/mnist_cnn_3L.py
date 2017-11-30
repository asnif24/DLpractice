
# coding: utf-8

# In[13]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

get_ipython().magic(u'matplotlib inline')


# In[14]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels


# In[15]:


#for NN layers
class layer:
    def __init__(self, inputs, in_size, out_size, activation_function=None):
#         self.W = tf.Variable(tf.zeros([in_size, out_size]))
        self.W = tf.Variable(tf.random_normal([in_size, out_size]))
        self.b = tf.Variable(tf.zeros([1,out_size]))
        self.Wx_plus_b = tf.matmul(inputs, self.W) + self.b
        self.activation_function = activation_function
    def output(self):
        if self.activation_function == None:
            result = self.Wx_plus_b
        else :
            result = self.activation_function(self.Wx_plus_b)
        return result


# In[16]:


# for convolution
class convolution:
    def __init__(self, shape):
        self.W = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
        self.b = tf.Variable(tf.constant(0.1, shape=[shape[3]]))
    def conv2d(self, inputs, padding='SAME'):
        return tf.nn.conv2d(inputs, self.W, strides=[1,1,1,1], padding=padding)
    def max_pooling_nxn(self, inputs, n):
        return tf.nn.max_pool(inputs, ksize=[1,n,n,1], strides=[1,n,n,1], padding='SAME')
    def conv_and_pooling(self, inputs, activation_function=None, pooling_size=2):
        if activation_function==None:
            h_conv =self.conv2d(inputs)+self.b
        else:
            h_conv = activation_function(self.conv2d(inputs)+self.b)    
        return self.max_pooling_nxn(h_conv, pooling_size)


# In[17]:


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

#for dropout
keep_prob = tf.placeholder(tf.float32)

#dealing eith inputs. 
# -1 for not considering input number of images
# 28,28 size per iamge
# 1 for channel. 1 for grayscale ; 3 for color image
x_image=tf.reshape(x,[-1,28,28,1])


# In[18]:


#AlexNet => 5 convolution layers & 3 fully connected neural network

conv1 = convolution([3,3,1,12])
conv2 = convolution([3,3,12,24])
conv3 = convolution([3,3,24,48])
# conv4 = convolution([3,3,80,80])
# conv5 = convolution([3,3,80,80])

output_conv1 = conv1.conv_and_pooling(x_image, tf.nn.relu)
output_conv2 = conv2.conv_and_pooling(output_conv1, tf.nn.relu)
output_conv3 = conv3.conv_and_pooling(output_conv2, tf.nn.relu, pooling_size=2)
# output_conv4 = conv4.conv_and_pooling(output_conv3, tf.nn.relu, pooling_size=1)
# output_conv5 = conv5.conv_and_pooling(output_conv4, tf.nn.relu, pooling_size=1)

# h_pool_flat = tf.reshape(output_conv2, [-1,7*7*40])
h_pool_flat = tf.reshape(output_conv3, [-1,4*4*48])

# layer1 = layer(h_pool_flat, 7*7*64, 500, tf.nn.relu)
# layer1 = layer(h_pool_flat, 7*7*64, 10, tf.nn.softmax)
# layer1 = layer(h_pool_flat, 7*7*40, 10, tf.nn.softmax)
layer1 = layer(h_pool_flat, 4*4*48, 100, tf.nn.sigmoid)
# layer2 = layer(layer1.output(), 400, 100, tf.nn.sigmoid)
layer3 = layer(layer1.output(), 100, 10, tf.nn.softmax)
# layer2 = layer(layer1.output(), 500, 100, tf.nn.relu)
# layer3 = layer(layer2.output(), 100, 10, tf.nn.softmax)

# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=layer1.output()))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=layer3.output()))
# train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)
# train_step =  tf.train.MomentumOptimizer(0.005 , 0.9).minimize(loss)
train_step = tf.train.AdamOptimizer(0.003).minimize(loss)


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(layer3.output(), 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[19]:


init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

batch_size = 100
batches = x_train.shape[0]//batch_size


# In[20]:


for epoch in range(31):
    print "start "+str(epoch)
    for batch in range(batches):
        sess.run(train_step, feed_dict={x: x_train[batch_size*batch:batch_size*(batch+1)], y: y_train[batch_size*batch:batch_size*(batch+1)]})
    if epoch%2==0:
        print "epoch: "+str(epoch)+" loss: "+str(sess.run(loss, feed_dict={x: x_train, y: y_train}))+" accuracy: "+str(sess.run(accuracy, feed_dict={x: x_train, y: y_train}))
print "Test accuracy: "+str(sess.run(accuracy, feed_dict={x: x_test, y: y_test}))


# In[21]:


for epoch in range(21):
    print "start "+str(epoch)
    for batch in range(batches):
        sess.run(train_step, feed_dict={x: x_train[batch_size*batch:batch_size*(batch+1)], y: y_train[batch_size*batch:batch_size*(batch+1)]})
    if epoch%2==0:
        print "epoch: "+str(epoch)+" loss: "+str(sess.run(loss, feed_dict={x: x_train, y: y_train}))+" accuracy: "+str(sess.run(accuracy, feed_dict={x: x_train, y: y_train}))
print "Test accuracy: "+str(sess.run(accuracy, feed_dict={x: x_test, y: y_test}))


# In[ ]:




