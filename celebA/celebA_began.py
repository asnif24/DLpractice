import tensorflow as tf
import numpy as np
import os
import imageio
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

ls = np.array(os.listdir("./img_align_celeba"))
pic_dim = 116412
batch_size = 112
z_dim = 64
hidden_dim = 128
gamma = 0.5
data_directory = "./img_align_celeba/"
target_dir = 'began_pic/pic0214/'

class Plotter(object):
    def __init__(self):
        pass
    def plot(self, pic, D_pic):
        fig = plt.figure(figsize=(8,2))
        for i in range(10):
            plt.subplot(2, 10, i+1)
            plt.imshow(pic[i])
            plt.axis('off')
            plt.subplot(2, 10, i+1+10)
            plt.imshow(D_pic[i])
            plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
    def plotAndSaveRealPic(self, step, pic, D_pic):
        self.plot(pic, D_pic)
        plt.savefig(target_dir+str(step)+'_real_Dreal_.png', bbox_inches='tight', pad_inches = 0)
    def plotAndSaveFakePic(self, step, pic, D_pic):
        self.plot(pic, D_pic)
        plt.savefig(target_dir+str(step)+'_fake_Dfake_.png', bbox_inches='tight', pad_inches = 0)

class  VariableDefiner(object):
    def __init__(self):
        super(VariableDefiner, self).__init__()
    def weight_variable(self, shape):
        initial = tf.random_normal(shape, mean=0.0, stddev=0.1)
        return tf.Variable(initial)
    def bias_variable(self, shape):
        initial = tf.random_normal(shape, mean=0.0, stddev=0.1)
        return tf.Variable(initial)

class Layer(VariableDefiner):
    def __init__(self, in_size, out_size):
        super(Layer, self).__init__()
        self.W = self.weight_variable([in_size, out_size])
        self.b = self.weight_variable([1, out_size])
        # self.W = tf.Variable(tf.random_normal([in_size, out_size], mean=0.0, stddev=0.1))
        # self.b = tf.Variable(tf.random_normal([1, out_size], mean=0.0, stddev=0.1))
    def output(self, inputs, activation_function=None):
        if activation_function == None:
            return tf.matmul(inputs, self.W) + self.b
        else :
            return activation_function(tf.matmul(inputs, self.W) + self.b)
    def getVariables(self):
        return [self.W, self.b]



class DataLoader(object):
    def __init__(self, data_directory = data_directory):
        self.data_directory = data_directory
    def generateBatch(self, batch_size = 100):
        return [np.reshape(imageio.imread(self.data_directory+x), 116412)/255. for x in ls[[(np.random.sample(batch_size)*202599).astype(int)]]]

class Coder(object):
    def __init__(self):
        super(Coder, self).__init__()
    def conv2d(self, x, W, s):
        return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding = 'SAME')
    def deconv2d(self, x, W, output_shape, s):
        return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, s, s, 1], padding = 'SAME')
    def getWeightVariables(self):
        return [self.conv_var_W[w] for w in self.conv_var_W]
    def getBiasVariables(self):
        return [self.conv_var_b[b] for b in self.conv_var_b]
    def getLayerVariables(self):
        return self.layer1.getVariables()+self.layer2.getVariables()
    def getCoderVariables(self):
        return self.getWeightVariables()+self.getBiasVariables()+self.getLayerVariables()

class Encoder(Coder, VariableDefiner):
    def __init__(self, output_dim = hidden_dim):
        super(Encoder, self).__init__()
        self.conv_var_W = {
            "conv1" : self.weight_variable([3,3, 3,24]),
            "conv2" : self.weight_variable([3,3,24,24]),
            "conv3" : self.weight_variable([3,3,24,48]),
            "conv4" : self.weight_variable([3,3,48,48]),
            "conv5" : self.weight_variable([3,3,48,72]),
            "conv6" : self.weight_variable([3,3,72,72])
        }
        self.conv_var_b = {
            "conv1" : self.bias_variable([24]),   
            "conv2" : self.bias_variable([24]),
            "conv3" : self.bias_variable([48]),
            "conv4" : self.bias_variable([48]),   
            "conv5" : self.bias_variable([72]),
            "conv6" : self.bias_variable([72])
        }
        self.layer1 = Layer(28*23*72, 1024)
        self.layer2 = Layer(1024, output_dim)

    def encoderOutput(self, x):
        x_origin = tf.reshape(x, [-1,218,178,3])
        h_e_conv1 = tf.nn.relu(tf.add(self.conv2d(x_origin, self.conv_var_W["conv1"], 1), self.conv_var_b["conv1"]))     
        h_e_conv2 = tf.nn.relu(tf.add(self.conv2d(h_e_conv1, self.conv_var_W["conv2"], 2), self.conv_var_b["conv2"]))    
        
        h_e_conv3 = tf.nn.relu(tf.add(self.conv2d(h_e_conv2, self.conv_var_W["conv3"], 1), self.conv_var_b["conv3"]))    
        h_e_conv4 = tf.nn.relu(tf.add(self.conv2d(h_e_conv3, self.conv_var_W["conv4"], 2), self.conv_var_b["conv4"]))    
        
        h_e_conv5 = tf.nn.relu(tf.add(self.conv2d(h_e_conv4, self.conv_var_W["conv5"], 1), self.conv_var_b["conv5"]))    
        h_e_conv6 = tf.nn.relu(tf.add(self.conv2d(h_e_conv5, self.conv_var_W["conv6"], 2), self.conv_var_b["conv6"]))    
        
        h_e_conv6_reshape = tf.reshape(h_e_conv6, [-1,28*23*72])
        h_e_layer1 = self.layer1.output(h_e_conv6_reshape, tf.nn.relu)
        h_e_layer2 = self.layer2.output(h_e_layer1, tf.nn.sigmoid)
        return h_e_layer2
    # def getWeightVariables(self):
    #     return [self.conv_var_W[w] for w in self.conv_var_W]
    # def getBiasVariables(self):
    #     return [self.conv_var_b[w] for b in self.conv_var_b]
    # def getLayerVariables(self):
    #     return self.layer1.getVariables()+self.layer2.getVariables()
    # def getEncoderVariables(self):
    #     return self.getWeightVariables()+self.getBiasVariables()+self.getLayerVariables()

class Decoder(Coder, VariableDefiner):
    def __init__(self, input_dim = hidden_dim):
        super(Decoder, self).__init__()
        self.conv_var_W = {
            "conv1" : self.weight_variable([3,3,72,72]),
            "conv2" : self.weight_variable([3,3,48,72]),
            "conv3" : self.weight_variable([3,3,48,48]),
            "conv4" : self.weight_variable([3,3,24,48]),
            "conv5" : self.weight_variable([3,3,24,24]),
            "conv6" : self.weight_variable([3,3, 3,24])
        }
        self.conv_var_b = {
            "conv1" : self.bias_variable([72]),
            "conv2" : self.bias_variable([48]),
            "conv3" : self.bias_variable([48]),
            "conv4" : self.bias_variable([24]),
            "conv5" : self.bias_variable([24]),
            "conv6" : self.bias_variable([3])
        }
        self.layer1 = Layer(input_dim, 1024)
        self.layer2 = Layer(1024, 28*23*72)

    def decoderOutput(self, z):
        h_d_layer1 = self.layer1.output(z, tf.nn.relu)
        h_d_layer2 = self.layer2.output(h_d_layer1, tf.nn.sigmoid)

        h_d_layer_reshape = tf.reshape(h_d_layer2, [-1,28,23,72])
        
        output_shape_d_conv1 = tf.stack([tf.shape(z)[0], 55, 45, 72])
        h_d_conv1 = tf.nn.relu(self.deconv2d(h_d_layer_reshape, self.conv_var_W["conv1"], output_shape_d_conv1, 2)+self.conv_var_b["conv1"])
        output_shape_d_conv2 = tf.stack([tf.shape(z)[0], 55, 45, 48])
        h_d_conv2 = tf.nn.relu(self.deconv2d(h_d_conv1, self.conv_var_W["conv2"], output_shape_d_conv2, 1)+self.conv_var_b["conv2"])

        output_shape_d_conv3 = tf.stack([tf.shape(z)[0], 109, 89, 48])
        h_d_conv3 = tf.nn.relu(self.deconv2d(h_d_conv2, self.conv_var_W["conv3"], output_shape_d_conv3, 2)+self.conv_var_b["conv3"])
        output_shape_d_conv4 = tf.stack([tf.shape(z)[0], 109, 89, 24])
        h_d_conv4 = tf.nn.relu(self.deconv2d(h_d_conv3, self.conv_var_W["conv4"], output_shape_d_conv4, 1)+self.conv_var_b["conv4"])

        output_shape_d_conv5 = tf.stack([tf.shape(z)[0], 218, 178, 24])
        h_d_conv5 = tf.nn.relu(self.deconv2d(h_d_conv4, self.conv_var_W["conv5"], output_shape_d_conv5, 2)+self.conv_var_b["conv5"])
        output_shape_d_conv6 = tf.stack([tf.shape(z)[0], 218, 178, 3])
        h_d_conv6 = tf.nn.relu(self.deconv2d(h_d_conv5, self.conv_var_W["conv6"], output_shape_d_conv6, 1)+self.conv_var_b["conv6"])
        
        return h_d_conv6
    # def getWeightVariables(self):
    #     return [self.conv_var_W[w] for w in self.conv_var_W]
    # def getBiasVariables(self):
    #     return [self.conv_var_b[w] for b in self.conv_var_b]
    # def getLayerVariables(self):
    #     return self.layer1.getVariables()+self.layer2.getVariables()
    # def getEncoderVariables(self):
    #     return self.getWeightVariables()+self.getBiasVariables()+self.getLayerVariables()

# class Discriminator(Encoder, Decoder):
#     def __init__(self):
#     super(Discriminator, self).__init__()

#     def output(self)



class Began(object):
    """docstring for Began"""
    def __init__(self, gamma = gamma, learning_rate = 0.001):
        super(Began, self).__init__()
        self.dataloader = DataLoader()
        self.generator = Decoder(z_dim)
        self.disciminator_encoder = Encoder()
        self.disciminator_decoder = Decoder()
        self.plotter = Plotter()

        self.x_d = tf.placeholder(tf.float32, shape = [None, pic_dim])
        self.x_g = tf.placeholder(tf.float32, shape = [None, z_dim])

        self.k_t = tf.Variable(0.0, tf.float32)

        self.d_real = self.disciminator_decoder.decoderOutput(self.disciminator_encoder.encoderOutput(self.x_d))
        self.g_sample = self.generator.decoderOutput(self.x_g)
        self.d_fake = self.disciminator_decoder.decoderOutput(self.disciminator_encoder.encoderOutput(self.g_sample))

        self.d_loss = tf.reduce_mean(self.loss(self.x_d)-self.k_t*self.loss(self.g_sample))
        self.g_loss = tf.reduce_mean(self.loss(self.g_sample))
        self.M_global = self.loss(self.x_d) + tf.abs(gamma*self.loss(self.x_d) - self.loss(self.g_sample))

        self.d_optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.d_loss, var_list= self.disciminator_encoder.getCoderVariables()+self.disciminator_decoder.getCoderVariables())
        self.g_optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.g_loss, var_list= self.generator.getCoderVariables())

        self.balancer = tf.reduce_mean(gamma*self.loss(self.x_d) - self.loss(self.g_sample))
        self.update_k = self.k_t.assign(tf.clip_by_value(self.k_t + learning_rate*self.balancer, 0, 1))

        self.real_batch = self.dataloader.generateBatch(20)
        self.z_sample = self.sample_Z(20, z_dim)

    def printParameters(self):
        print "batch_size: ", batch_size, ", z_dim: ", z_dim, ", hidden_dim: ", hidden_dim, ", gamma: ",gamma


    def sample_Z(self, m, n):
        return np.random.uniform(-1., 1., size=[m, n])

    def loss(self, x):
        return tf.reduce_mean(tf.abs(tf.reshape(x, [-1, 116412]) - tf.reshape(self.disciminator_decoder.decoderOutput(self.disciminator_encoder.encoderOutput(x)), [-1, 116412])))

    def showStatus(self, sess, batch_x, step):
        d_loss_train, g_loss_train, M_global_train, k_t_step = sess.run([self.d_loss, self.g_loss, self.M_global, self.k_t], feed_dict=
                            {self.x_d: batch_x, self.x_g: self.sample_Z(batch_size, z_dim)})
        print 'step:', step, ' d-loss:', d_loss_train, ' g-loss:', g_loss_train, 'k_t:', k_t_step,"M_global:", M_global_train
    
        Dreal_batch = sess.run(self.d_real, feed_dict = {self.x_d: self.real_batch})
        Dreal_pic = np.array([np.reshape(m,(218,178,3)) for m in Dreal_batch])
        real_pic = np.array([np.reshape(m,(218,178,3)) for m in self.real_batch])

        fake_pic = np.array([np.reshape(m,(218,178,3)) for m in sess.run(self.g_sample, feed_dict = {self.x_g: self.z_sample})])
        Dfake_pic = np.array([np.reshape(m,(218,178,3)) for m in sess.run(self.d_fake, feed_dict = {self.x_g: self.z_sample})])

        self.plotter.plotAndSaveRealPic(step, real_pic, Dreal_pic)
        self.plotter.plotAndSaveFakePic(step, fake_pic, Dfake_pic)

    def train(self, training_steps):
        self.printParameters()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for step in range(training_steps):
            batch_x = self.dataloader.generateBatch(batch_size)
            z_D = self.sample_Z(batch_size, z_dim)
            sess.run([self.d_optimizer, self.g_optimizer], feed_dict={self.x_d: batch_x, self.x_g: z_D})
            z_G = self.sample_Z(batch_size, z_dim)
            sess.run(self.g_optimizer, feed_dict={self.x_g: z_G})
            sess.run(self.update_k, feed_dict={self.x_d: batch_x, self.x_g: z_G})

            if step%500==0 :
                self.showStatus(sess, batch_x, step)




if __name__ == '__main__':
    began = Began()
    began.train(500000)

