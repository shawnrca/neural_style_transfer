
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os, re, gc
from imageio import imread
from skimage import transform


# In[2]:


class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()

        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.avg_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.avg_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.avg_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.avg_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.avg_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')
        
    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
       
        for i, k in enumerate(keys):
            if k.startswith("fc"):
                continue
            else:
                print (i, k, np.shape(weights[k]), end= " ")
                
                sess.run(self.parameters[i].assign(weights[k]))
        print("\n")


class vgg16_nts(vgg16):
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self._sess = sess

        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
    
    def get_content_loss(self, content_act, generated_act):
        return(0.5*tf.reduce_sum(tf.square(tf.subtract(content_act,generated_act))))
    
    def get_gram(self, content_act):
        assert(len(content_act.shape) == 4)
        assert(content_act.shape[0]==1)
        assert(content_act.shape[1]==content_act.shape[2])
        content_act = tf.squeeze(content_act, axis=0)
        gram = tf.reshape(content_act, (content_act.shape[0]*content_act.shape[1], content_act.shape[2]))
        return(tf.matmul(tf.transpose(gram), gram))
    
    
    def get_style_El(self, content_gram, generated_gram, no_of_filters, filter_size):
        El = tf.reduce_sum(tf.square(tf.subtract(generated_gram, content_gram)))/(4*(no_of_filters**2)*(filter_size**2))
        return(El)
    
    def get_total_style_loss(self, layers_Els_pars):
        total_style_loss = 0 
        for content_gram, generated_gram, no_of_filters, filter_size, w in layers_Els_pars:
            total_style_loss += w * self.get_style_El(content_gram, generated_gram, no_of_filters, filter_size)
        return (total_style_loss)
    
def run_tests():
    def print_suc_msg(s):
        print("{} Passed".format(s))
    
        
    tf.reset_default_graph()
    with tf.Session() as sess:
        imgs = tf.placeholder(tf.float32,[1, 224, 224, 3])
        vgg = vgg16_nts(imgs, "vgg16_weights_scaled.npz", sess)

        #################
        #Testing content#
        #################

        np.random.seed(0)
        content_act = np.random.uniform(size=(10, 10)).astype(np.float32) - 0.5
        np.random.seed(1)
        generated_act = np.random.uniform(size=(10, 10)).astype(np.float32) - 0.5
        assert(np.sum((content_act - generated_act)**2)/2 == 
               sess.run(vgg.get_content_loss(tf.constant(content_act), tf.constant(generated_act))))

        print_suc_msg("Testing Content")


        ##############
        #Testing gram#
        ##############

        np.random.seed(0)
        content_act = np.random.uniform(size=(1, 5, 5, 10)).astype(np.float32) - 0.5

        g_content = sess.run(vgg.get_gram(tf.constant(content_act)))
        assert(np.abs(g_content[2,8] - np.sum(np.multiply(content_act[0,:,:,2].ravel(),content_act[0,:,:,8].ravel())))<1e-4)
        assert(np.abs(g_content[4,5] - np.sum(np.multiply(content_act[0,:,:,4].ravel(),content_act[0,:,:,5].ravel())))<1e-4)
        print_suc_msg("Testing Gram")

        ###############
        #Testing Style#
        ###############

        np.random.seed(0)
        content_act = np.random.uniform(size=(1, 5, 5, 10)).astype(np.float32) - 0.5
        np.random.seed(1)
        generated_act = np.random.uniform(size=(1, 5, 5, 10)).astype(np.float32) - 0.5

        el_style = sess.run(vgg.get_style_El(content_gram=vgg.get_gram(tf.constant(content_act)), 
                                    generated_gram=vgg.get_gram(tf.constant(generated_act)),
                                    no_of_filters=content_act.shape[3], 
                                    filter_size=content_act.shape[2]*content_act.shape[1]))

        assert(el_style - np.sum(np.square(sess.run(vgg.get_gram(tf.constant(content_act))) - 
                                           sess.run(vgg.get_gram(tf.constant(generated_act)))))/(4*(25**2)*100)<1e-4)

        np.random.seed(0)
        content_act_l1 = np.random.uniform(size=(1, 5, 5, 10)).astype(np.float32) - 0.5
        np.random.seed(1)
        generated_act_l1 = np.random.uniform(size=(1, 5, 5, 10)).astype(np.float32) - 0.5
        np.random.seed(2)
        content_act_l2 = np.random.uniform(size=(1, 3, 3, 20)).astype(np.float32) - 0.5
        np.random.seed(3)
        generated_act_l2 = np.random.uniform(size=(1, 3, 3, 20)).astype(np.float32) - 0.5

        el_pars = [
            (vgg.get_gram(tf.constant(content_act_l1)), vgg.get_gram(tf.constant(generated_act_l1)), 10, 25, 0.5),
            (vgg.get_gram(tf.constant(content_act_l2)), vgg.get_gram(tf.constant(generated_act_l2)), 20, 9, 0.5)
        ]
        assert(sess.run(
            vgg.get_style_El(vgg.get_gram(tf.constant(content_act_l1)), vgg.get_gram(tf.constant(generated_act_l1)), 10, 25)*0.5 +
            vgg.get_style_El(vgg.get_gram(tf.constant(content_act_l2)), vgg.get_gram(tf.constant(generated_act_l2)), 20, 9)*0.5)\
            == sess.run(vgg.get_total_style_loss(el_pars)))
        print_suc_msg("Testing Style")
    tf.reset_default_graph()

    

