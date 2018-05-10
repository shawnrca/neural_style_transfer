import os, gc
import tensorflow as tf
import numpy as np
from imageio import imread
from skimage import transform
class Vgg19:
    def __init__(self, vgg19_npy_path=None, image_size=224):
        self._image_size = image_size
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        self.VGG_MEAN = [103.939, 116.779, 123.68]
        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [self._image_size, self._image_size, 1]
        assert green.get_shape().as_list()[1:] == [self._image_size, self._image_size, 1]
        assert blue.get_shape().as_list()[1:] == [self._image_size, self._image_size, 1]
        bgr = tf.concat(axis=3, values=[
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [self._image_size, self._image_size, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.avg_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.avg_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.avg_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.avg_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.avg_pool(self.conv5_4, 'pool5')

        self.data_dict = None
        print("build model finished")

    def avg_pool(self, bottom, name):
        return (tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name))

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias, name=name)
            return (relu)

   

    def get_conv_filter(self, name):
        return (tf.constant(self.data_dict[name][0], name="filter"))

    def get_bias(self, name):
        return (tf.constant(self.data_dict[name][1], name="biases"))

#Neural Style transfer 
    
class Vgg19_nts(Vgg19):
    def __init__(self, content_image_path, style_img_path, image_size=224, 
                 ratio_content_to_style=1e-3, vgg19_npy_path="vgg19.npy", 
                 content_acts=None, style_acts=None):
        tf.reset_default_graph()
        self.style_acts = style_acts
             
        if self.style_acts is None:
            self.style_acts = [{"layer_name":"conv1_1", "style_act":0, "w":1e3/64**2},
                              {"layer_name":"conv2_1", "style_act":0, "w":1e3/128**2},
                              {"layer_name":"conv3_1", "style_act":0, "w":1e3/256**2},
                              {"layer_name":"conv4_1", "style_act":0, "w":1e3/512**2},
                              {"layer_name":"conv5_1", "style_act":0, "w":1e3/512**2}]

        
        self.content_acts = content_acts
        
        if self.content_acts is None:
            self.content_acts = [{"layer_name":"conv4_2", "content_act":0, "w":1}]
       
        self.vgg_19_npy_path = vgg19_npy_path
        
        self._image_size = image_size
        self.content_img = imread(content_image_path)
        self.content_img = transform.resize(self.content_img, [image_size, image_size])
        self.content_img = np.expand_dims(self.content_img, axis=0)
        
        self.style_img = imread(style_img_path)
        self.style_img = transform.resize(self.style_img, [image_size, image_size])
        self.style_img = np.expand_dims(self.style_img, axis=0)
        self._ratio_content_to_style = ratio_content_to_style
        
        self.set_content_and_style_activations()
          
    def get_content_loss(self, content, generated_act):
        return(0.5*tf.reduce_sum(tf.square(tf.subtract(content,generated_act))))
    
    def get_gram(self, content):
        content = tf.squeeze(content, axis=0)
        gram = tf.reshape(content, (content.shape[0].value*content.shape[1].value, content.shape[2].value))
        return(tf.matmul(tf.transpose(gram), gram))
    
    def get_style_El(self, content_gram, generated_gram, no_of_filters, filter_size):
        El = tf.reduce_sum(tf.square(tf.subtract(generated_gram, content_gram)))/(4*(no_of_filters**2)*(filter_size**2))
        return(El)
 
    def calculate_total_loss(self):
        loss_content = 0
        loss_style = 0
        for it in self.content_acts:
            generated_act = self.graph.get_tensor_by_name("{0}/{0}:0".format(it["layer_name"]))
            content_act = tf.constant(it["content_act"])
            loss_content += it["w"] * self.get_content_loss(content_act, generated_act)
        
        for it in self.style_acts:
            generated_act = self.graph.get_tensor_by_name("{0}/{0}:0".format(it["layer_name"]))
            content_act = tf.constant(it["style_act"])
            
            generated_gram = self.get_gram(generated_act)
            content_gram = self.get_gram(content_act)
           
            no_of_filters = content_act.shape[3].value
            filter_size = content_act.shape[1].value*content_act.shape[2].value

            loss_style += it["w"] * self.get_style_El(content_gram, generated_gram, no_of_filters, filter_size)
            
        return(self._ratio_content_to_style*loss_content + loss_style)
       
    def build_optmizer(self, learning_rate):
        loss = self.calculate_total_loss()
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return([optimizer, optimizer.minimize(loss), loss])
    
    def set_content_and_style_activations(self):
   
        tf.reset_default_graph()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            with tf.device("/cpu:0"):
                imgs = tf.placeholder(shape=[1, self._image_size, self._image_size, 3], dtype=tf.float32)
                vgg19_ref = Vgg19(self.vgg_19_npy_path, image_size=self._image_size)
                vgg19_ref.build(imgs)
                self.graph = sess.graph

                for i, ls in enumerate(self.style_acts):
                    self.style_acts[i]["style_act"] = sess.run("{0}/{0}:0".format(ls["layer_name"]), 
                                                          feed_dict={imgs:self.style_img})
    

                for i, lc in enumerate(self.content_acts):
                    self.content_acts[i]["content_act"] = sess.run("{0}/{0}:0".format(lc["layer_name"]), 
                                                              feed_dict={imgs:self.content_img})
        del vgg19_ref
        gc.collect()
      
      
    
    def paint(self, learning_rate=0.005, no_of_epochs = 100, sample_rate=10, stack_len=10):
        tf.reset_default_graph()
        images = list()

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
       
            with tf.device("/cpu:0"):
                imgs = tf.get_variable(shape=[1, self._image_size, self._image_size, 3], initializer=tf.random_uniform_initializer(0, 1), 
                                   name="painted", dtype=tf.float32) 
           
                Vgg19.__init__(self, vgg19_npy_path=self.vgg_19_npy_path,image_size=self._image_size)
                Vgg19.build(self, imgs)
                
                self.graph = sess.graph
                
                sess.run(tf.variables_initializer([imgs]))
 
                adam_op, op_min, loss = self.build_optmizer(learning_rate=learning_rate)
                adam_vars = [v for v in self.graph.get_collection("variables") 
                             if v.name in ['beta1_power:0', 'beta2_power:0', 'painted/Adam:0', 'painted/Adam_1:0']]
                sess.run(tf.variables_initializer(adam_vars))
        
                for epoch in range(no_of_epochs):
                    _, ls = sess.run([op_min, loss])
                    if epoch%sample_rate==0:
                        images.append(np.clip(sess.run(imgs), 0, 1))
                        if len(images)>stack_len:
                            images = images[-stack_len::]
                    print("\r loss:{:<8}  epoch:{} of {}".format(ls, epoch, no_of_epochs), end=" ")   
        return(images)
