# Importing packages
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import scipy
from scipy.io import loadmat
import re
import string
import random
import time
import argparse
import threading
import scipy.ndimage as ndi
from skimage import transform
from skimage import exposure
import skimage

# Generator
class Generator:
    def __init__(self, input_z, input_rnn, is_training, reuse):
        self.input_z = input_z
        self.input_rnn = input_rnn
        self.is_training = is_training
        self.reuse = reuse
        self.t_dim = 128
        self.gf_dim = 128
        self.image_size = 64
        self.c_dim = 3
        self._build_model()

    def _build_model(self):
        s = self.image_size
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        gf_dim = self.gf_dim
        t_dim = self.t_dim
        c_dim = self.c_dim

        with tf.variable_scope("generator", reuse=self.reuse):
            net_txt = fc(inputs=self.input_rnn, num_out=t_dim, activation_fn=tf.nn.leaky_relu, name='rnn_fc')
            net_in = concat([self.input_z, net_txt], axis=1, name='concat_z_txt')

            net_h0 = fc(inputs=net_in, num_out=gf_dim*8*s16*s16, name='g_h0/fc', biased=False)
            net_h0 = batch_normalization(net_h0, activation_fn=None, is_training=self.is_training, name='g_h0/batch_norm')
            net_h0 = reshape(net_h0, [-1, s16, s16, gf_dim*8], name='g_h0/reshape')
            
            net = Conv2d(net_h0, 1, 1, gf_dim*2, 1, 1, name='g_h1_res/conv2d')
            net = batch_normalization(net, activation_fn=tf.nn.relu, is_training=self.is_training, name='g_h1_res/batch_norm')
            net = Conv2d(net, 3, 3, gf_dim*2, 1, 1, name='g_h1_res/conv2d2', padding='SAME')
            net = batch_normalization(net, activation_fn=tf.nn.relu, is_training=self.is_training, name='g_h1_res/batch_norm2')
            net = Conv2d(net, 3, 3, gf_dim*8, 1, 1, name='g_h1_res/conv2d3', padding='SAME')
            net = batch_normalization(net, activation_fn=None, is_training=self.is_training, name='g_h1_res/batch_norm3')

            net_h1 = add([net_h0, net], name='g_h1_res/add')
            net_h1_output = tf.nn.relu(net_h1)
            
            net_h2 = UpSample(net_h1_output, size=[s8, s8], method=1, align_corners=False, name='g_h2/upsample2d')
            net_h2 = Conv2d(net_h2, 3, 3, gf_dim*4, 1, 1, name='g_h2/conv2d', padding='SAME')
            net_h2 = batch_normalization(net_h2, activation_fn=None, is_training=self.is_training, name='g_h2/batch_norm')

            net = Conv2d(net_h2, 1, 1, gf_dim, 1, 1, name='g_h3_res/conv2d')
            net = batch_normalization(net, activation_fn=tf.nn.relu, is_training=self.is_training, name='g_h3_res/batch_norm')
            net = Conv2d(net, 3, 3, gf_dim, 1, 1, name='g_h3_res/conv2d2', padding='SAME')
            net = batch_normalization(net, activation_fn=tf.nn.relu, is_training=self.is_training, name='g_h3_res/batch_norm2')
            net = Conv2d(net, 3, 3, gf_dim*4, 1, 1, name='g_h3_res/conv2d3', padding='SAME')
            net = batch_normalization(net, activation_fn=None, is_training=self.is_training, name='g_h3_res/batch_norm3')
            
            net_h3 = add([net_h2, net], name='g_h3/add')
            net_h3_outputs = tf.nn.relu(net_h3)

            net_h4 = UpSample(net_h3_outputs, size=[s4, s4], method=1, align_corners=False, name='g_h4/upsample2d')
            net_h4 = Conv2d(net_h4, 3, 3, gf_dim*2, 1, 1, name='g_h4/conv2d', padding='SAME')
            net_h4 = batch_normalization(net_h4, activation_fn=tf.nn.relu, is_training=self.is_training, name='g_h4/batch_norm')

            net_h5 = UpSample(net_h4, size=[s2, s2], method=1, align_corners=False, name='g_h5/upsample2d')
            net_h5 = Conv2d(net_h5, 3, 3, gf_dim, 1, 1, name='g_h5/conv2d', padding='SAME')
            net_h5 = batch_normalization(net_h5, activation_fn=tf.nn.relu, is_training=self.is_training, name='g_h5/batch_norm')

            net_ho = UpSample(net_h5, size=[s, s], method=1, align_corners=False, name='g_ho/upsample2d')
            net_ho = Conv2d(net_ho, 3, 3, c_dim, 1, 1, name='g_ho/conv2d', padding='SAME', biased=True) ## biased = True

            self.outputs = tf.nn.tanh(net_ho)
            self.logits = net_ho

# Discriminator
class Discriminator:
    def __init__(self, input_image, input_rnn, is_training, reuse):
        self.input_image = input_image
        self.input_rnn = input_rnn
        self.is_training = is_training
        self.reuse = reuse
        self.df_dim = 64
        self.t_dim = 128
        self.image_size = 64
        self._build_model()

    def _build_model(self):
        s = self.image_size
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        df_dim = self.df_dim
        t_dim = self.t_dim

        with tf.variable_scope("discriminator", reuse=self.reuse):
            net_h0 = Conv2d(self.input_image, 4, 4, df_dim, 2, 2, name='d_h0/conv2d', activation_fn=tf.nn.leaky_relu, padding='SAME', biased=True)

            net_h1 = Conv2d(net_h0, 4, 4, df_dim*2, 2, 2, name='d_h1/conv2d', padding='SAME')
            net_h1 = batch_normalization(net_h1, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='d_h1/batchnorm')

            net_h2 = Conv2d(net_h1, 4, 4, df_dim*4, 2, 2, name='d_h2/conv2d', padding='SAME')
            net_h2 = batch_normalization(net_h2, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='d_h2/batchnorm')

            net_h3 = Conv2d(net_h2, 4, 4, df_dim*8, 2, 2, name='d_h3/conv2d', padding='SAME')
            net_h3 = batch_normalization(net_h3, activation_fn=None, is_training=self.is_training, name='d_h3/batchnorm')

            net = Conv2d(net_h3, 1, 1, df_dim*2, 1, 1, name='d_h4_res/conv2d')
            net = batch_normalization(net, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='d_h4_res/batchnorm')
            net = Conv2d(net, 3, 3, df_dim*2, 1, 1, name='d_h4_res/conv2d2', padding='SAME')
            net = batch_normalization(net, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='d_h4_res/batchnorm2')
            net = Conv2d(net, 3, 3, df_dim*8, 1, 1, name='d_h4_res/conv2d3', padding='SAME')
            net = batch_normalization(net, activation_fn=None, is_training=self.is_training, name='d_h4_res/batchnorm3')

            net_h4 = add([net_h3, net], name='d_h4/add')
            net_h4_outputs = tf.nn.leaky_relu(net_h4)

            net_txt = fc(self.input_rnn, num_out=t_dim, activation_fn=tf.nn.leaky_relu, name='d_reduce_txt/dense')
            net_txt = tf.expand_dims(net_txt, axis=1, name='d_txt/expanddim1')
            net_txt = tf.expand_dims(net_txt, axis=1, name='d_txt/expanddim2')
            net_txt = tf.tile(net_txt, [1, 4, 4, 1], name='d_txt/tile')
            
            net_h4_concat = concat([net_h4_outputs, net_txt], axis=3, name='d_h3_concat')

            net_h4 = Conv2d(net_h4_concat, 1, 1, df_dim*8, 1, 1, name='d_h3/conv2d_2')
            net_h4 = batch_normalization(net_h4, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='d_h3/batch_norm_2')

            net_ho = Conv2d(net_h4, s16, s16, 1, s16, s16, name='d_ho/conv2d', biased=True) # biased = True

            self.outputs = tf.nn.sigmoid(net_ho)
            self.logits = net_ho

# RNN ENCODER
class rnn_encoder:
    def __init__(self, input_seqs, is_training, reuse):
        self.input_seqs = input_seqs
        self.is_training = is_training
        self.reuse = reuse
        self.t_dim = 128  
        self.rnn_hidden_size = 128
        self.vocab_size = 8000
        self.word_embedding_size = 256
        self.keep_prob = 1.0
        self.batch_size = 64
        self._build_model()

    def _build_model(self):
        w_init = tf.random_normal_initializer(stddev=0.02)
        GRUCell = tf.nn.rnn_cell.GRUCell

        with tf.variable_scope("rnnftxt", reuse=self.reuse):
            word_embed_matrix = tf.get_variable('rnn/wordembed', 
                shape=(self.vocab_size, self.word_embedding_size),
                initializer=tf.random_normal_initializer(stddev=0.02),
                dtype=tf.float32)
            embedded_word_ids = tf.nn.embedding_lookup(word_embed_matrix, self.input_seqs)

            # RNN encoder
           GRUCell = tf.nn.rnn_cell.GRUCell(self.t_dim, reuse=self.reuse)
            initial_state = GRUCell.zero_state(self.batch_size, dtype=tf.float32)            
            rnn_net = tf.nn.dynamic_rnn(cell= GRUCell,
                                    inputs=embedded_word_ids,
                                    initial_state=initial_state,
                                    dtype=np.float32,
                                    time_major=False,
                                    scope='rnn/dynamic')

            self.rnn_net = rnn_net
            self.outputs = rnn_net[0][:, -1, :]

# CNN ENCODER
class cnn_encoder:
    def __init__(self, inputs, is_training=True, reuse=False):
        self.inputs = inputs
        self.is_training = is_training
        self.reuse = reuse
        self.df_dim = 64
        self.t_dim = 128
        self._build_model()

    def _build_model(self):
        df_dim = self.df_dim

        with tf.variable_scope('cnnftxt', reuse=self.reuse):
            net_h0 = Conv2d(self.inputs, 4, 4, df_dim, 2, 2, name='cnnf/h0/conv2d', activation_fn=tf.nn.leaky_relu, padding='SAME', biased=True)
            net_h1 = Conv2d(net_h0, 4, 4, df_dim*2, 2, 2, name='cnnf/h1/conv2d', padding='SAME')
            net_h1 = batch_normalization(net_h1, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='cnnf/h1/batch_norm')

            net_h2 = Conv2d(net_h1, 4, 4, df_dim*4, 2, 2, name='cnnf/h2/conv2d', padding='SAME')
            net_h2 = batch_normalization(net_h2, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='cnnf/h2/batch_norm')

            net_h3 = Conv2d(net_h2, 4, 4, df_dim*8, 2, 2, name='cnnf/h3/conv2d', padding='SAME')
            net_h3 = batch_normalization(net_h3, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='cnnf/h3/batch_norm')

            net_h4 = flatten(net_h3, name='cnnf/h4/flatten')
            net_h4 = fc(net_h4, num_out=self.t_dim, name='cnnf/h4/embed', biased=False)
        
        self.outputs = net_h4

# Importing Dataset
vocab = np.load(r'C:\Users\riagu\Documents\text-to-image dictionary\vocab.npy')
print('there are {} vocabularies in total'.format(len(vocab)))
 
word2Id_dict = dict(np.load(r"C:\Users\riagu\Documents\text-to-image dictionary\word2Id.npy"))
 
id2word_dict = dict(np.load(r"C:\Users\riagu\Documents\text-to-image dictionary\id2Word.npy"))
train_images = np.load(r"C:\Users\riagu\Documents\text-to-image\train_images.npy", encoding='latin1')
train_captions = np.load(r"C:\Users\riagu\Documents\text-to-image\train_captions.npy", encoding='latin1')
 

# Data Preprocessing 
# Helper Functions used:
def fc(inputs, num_out, name, activation_fn=None, biased=True):
    w_init = tf.random_normal_initializer(stddev=0.02)
    return tf.layers.dense(inputs=inputs, units=num_out, activation=activation_fn, kernel_initializer=w_init, use_bias=biased, name=name)


def concat(inputs, axis, name):
    return tf.concat(values=inputs, axis=axis, name=name)

def batch_normalization(inputs, is_training, name, activation_fn=None):
    output = tf.layers.batch_normalization(
                    inputs,
                    momentum=0.95,
                    epsilon=1e-5,
                    training=is_training,
                    name=name
                )

    if activation_fn is not None:
        output = activation_fn(output)

    return output

def reshape(inputs, shape, name):
    return tf.reshape(inputs, shape, name)

def Conv2d(input, k_h, k_w, c_o, s_h, s_w, name, activation_fn=None, padding='VALID', biased=False):
    c_i = input.get_shape()[-1]
    w_init = tf.random_normal_initializer(stddev=0.02)

    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable(name='weights', shape=[k_h, k_w, c_i, c_o], initializer=w_init)
        output = convolve(input, kernel)

        if biased:
            biases = tf.get_variable(name='biases', shape=[c_o])
            output = tf.nn.bias_add(output, biases)
        if activation_fn is not None:
            output = activation_fn(output, name=scope.name)

        return output

def add(inputs, name):
    return tf.add_n(inputs, name=name)

def UpSample(inputs, size, method, align_corners, name):
    return tf.image.resize_images(inputs, size, method, align_corners)

def flatten(input, name):
    input_shape = input.get_shape()
    dim = 1
    for d in input_shape[1:].as_list():
        dim *= d
        input = tf.reshape(input, [-1, dim])
    
    return input

def IdList2sent(caption):
    sentence = []
    for ID in caption:
        if ID != word2Id_dict['<PAD>']:
            sentence.append(id2word_dict[ID])

    return sentence
def sent2IdList(line, MAX_SEQ_LENGTH=20):
    MAX_SEQ_LIMIT = MAX_SEQ_LENGTH
    padding = 0
    prep_line = re.sub('[%s]' % re.escape(string.punctuation), ' ', line.rstrip())
    prep_line = prep_line.replace('-', ' ')
    prep_line = prep_line.replace('-', ' ')
    prep_line = prep_line.replace('  ', ' ')
    prep_line = prep_line.replace('.', '')
    tokens = prep_line.split(' ')
    tokens = [
        tokens[i] for i in range(len(tokens))
        if tokens[i] != ' ' and tokens[i] != ''
    ]
    l = len(tokens)
    padding = MAX_SEQ_LIMIT - l
    for i in range(padding):
        tokens.append('<PAD>')
    
    line = [
        word2Id_dict[tokens[k]]
        if tokens[k] in word2Id_dict else word2Id_dict['<RARE>']
        for k in range(len(tokens))
    ]

    return line
def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')
def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))
def apply_transform(x, transform_matrix, channel_index=2, fill_mode='nearest', cval=0., order=1):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=order, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def get_random_int(min=0, max=10, number=5):
    """Return a list of random integer by the given range and quantity.
    Examples
    ---------
    >>> r = get_random_int(min=0, max=10, number=5)
    ... [10, 2, 3, 3, 7]
    """
    return [random.randint(min,max) for p in range(0,number)]

def threading_data(data=None, fn=None, **kwargs):
    def apply_fn(results, i, data, kwargs):
        results[i] = fn(data, **kwargs)

    ## start multi-threaded reading.
    results = [None] * len(data) ## preallocate result list
    threads = []
    for i in range(len(data)):
        t = threading.Thread(
                        name='threading_and_return',
                        target=apply_fn,
                        args=(results, i, data[i], kwargs)
                        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return np.asarray(results)

def flip_axis(x, axis, is_random=False):
    if is_random:
        factor = np.random.uniform(-1, 1)
        if factor > 0:
            x = np.asarray(x).swapaxes(axis, 0)
            x = x[::-1, ...]
            x = x.swapaxes(0, axis)
            return x
        else:
            return x
    else:
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x
    
def imresize(x, size=[100, 100], interp='bilinear', mode=None):
    if x.shape[-1] == 1:
        # greyscale
        x = scipy.misc.imresize(x[:,:,0], size, interp=interp, mode=mode)
        return x[:, :, np.newaxis]
    elif x.shape[-1] == 3:
        # rgb, bgr ..
        return scipy.misc.imresize(x, size, interp=interp, mode=mode)
    else:
        raise Exception("Unsupported channel %d" % x.shape[-1])

def rotation(x, rg=20, is_random=False, row_index=0, col_index=1, channel_index=2,
                    fill_mode='nearest', cval=0.):
    if is_random:
        theta = np.pi / 180 * np.random.uniform(-rg, rg)
    else:
        theta = np.pi /180 * rg
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x

def crop(x, wrg, hrg, is_random=False, row_index=0, col_index=1, channel_index=2):
    h, w = x.shape[row_index], x.shape[col_index]
    assert (h > hrg) and (w > wrg), "The size of cropping should smaller than the original image"
    if is_random:
        h_offset = int(np.random.uniform(0, h-hrg) -1)
        w_offset = int(np.random.uniform(0, w-wrg) -1)
        return x[h_offset: hrg+h_offset ,w_offset: wrg+w_offset]
    else:   # central crop
        h_offset = int(np.floor((h - hrg)/2.))
        w_offset = int(np.floor((w - wrg)/2.))
        h_end = h_offset + hrg
        w_end = w_offset + wrg
        return x[h_offset: h_end, w_offset: w_end]        
        

def prepro_img(x, mode=None):
    # rescale [0, 255] --> (-1, 1), random flip, crop, rotate

    if mode=='train':
        x = flip_axis(x, axis=1, is_random=True)
        x = rotation(x, rg=16, is_random=True, fill_mode='nearest')
        x = imresize(x, size=[64+15, 64+15], interp='bilinear', mode=None)
        x = crop(x, wrg=64, hrg=64, is_random=True)
        x = x / (255. / 2.)
        x = x - 1.
        # x = x * 0.9999

    return x
## Save images
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def cosine_similarity(v1, v2):
    cost = tf.reduce_sum(tf.multiply(v1, v2), 1) / (tf.sqrt(tf.reduce_sum(tf.multiply(v1, v1), 1)) * tf.sqrt(tf.reduce_sum(tf.multiply(v2, v2), 1)))
    return cost

# Data Preprocessing
print('----example of captions[0]--------')
for caption in train_captions[0]:
    print(IdList2sent(caption))
  

captions_list = []
for captions in train_captions:
    assert len(captions) >= 5
    captions_list.append(captions[:5])
train_captions = np.concatenate(captions_list, axis=0)
n_captions_train = len(train_captions)
n_captions_per_image = 5
n_images_train = len(train_images)
print('Total captions: ', n_captions_train)
print('----example of captions[0] (modified)--------')
for caption in train_captions[:5]:
    print(IdList2sent(caption))
 

# Setting parameter for Training Data

lr = 0.0002
lr_decay = 0.5      
decay_every = 100  
beta1 = 0.5
z_dim = 512         # Noise dimension
image_size = 64     # 64 x 64
c_dim = 3           # for rgb
batch_size = 64
ni = int(np.ceil(np.sqrt(batch_size)))

# Sample Sentences for Testing 
### Testing setting
sample_size = batch_size
sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
sample_sentence = ["the flower shown has yellow anther red pistil and bright red petals."] * int(sample_size/ni) + \
                  ["this flower has petals that are yellow, white and purple and has dark lines"] * int(sample_size/ni) + \
                  ["the petals on this flower are white with a yellow center"] * int(sample_size/ni) + \
                  ["this flower has a lot of small round pink petals."] * int(sample_size/ni) + \
                  ["this flower is orange in color, and has petals that are ruffled and rounded."] * int(sample_size/ni) + \
                  ["the flower has yellow petals and the center of it is brown."] * int(sample_size/ni) + \
                  ["this flower has petals that are blue and white."] * int(sample_size/ni) +\
                  ["these white flowers have petals that start off white in color and end in a white towards the tips."] * int(sample_size/ni)
for i, sent in enumerate(sample_sentence):
    sample_sentence[i] = sent2IdList(sent)

# Training
t_real_image = tf.placeholder('float32',[batch_size,image_size,image_size,3],name='real_image')
t_wrong_image = tf.placeholder('float32',[batch_size,image_size,image_size,3],name='wrong_image')
t_real_caption = tf.placeholder(dtype= tf.int64 ,shape=[batch_size,None],name='real_caption_input')
t_wrong_caption = tf.placeholder(dtype=tf.int64,shape=[batch_size,None],name='wrong_caption_input')
t_z = tf.placeholder(tf.float32,[batch_size , z_dim],name='z_noise')
Training Phase - CNN - RNN mapping
net_cnn = cnn_encoder(t_real_image,is_training=True, reuse=False)
x = net_cnn.outputs
net_rnn = rnn_encoder(t_real_caption,is_training=True, reuse=False)
v= net_rnn.outputs
x_wrong =cnn_encoder(t_wrong_image,is_training=True, reuse=True).outputs
v_wrong = rnn_encoder(t_wrong_caption,is_training=True, reuse=True).outputs
alpha =0.2
rnn_loss = tf.reduce_mean(tf.maximum(0.,alpha - cosine_similarity(x,v) + cosine_similarity(x,v_wrong)))+ tf.reduce_mean(tf.maximum(0.,alpha - cosine_similarity(x,v) + cosine_similarity(x_wrong,v)))

# Training Phase – GAN
net_rnn = rnn_encoder(t_real_caption,is_training=False, reuse=True)
net_fake_image = Generator(t_z,net_rnn.outputs,is_training=True, reuse=False)

net_disc_fake = Discriminator(net_fake_image.outputs,net_rnn.outputs,is_training=True, reuse=False)
disc_fake_logits = net_disc_fake.logits

net_disc_real = Discriminator(t_real_image ,net_rnn.outputs,is_training=True, reuse=True)
disc_real_logits = net_disc_real.logits

net_disc_mismatch = Discriminator(t_real_image ,rnn_encoder(t_wrong_caption, is_training=False, reuse=True).outputs,
                                is_training=True, reuse=True)
disc_mismatch_logits = net_disc_mismatch.logits
d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_logits,     labels=tf.ones_like(disc_real_logits),      name='d1'))
d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_mismatch_logits, labels=tf.zeros_like(disc_mismatch_logits), name='d2'))
d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits,     labels=tf.zeros_like(disc_fake_logits),     name='d3'))
d_loss = d_loss1 + (d_loss2 + d_loss3) * 0.5

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.ones_like(disc_fake_logits), name='g'))
net_g = Generator(t_z, rnn_encoder(t_real_caption, is_training=False, reuse=True).outputs,
                    is_training=False, reuse=True)

rnn_vars = [var for var in tf.trainable_variables() if 'rnn' in var.name]
g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
d_vars = [var for var in tf.trainable_variables() if 'discrim' in var.name]
cnn_vars = [var for var in tf.trainable_variables() if 'cnn' in var.name]

update_ops_D = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'discrim' in var.name]
update_ops_G = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'generator' in var.name]
update_ops_CNN = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'cnn' in var.name]

print('----------Update_ops_D--------')
for var in update_ops_D:
        print(var.name)
print('----------Update_ops_G--------')
for var in update_ops_G:
        print(var.name)
print('----------Update_ops_CNN--------')
for var in update_ops_CNN:
        print(var.name)
 

 
checkpoint_dir = './checkpoint'
with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)

with tf.control_dependencies(update_ops_D):
        d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

with tf.control_dependencies(update_ops_G):
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)

with tf.control_dependencies(update_ops_CNN):
        grads, _ = tf.clip_by_global_norm(tf.gradients(rnn_loss, rnn_vars + cnn_vars), 10)
        optimizer = tf.train.AdamOptimizer(lr_v, beta1=beta1)
        rnn_optim = optimizer.apply_gradients(zip(grads, rnn_vars + cnn_vars))

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5)

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    loader = tf.train.Saver(var_list=tf.global_variables())
    load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
    load(loader, sess, ckpt.model_checkpoint_path)
else:
    print('no checkpoints find.')
 
# Commencing Training
n_epoch = 600
n_batch_epoch = int(n_images_train / batch_size)
for epoch in range(n_epoch):
    start_time = time.time()

    if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
            
    elif epoch == 0:
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)

    for step in range(n_batch_epoch):
        step_time = time.time()

        ## get matched text & image
        idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
        b_real_caption = train_captions[idexs]
        b_real_images = train_images[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]

        """ check for loading right images
        save_images(b_real_images, [ni, ni], 'train_samples/train_00.png')
        for caption in b_real_caption[:8]:
        print(IdList2sent(caption))
        exit()
        """

        ## get wrong caption & wrong image
        idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
        b_wrong_caption = train_captions[idexs]
        idexs2 = get_random_int(min=0, max=n_images_train-1, number=batch_size)
        b_wrong_images = train_images[idexs2]

        ## get noise
        b_z = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, z_dim)).astype(np.float32)

        b_real_images = threading_data(b_real_images, prepro_img, mode='train')   # [0, 255] --> [-1, 1] + augmentation
        b_wrong_images = threading_data(b_wrong_images, prepro_img, mode='train')

        ## update RNN
        if epoch < 80:
            errRNN, _ = sess.run([rnn_loss, rnn_optim], feed_dict={
                                                t_real_image : b_real_images,
                                                t_wrong_image : b_wrong_images,
                                                t_real_caption : b_real_caption,
                                                t_wrong_caption : b_wrong_caption})
        else:
            errRNN = 0

        ## updates D
        errD, _ = sess.run([d_loss, d_optim], feed_dict={
                            t_real_image : b_real_images,
                            t_wrong_caption : b_wrong_caption,
                            t_real_caption : b_real_caption,
                            t_z : b_z})
        ## updates G
        errG, _ = sess.run([g_loss, g_optim], feed_dict={
                            t_real_caption : b_real_caption,
                            t_z : b_z})

        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.8f, g_loss: %.8f, rnn_loss: %.8f" \
                        % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errD, errG, errRNN))
        
        if (epoch + 1) % 1 == 0:
            print(" ** Epoch %d took %fs" % (epoch, time.time()-start_time))
            img_gen, rnn_out = sess.run([net_g.outputs, net_rnn.outputs], feed_dict={
                                        t_real_caption : sample_sentence,
                                        t_z : sample_seed})

            save_images(img_gen, [ni, ni],r"C:\Users\riagu\Documents\train_samples\train_{:02d}.png".format(epoch))

        if (epoch != 0) and (epoch % 10) == 0:
            save(saver, sess, checkpoint_dir, epoch)
            print("[*] Save checkpoints SUCCESS!")

testData = os.path.join('dataset', 'testData.pkl')

# Testing
    captions = train_captions
    caption = []
    for i in range(len(captions)):
        caption.append(captions[i])
    caption = np.asarray(caption)
    #index = data['ID'].values
    #index = np.asarray(index)

    t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

    net_g = Generator(t_z, rnn_encoder(t_real_caption, is_training=False, reuse=True).outputs,
                    is_training=False, reuse=True)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=tf.global_variables())
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('no checkpoints find.')

    n_caption_test = len(caption)
    n_batch_epoch = int(n_caption_test / batch_size) + 1

    ## repeat
    caption = np.tile(caption, (2, 1))
    #index = np.tile(index, 2)

    #assert index[0] == index[n_caption_test]

    for i in range(n_batch_epoch):
        test_cap = caption[i*batch_size: (i+1)*batch_size]

        z = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, z_dim)).astype(np.float32)
        gen = sess.run(net_g.outputs, feed_dict={t_real_caption: test_cap, t_z: z})
        for j in range(batch_size):
            save_images(np.expand_dims(gen[j], axis=0), [1, 1],r"C:\Users\riagu\Documents\inference\inference_{:04d}.png".format(i*batch_size + j))

