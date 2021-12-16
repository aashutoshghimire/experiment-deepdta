from __future__ import print_function
#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import random as rn

### We modified Pahikkala et al. (2014) source code for cross-val process ###

import os

#below 3 lines helps to align fixed gpu
os.environ['PYTHONHASHSEED'] = '6'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "6";

np.random.seed(1)
rn.seed(1)

# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
import keras
from keras import backend as K
# tf.set_random_seed(0)
tf.random.set_seed(0)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)



from datahelper import *
#import logging
from itertools import product
from arguments import argparser, logging

import keras
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
# from tensorflow.keras.models import Sequential
from keras.layers import BatchNormalization

from keras.layers import Conv2D, GRU
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, merge, Flatten
from keras.models import Model
from keras.utils.vis_utils import plot_model
# from keras.utils import plot_model
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers, layers


import sys, pickle, os
import math, json, time
import decimal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import shuffle
from copy import deepcopy
from sklearn import preprocessing
from emetrics import get_aupr, get_cindex, get_rm2

from keras.layers import Layer
from keras.layers import Conv2D
from keras.layers import Concatenate, concatenate, Reshape



TABSY = "\t"
figdir = "figures/"

def build_combined_onehot(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    XDinput = Input(shape=(FLAGS.max_smi_len, FLAGS.charsmiset_size))
    XTinput = Input(shape=(FLAGS.max_seq_len, FLAGS.charseqset_size))


    encode_smiles= Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles) #pool_size=pool_length[i]


    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)



    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein])
    #encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected 
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    predictions = Dense(1, kernel_initializer='normal')(FC2) 

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']
    

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_combined_onehot.png')

    return interactionModel

# https://github.com/titu1994/keras-attention-augmented-convs/blob/master/attn_augconv.py
# most of the implementation is taken from this repo
# I extended to add 1-D CNN implementation.
# Even though, it's (the extension) not tested properly yet.
# Code is tested.
# Updated to be used in a tf graph

from keras import initializers


def _conv_layer(filters, kernel_size, strides=(1, 1), padding='same', name=None):
    return Conv2D(filters, kernel_size, strides=strides, padding=padding,
                  use_bias=True, kernel_initializer='he_normal', name=name)


def _conv_layer1d(ip, t_n, f_n, filters, kernel_size, strides=1, padding='same', name=None):
    

    conv1 = Conv1D(filters, kernel_size, strides=strides, padding=padding,
                  use_bias=True, kernel_initializer='he_normal', name=name)(ip)
    
    reshape = Reshape((t_n, 1, filters))(conv1)
    
    return reshape



def _conv_layer1r(ip, t_n, f_n, filters, kernel_size, strides=1, padding='same', name=None):
    
    reshape1 = Reshape((t_n, f_n))(ip)
    

    conv1 = Conv1D(filters, kernel_size, strides=strides, padding=padding,
                  use_bias=True, kernel_initializer='he_normal', name=name)(reshape1)
    
    reshape2 = Reshape((t_n, 1, filters))(conv1)
    
    return reshape2


def _normalize_depth_vars(depth_k, depth_v, filters):
    """
    Accepts depth_k and depth_v as either floats or integers
    and normalizes them to integers.
    Args:
        depth_k: float or int.
        depth_v: float or int.
        filters: number of output filters.
    Returns:
        depth_k, depth_v as integers.
    """

    if type(depth_k) == float:
        depth_k = int(filters * depth_k)
    else:
        depth_k = int(depth_k)

    if type(depth_v) == float:
        depth_v = int(filters * depth_v)
    else:
        depth_v = int(depth_v)

    return depth_k, depth_v

## ######################## ##
#
#  AttentionAugmentation2D Class
#
## ######################## ## 

class AttentionAugmentation2D(Layer):

    def __init__(self, depth_k, depth_v, num_heads, relative=True, **kwargs):
        """
        Applies attention augmentation on a convolutional layer
        output.
        Args:
            depth_k: float or int. Number of filters for k.
            Computes the number of filters for `v`.
            If passed as float, computed as `filters * depth_k`.
        depth_v: float or int. Number of filters for v.
            Computes the number of filters for `k`.
            If passed as float, computed as `filters * depth_v`.
        num_heads: int. Number of attention heads.
            Must be set such that `depth_k // num_heads` is > 0.
        relative: bool, whether to use relative encodings.
        Raises:
            ValueError: if depth_v or depth_k is not divisible by
                num_heads.
        Returns:
            Output tensor of shape
            -   [Batch, Height, Width, Depth_V] if
                channels_last data format.
            -   [Batch, Depth_V, Height, Width] if
                channels_first data format.
        """
        super(AttentionAugmentation2D, self).__init__(**kwargs)

        if depth_k % num_heads != 0:
            raise ValueError('`depth_k` (%d) is not divisible by `num_heads` (%d)' % (
                depth_k, num_heads))

        if depth_v % num_heads != 0:
            raise ValueError('`depth_v` (%d) is not divisible by `num_heads` (%d)' % (
                depth_v, num_heads))

        if depth_k // num_heads < 1.:
            raise ValueError('depth_k / num_heads cannot be less than 1 ! '
                             'Given depth_k = %d, num_heads = %d' % (
                             depth_k, num_heads))

        if depth_v // num_heads < 1.:
            raise ValueError('depth_v / num_heads cannot be less than 1 ! '
                             'Given depth_v = %d, num_heads = %d' % (
                                 depth_v, num_heads))

        self.depth_k = depth_k
        self.depth_v = depth_v
        self.num_heads = num_heads
        self.relative = relative

        self.axis = 1 if K.image_data_format() == 'channels_first' else -1

    def build(self, input_shape):
        self._shape = input_shape

        # normalize the format of depth_v and depth_k
        self.depth_k, self.depth_v = _normalize_depth_vars(self.depth_k, self.depth_v,
                                                           input_shape)

        if self.axis == 1:
            _, channels, height, width = input_shape
        else:
            _, height, width, channels = input_shape

        if self.relative:
            dk_per_head = self.depth_k // self.num_heads
            
            # print(dk_per_head)

            if dk_per_head == 0:
                print('dk per head', dk_per_head)

            self.key_relative_w = self.add_weight('key_rel_w',
                                                  shape=tf.TensorShape([2 * width - 1, dk_per_head]),
                                                  initializer=initializers.RandomNormal(stddev=dk_per_head ** -0.5))
            # 2 * width - 1

            self.key_relative_h = self.add_weight('key_rel_h',
                                                  shape=tf.TensorShape([2 * height - 1, dk_per_head]),
                                                  initializer=initializers.RandomNormal(stddev=dk_per_head ** -0.5))
            # 2 * height - 1

        else:
            self.key_relative_w = None
            self.key_relative_h = None

    def call(self, inputs, **kwargs):
        if self.axis == 1:
            # If channels first, force it to be channels last for these ops
            inputs = K.permute_dimensions(inputs, [0, 2, 3, 1])

        q, k, v = tf.split(inputs, [self.depth_k, self.depth_k, self.depth_v], axis=-1)

        q = self.split_heads_2d(q)
        k = self.split_heads_2d(k)
        v = self.split_heads_2d(v)

        # scale query
        depth_k_heads = self.depth_k / self.num_heads
        q *= (depth_k_heads ** -0.5)

        # [Batch, num_heads, height * width, depth_k or depth_v] if axis == -1
        qk_shape = [self._batch, self.num_heads, self._height * self._width, self.depth_k // self.num_heads]
        v_shape = [self._batch, self.num_heads, self._height * self._width, self.depth_v // self.num_heads]
        flat_q = K.reshape(q, K.stack(qk_shape))
        flat_k = K.reshape(k, K.stack(qk_shape))
        flat_v = K.reshape(v, K.stack(v_shape))

        # [Batch, num_heads, HW, HW]
        logits = tf.matmul(flat_q, flat_k, transpose_b=True)

        # Apply relative encodings
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits

        weights = K.softmax(logits, axis=-1)
        attn_out = tf.matmul(weights, flat_v)

        attn_out_shape = [self._batch, self.num_heads, self._height, self._width, self.depth_v // self.num_heads]
        attn_out_shape = K.stack(attn_out_shape)
        attn_out = K.reshape(attn_out, attn_out_shape)
        attn_out = self.combine_heads_2d(attn_out)
        # [batch, height, width, depth_v]

        if self.axis == 1:
            # return to [batch, depth_v, height, width] for channels first
            attn_out = K.permute_dimensions(attn_out, [0, 3, 1, 2])

        return attn_out

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = self.depth_v
        return tuple(output_shape)

    def split_heads_2d(self, ip):
        tensor_shape = K.shape(ip)

        # batch, height, width, channels for axis = -1
        tensor_shape = [tensor_shape[i] for i in range(len(self._shape))]

        batch = tensor_shape[0]
        height = tensor_shape[1]
        width = tensor_shape[2]
        channels = tensor_shape[3]

        # Save the spatial tensor dimensions
        self._batch = batch
        self._height = height
        self._width = width

        ret_shape = K.stack([batch, height, width,  self.num_heads, channels // self.num_heads])
        split = K.reshape(ip, ret_shape)
        transpose_axes = (0, 3, 1, 2, 4)
        split = K.permute_dimensions(split, transpose_axes)

        return split

    def relative_logits(self, q):
        shape = K.shape(q)
        # [batch, num_heads, H, W, depth_v]
        shape = [shape[i] for i in range(5)]

        height = shape[2]
        width = shape[3]

        rel_logits_w = self.relative_logits_1d(q, self.key_relative_w, height, width,
                                               transpose_mask=[0, 1, 2, 4, 3, 5])

        rel_logits_h = self.relative_logits_1d(
            K.permute_dimensions(q, [0, 1, 3, 2, 4]),
            self.key_relative_h, width, height,
            transpose_mask=[0, 1, 4, 2, 5, 3])

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, transpose_mask):
        rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = K.reshape(rel_logits, [-1, self.num_heads * H, W, 2 * W - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = K.reshape(rel_logits, [-1, self.num_heads, H, W, W])
        rel_logits = K.expand_dims(rel_logits, axis=3)
        rel_logits = K.tile(rel_logits, [1, 1, 1, H, 1, 1])
        rel_logits = K.permute_dimensions(rel_logits, transpose_mask)
        rel_logits = K.reshape(rel_logits, [-1, self.num_heads, H * W, H * W])
        return rel_logits

    def rel_to_abs(self, x):
        shape = K.shape(x)
        shape = [shape[i] for i in range(3)]
        B, Nh, L, = shape
        col_pad = K.zeros(K.stack([B, Nh, L, 1]))
        x = K.concatenate([x, col_pad], axis=3)
        flat_x = K.reshape(x, [B, Nh, L * 2 * L])
        flat_pad = K.zeros(K.stack([B, Nh, L - 1]))
        flat_x_padded = K.concatenate([flat_x, flat_pad], axis=2)
        final_x = K.reshape(flat_x_padded, [B, Nh, L + 1, 2 * L - 1])
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

    def combine_heads_2d(self, inputs):
        # [batch, num_heads, height, width, depth_v // num_heads]
        transposed = K.permute_dimensions(inputs, [0, 2, 3, 1, 4])
        # [batch, height, width, num_heads, depth_v // num_heads]
        shape = K.shape(transposed)
        shape = [shape[i] for i in range(5)]

        a, b = shape[-2:]
        ret_shape = K.stack(shape[:-2] + [a * b])
        # [batch, height, width, depth_v]
        return K.reshape(transposed, ret_shape)

    def get_config(self):
        config = {
            'depth_k': self.depth_k,
            'depth_v': self.depth_v,
            'num_heads': self.num_heads,
            'relative': self.relative,
        }
        base_config = super(AttentionAugmentation2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def augmented_conv2d(ip, filters, kernel_size=(3, 3), strides=(1, 1),
                     depth_k=0.2, depth_v=0.2, num_heads=8, relative_encodings=True):
    """
    Builds an Attention Augmented Convolution block.
    Args:
        ip: keras tensor.
        filters: number of output filters.
        kernel_size: convolution kernel size.
        strides: strides of the convolution.
        depth_k: float or int. Number of filters for k.
            Computes the number of filters for `v`.
            If passed as float, computed as `filters * depth_k`.
        depth_v: float or int. Number of filters for v.
            Computes the number of filters for `k`.
            If passed as float, computed as `filters * depth_v`.
        num_heads: int. Number of attention heads.
            Must be set such that `depth_k // num_heads` is > 0.
        relative_encodings: bool. Whether to use relative
            encodings or not.
    Returns:
        a keras tensor.
    """
    # input_shape = K.int_shape(ip)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    depth_k, depth_v = _normalize_depth_vars(depth_k, depth_v, filters)

    conv_out = _conv_layer(filters - depth_v, kernel_size, strides)(ip)

    # Augmented Attention Block
    qkv_conv = _conv_layer(2 * depth_k + depth_v, (1, 1), strides)(ip)
    attn_out = AttentionAugmentation2D(depth_k, depth_v, num_heads, relative_encodings)(qkv_conv)
    attn_out = _conv_layer(depth_v, kernel_size=(1, 1))(attn_out)

    output = concatenate([conv_out, attn_out], axis=channel_axis)
    return output

def augmented_conv1d(ip, shape, filters, kernel_size=3, strides=1, padding = 'same',
                     depth_k=0.2, depth_v=0.2, num_heads=2, relative_encodings=True):
    """
    Builds an Attention Augmented Convolution block.
    Args:
        ip: keras tensor.
        filters: number of output filters.
        kernel_size: convolution kernel size.
        strides: strides of the convolution.
        depth_k: float or int. Number of filters for k.
            Computes the number of filters for `v`.
            If passed as float, computed as `filters * depth_k`.
        depth_v: float or int. Number of filters for v.
            Computes the number of filters for `k`.
            If passed as float, computed as `filters * depth_v`.
        num_heads: int. Number of attention heads.
            Must be set such that `depth_k // num_heads` is > 0.
        relative_encodings: bool. Whether to use relative
            encodings or not.
    Returns:
        a keras tensor.
    """

    
    if type(kernel_size) == int:
        pass
    else:
        kernel_size = kernel_size[0]
        
    if type(strides) == int:
        pass
    else:
        strides = strides[0]
        
    t_n = shape[0]
    f_n = shape[1]
        
    # input_shape = K.int_shape(ip)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    depth_k, depth_v = _normalize_depth_vars(depth_k, depth_v, filters)
    
    # print(kernel_size)
    # print(strides)

    conv_out = _conv_layer1d(ip, t_n, f_n, filters - depth_v, kernel_size, strides, padding = 'same')

    # Augmented Attention Block
    qkv_conv = _conv_layer1d(ip, t_n, f_n,  2 * depth_k + depth_v, 1, strides, padding = 'same')
    attn_out = AttentionAugmentation2D(depth_k, depth_v, num_heads, relative_encodings)(qkv_conv)
    attn_out = _conv_layer1r(attn_out, t_n, depth_v,  depth_v, 1, strides, padding = 'same')
    
    output = keras.layers.concatenate([conv_out, attn_out], axis=-1)
#     output = Concatenate(axis=channel_axis)([conv_out, attn_out])
   
    reshape = Reshape((t_n, filters))(output)

    return reshape
    #last option change filters to *2, *3  like deepdta



def build_combined_categorical(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
   
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32') ### Buralar flagdan gelmeliii
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')

    ### SMI_EMB_DINMS  FLAGS GELMELII 
    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)(XDinput) 
    # encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    # encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    # encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    x1 = (encode_smiles.shape[1])
    y1 = (encode_smiles.shape[2])

    x_smiles = augmented_conv1d(encode_smiles, shape = (x1, y1), filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,
                             strides = 1,
                             padding = 'valid', # if causal convolution is needed
                             depth_k=4, depth_v=4,  
                             num_heads=4, relative_encodings=True)
    
    encode_smiles = GlobalMaxPooling1D()(x_smiles)


    encode_protein = Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128, input_length=FLAGS.max_seq_len)(XTinput)
    # encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    # encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    # encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    x2 = (encode_protein.shape[1])
    y2 = (encode_protein.shape[2])
#uncomment -- 100
    x_protein = augmented_conv1d(encode_protein, shape = (x2, y2), filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,
                             strides = 1,
                             padding = 'valid', # if causal convolution is needed
                             depth_k=4, depth_v=4,  
                             num_heads=4, relative_encodings=True)
    
    encode_protein = GlobalMaxPooling1D()(x_protein)


    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected 
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC2) #OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']
    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_combined_categorical.png')

    return interactionModel



def build_single_drug(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
   
    interactionModel = Sequential()
    XTmodel = Sequential()
    XTmodel.add(Activation('linear', input_shape=(FLAGS.target_count,)))


    encode_smiles = Sequential()
    encode_smiles.add(Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)) 
    encode_smiles.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)) #input_shape=(MAX_SMI_LEN, SMI_EMBEDDING_DIMS)
    encode_smiles.add(Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1))
    encode_smiles.add(Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1))
    encode_smiles.add(GlobalMaxPooling1D())


    interactionModel.add(Merge([encode_smiles, XTmodel], mode='concat', concat_axis=1))
    #interactionModel.add(layers.merge.Concatenate([XDmodel, XTmodel]))

    # Fully connected 
    interactionModel.add(Dense(1024, activation='relu')) #1024
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(1024, activation='relu')) #1024
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(512, activation='relu')) 


    interactionModel.add(Dense(1, kernel_initializer='normal'))
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_single_drug.png')

    return interactionModel


def build_single_prot(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
   
    interactionModel = Sequential()
    XDmodel = Sequential()
    XDmodel.add(Activation('linear', input_shape=(FLAGS.drugcount,)))


    XTmodel1 = Sequential()
    XTmodel1.add(Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128,  input_length=FLAGS.max_seq_len))
    XTmodel1.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)) #input_shape=(MAX_SEQ_LEN, SEQ_EMBEDDING_DIMS)
    XTmodel1.add(Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1))
    XTmodel1.add(Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1))
    XTmodel1.add(GlobalMaxPooling1D())


    interactionModel.add(Merge([XDmodel, XTmodel1], mode='concat', concat_axis=1))

    # Fully connected 
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(512, activation='relu'))

    interactionModel.add(Dense(1, kernel_initializer='normal'))
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_single_protein.png')

    return interactionModel

def build_baseline(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    interactionModel = Sequential()

    XDmodel = Sequential()
    XDmodel.add(Dense(1, activation='linear', input_shape=(FLAGS.drug_count, )))

    XTmodel = Sequential()
    XTmodel.add(Dense(1, activation='linear', input_shape=(FLAGS.target_count,)))


    interactionModel.add(Merge([XDmodel, XTmodel], mode='concat', concat_axis=1))

    # Fully connected 
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(512, activation='relu'))

    interactionModel.add(Dense(1, kernel_initializer='normal'))
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_baseline.png')

    return interactionModel

def nfold_1_2_3_setting_sample(XD, XT,  Y, label_row_inds, label_col_inds, measure, runmethod,  FLAGS, dataset):

    bestparamlist = []
    test_set, outer_train_sets = dataset.read_sets(FLAGS) 
    
    foldinds = len(outer_train_sets)

    test_sets = []
    ## TRAIN AND VAL
    val_sets = []
    train_sets = []

    #logger.info('Start training')
    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        train_sets.append(otherfoldsinds)
        test_sets.append(test_set)
        print("val set", str(len(val_fold)))
        print("train set", str(len(otherfoldsinds)))



    bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = general_nfold_cv(XD, XT,  Y, label_row_inds, label_col_inds, 
                                                                                                measure, runmethod, FLAGS, train_sets, val_sets)
   
    #print("Test Set len", str(len(test_set)))
    #print("Outer Train Set len", str(len(outer_train_sets)))
    bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(XD, XT,  Y, label_row_inds, label_col_inds, 
                                                                                                measure, runmethod, FLAGS, train_sets, test_sets)
    
    testperf = all_predictions[bestparamind]##pointer pos 

    logging("---FINAL RESULTS-----", FLAGS)
    logging("best param index = %s,  best param = %.5f" % 
            (bestparamind, bestparam), FLAGS)


    testperfs = []
    testloss= []

    avgperf = 0.

    for test_foldind in range(len(test_sets)):
        foldperf = all_predictions[bestparamind][test_foldind]
        foldloss = all_losses[bestparamind][test_foldind]
        testperfs.append(foldperf)
        testloss.append(foldloss)
        avgperf += foldperf

    avgperf = avgperf / len(test_sets)
    avgloss = np.mean(testloss)
    teststd = np.std(testperfs)

    logging("Test Performance CI", FLAGS)
    logging(testperfs, FLAGS)
    logging("Test Performance MSE", FLAGS)
    logging(testloss, FLAGS)

    return avgperf, avgloss, teststd




def general_nfold_cv(XD, XT,  Y, label_row_inds, label_col_inds, prfmeasure, runmethod, FLAGS, labeled_sets, val_sets): ## BURAYA DA FLAGS LAZIM????
    
    paramset1 = FLAGS.num_windows                              #[32]#[32,  512] #[32, 128]  # filter numbers
    paramset2 = FLAGS.smi_window_lengths                               #[4, 8]#[4,  32] #[4,  8] #filter length smi
    paramset3 = FLAGS.seq_window_lengths                               #[8, 12]#[64,  256] #[64, 192]#[8, 192, 384]
    epoch = FLAGS.num_epoch                                 #100
#    epoch = 1                                 #100
    batchsz = FLAGS.batch_size                             #256

    logging("---Parameter Search-----", FLAGS)

    w = len(val_sets)
    h = len(paramset1) * len(paramset2) * len(paramset3)

    all_predictions = [[0 for x in range(w)] for y in range(h)] 
    all_losses = [[0 for x in range(w)] for y in range(h)] 
    print(all_predictions)

    for foldind in range(len(val_sets)):
#    for foldind in range(1):
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]

        Y_train = np.mat(np.copy(Y))

        params = {}
        XD_train = XD
        XT_train = XT
        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]

        XD_train = XD[trrows]
        XT_train = XT[trcols]

        train_drugs, train_prots,  train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)
        
        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]
        #print("terows", str(terows), str(len(terows)))
        #print("tecols", str(tecols), str(len(tecols)))

        val_drugs, val_prots,  val_Y = prepare_interaction_pairs(XD, XT,  Y, terows, tecols)


        pointer = 0
       
        for param1ind in range(len(paramset1)): #hidden neurons
            param1value = paramset1[param1ind]
            for param2ind in range(len(paramset2)): #learning rate
                param2value = paramset2[param2ind]

                for param3ind in range(len(paramset3)):
                    param3value = paramset3[param3ind]

                    gridmodel = runmethod(FLAGS, param1value, param2value, param3value)
                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
                    gridres = gridmodel.fit(([np.array(train_drugs),np.array(train_prots) ]), np.array(train_Y), batch_size=batchsz, epochs=epoch, 
                            validation_data=( ([np.array(val_drugs), np.array(val_prots) ]), np.array(val_Y)),  shuffle=False, callbacks=[es] ) 


                    predicted_labels = gridmodel.predict([np.array(val_drugs), np.array(val_prots) ])
                    loss, rperf2 = gridmodel.evaluate(([np.array(val_drugs),np.array(val_prots) ]), np.array(val_Y), verbose=0)
                    rperf = prfmeasure(val_Y, predicted_labels)
                    rperf = rperf[0]


                    logging("P1 = %d,  P2 = %d, P3 = %d, Fold = %d, CI-i = %f, CI-ii = %f, MSE = %f" % 
                    (param1ind, param2ind, param3ind, foldind, rperf, rperf2, loss), FLAGS)

                    plotLoss(gridres, param1ind, param2ind, param3ind, foldind)

                    all_predictions[pointer][foldind] =rperf #TODO FOR EACH VAL SET allpredictions[pointer][foldind]
                    all_losses[pointer][foldind]= loss

                    pointer +=1

    bestperf = -float('Inf')
    bestpointer = None


    best_param_list = []
    ##Take average according to folds, then chooose best params
    pointer = 0
    for param1ind in range(len(paramset1)):
            for param2ind in range(len(paramset2)):
                for param3ind in range(len(paramset3)):
                
                    avgperf = 0.
                    for foldind in range(len(val_sets)):
                        foldperf = all_predictions[pointer][foldind]
                        avgperf += foldperf
                    avgperf /= len(val_sets)
                    #print(epoch, batchsz, avgperf)
                    if avgperf > bestperf:
                        bestperf = avgperf
                        bestpointer = pointer
                        best_param_list = [param1ind, param2ind, param3ind]

                    pointer +=1
        
    return  bestpointer, best_param_list, bestperf, all_predictions, all_losses



def cindex_score(y_true, y_pred):

    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.compat.v1.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f) #select


   
def plotLoss(history, batchind, epochind, param3ind, foldind):

    figname = "b"+str(batchind) + "_e" + str(epochind) + "_" + str(param3ind) + "_"  + str( foldind) + "_" + str(time.time()) 
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
	#plt.legend(['trainloss', 'valloss', 'cindex', 'valcindex'], loc='upper left')
    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.savefig("figures/"+figname +".png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait', 
                    papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
    plt.close()


    ## PLOT CINDEX
    plt.figure()
    plt.title('model concordance index')
    plt.ylabel('cindex')
    plt.xlabel('epoch')
    plt.plot(history.history['cindex_score'])
    plt.plot(history.history['val_cindex_score'])
    plt.legend(['traincindex', 'valcindex'], loc='upper left')
    plt.savefig("figures/"+figname + "_acc.png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait', 
                            papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
    plt.close()



def prepare_interaction_pairs(XD, XT,  Y, rows, cols):
    drugs = []
    targets = []
    targetscls = []
    affinity=[] 
        
    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target=XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind],cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data,target_data,  affinity


       
def experiment(FLAGS, perfmeasure, deepmethod, foldcount=6): #5-fold cross validation + test

    #Input
    #XD: [drugs, features] sized array (features may also be similarities with other drugs
    #XT: [targets, features] sized array (features may also be similarities with other targets
    #Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries
    #perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
    #higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
    #foldcount: number of cross-validation folds for settings 1-3, setting 4 always runs 3x3 cross-validation


    dataset = DataSet( fpath = FLAGS.dataset_path, ### BUNU ARGS DA GUNCELLE
                      setting_no = FLAGS.problem_type, ##BUNU ARGS A EKLE
                      seqlen = FLAGS.max_seq_len,
                      smilen = FLAGS.max_smi_len,
                      need_shuffle = False )
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size 
    FLAGS.charsmiset_size = dataset.charsmiset_size 

    XD, XT, Y = dataset.parse_data(FLAGS)

    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    drugcount = XD.shape[0]
    print(drugcount)
    targetcount = XT.shape[0]
    print(targetcount)

    FLAGS.drug_count = drugcount
    FLAGS.target_count = targetcount

    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)  #basically finds the point address of affinity [x,y]

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    print(FLAGS.log_dir)
    S1_avgperf, S1_avgloss, S1_teststd = nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds,
                                                                     perfmeasure, deepmethod, FLAGS, dataset)

    logging("Setting " + str(FLAGS.problem_type), FLAGS)
    logging("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" % 
            (S1_avgperf, S1_avgloss, S1_teststd), FLAGS)




def run_regression( FLAGS ): 

    perfmeasure = get_cindex
    deepmethod = build_combined_categorical

    experiment(FLAGS, perfmeasure, deepmethod)




if __name__=="__main__":
    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    logging(str(FLAGS), FLAGS)
    run_regression( FLAGS )
