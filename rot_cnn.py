#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.metrics
from tensorflow.keras.utils import get_custom_objects

import math

def angle_difference(x, y):
    return 180 - abs(abs(x - y) - 180)

def angle_error(y_true, y_pred):
    return K.mean(K.abs(angle_difference(y_true, y_pred)), axis=-1)

def angle_square_error(y_true, y_pred):
    return K.mean(K.square(angle_difference(y_true, y_pred)), axis=-1)

# Create a new activation that caps the output at 359
def linear_angle_activation(x):
    return K.switch(x < 360, x, x*0)

class cnn(object):
    ###############################################################################
    # Methods currently established as beta/stable
    def __init__(self, model_name, model_path, batch_size=20, nepochs=200, use_bn=True, lr=0.01, decay=1e-5, pdrop_conv=0.25, pdrop_fc =0.5, fc_layers=[512,512,256], padding='same'):

        if padding != 'same' and padding != 'valid':
            print('ParamError: Conv2D padding can either be \'valid\' or \'same\'.')
            sys.exit(0)

        if padding == 'same':
            padding_text = '_'+padding
        else:
            padding_text = '_valid'

        self.model_name = model_name
        self.model_path = model_path+padding_text+'_batch'+str(batch_size)+'_nepochs'+str(nepochs)+'_bn'+str(use_bn)+'_dropconv'+str(pdrop_conv)+'_dropfc'+str(pdrop_fc)+'.cnn'
        self.batch_size = batch_size # Number of training images used per iteration
        self.nepochs = nepochs # Max number of epochs in training; 1 epoch is when we pass through all training data
        self.use_bn = use_bn # Triggers batch normalization just before the activation functions on the inner layer; NOT applied in readout layer
        self.lr = lr # Learning rate in Adam optimizer
        self.decay = decay # decay learning factor in Adam optimizer
        self.pdrop_conv = pdrop_conv # percentage of neurons dropped in convolution layers, wherever Dropout is used; if set to '0', no dropout occurs
        self.pdrop_fc = pdrop_fc # percentage of neurons dropped in fully connected layer; if set to '0', no dropout occurs
        self.fc_layers = fc_layers # List of the number of nodes per fully connected inner layer
        self.padding = padding

        self.model = None
        self.scaling_factor = None
        self.kernel_padding = None

    def train(self, images, labels):
        # Divide data into training and validation samples in ratio 4:1
        ntrain = round(images.shape[0]*9/10)
        images_train = np.expand_dims(images[0:ntrain,:,:],axis=3)
        labels_train = labels[0:ntrain]
        images_val = np.expand_dims(images[ntrain:images.shape[0],:,:],axis=3)
        labels_val = labels[ntrain:images.shape[0]]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        get_custom_objects().update({'linear_angle_activation': Activation(linear_angle_activation)})

        # Setup in the tensorflow session
        sess = tf.Session(config=config)
        K.set_session(sess)

        # Build the graph
        #img = tf.placeholder(tf.float32, shape=(None, image_size))
        nrows = images.shape[1]
        ncols = images.shape[2]
        nchannels = 1

        image_shape = (nrows, ncols, nchannels)
        model = self.__build_graph(image_shape, padding=self.padding)

        model.summary() #print neural network summary

        patience = 5

        # Set so that the training stops when the validation loss doesn't improve for 3 epochs
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
        ]

        # Set optimisation and loss functions
        opt = keras.optimizers.Adam(lr=self.lr, decay=self.decay)
        print('Using Adam optimizer')
        print(opt.get_config())

        model.compile(loss=angle_square_error, optimizer=opt, metrics=['mae',angle_error])

        print('Batch size:',self.batch_size)

        model.fit(images_train, labels_train, batch_size=self.batch_size, epochs=self.nepochs, validation_data=(images_val, labels_val), shuffle=True, callbacks=callbacks)

        # Save the model
        model.save(self.model_path)
        print('Saved trained model to',self.model_path)

    def is_trained(self):
        if os.path.isfile(self.model_path):
            print('Model',self.model_path,'already trained.')
            return True
        else:
            print('Model',self.model_path,'not trained yet.')
            return False

    def set_session(self):
        sess = tf.Session()
        K.set_session(sess)

    def load_model(self):

        # Load model parameters
        self.model = load_model(self.model_path, custom_objects={'angle_error':angle_error, 'angle_square_error':angle_square_error, 'linear_angle_activation':linear_angle_activation})

    def summary(self):
        self.model.summary()

    def test(self, images):
        images = np.expand_dims(images,axis=3)

        labels = self.model.predict(images, batch_size=self.batch_size, verbose=1)

        return labels

    def get_cnn(self):
        return self.model

    # image_shape = (nrows, ncols, nchannels)
    def __build_graph(self, image_shape, padding='same'):
        # Step 1. Build inner convnet layers
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding=padding, input_shape=image_shape, name=self.model_name+'_conv1'))
        if self.use_bn:  # Important! Conv2D uses default data_format="channels_last". For that case BatchNormalization(axis=-1 (default))
            model.add(BatchNormalization(name=self.model_name+'_bn1'))
        model.add(Activation('relu', name=self.model_name+'_act1'))
        model.add(Conv2D(32, (3, 3), padding=padding, name=self.model_name+'_conv2'))
        if self.use_bn:
            model.add(BatchNormalization(name=self.model_name+'_bn2'))
        model.add(Activation('relu', name=self.model_name+'_act2'))
        if self.pdrop_conv > 0.:
            model.add(Dropout(self.pdrop_conv, name=self.model_name+'_drop1'))

        model.add(Conv2D(32, (3, 3), padding=padding, name=self.model_name+'_conv3'))
        if self.use_bn:
            model.add(BatchNormalization(name=self.model_name+'_bn3'))
        model.add(Activation('relu', name=self.model_name+'_act3'))
        model.add(Conv2D(32, (3, 3), padding=padding, name=self.model_name+'_conv4'))
        if self.use_bn:
            model.add(BatchNormalization(name=self.model_name+'_bn4'))
        model.add(Activation('relu', name=self.model_name+'_act4'))
        model.add(MaxPooling2D(pool_size=(2, 2), name=self.model_name+'_mp1'))
        if self.pdrop_conv > 0.:
            model.add(Dropout(self.pdrop_conv, name=self.model_name+'_drop2'))


        model.add(Conv2D(64, (5, 5), padding=padding, name=self.model_name+'_conv5'))
        if self.use_bn:
            model.add(BatchNormalization(name=self.model_name+'_bn5'))
        model.add(Activation('relu', name=self.model_name+'_act5'))
        model.add(Conv2D(64, (5, 5), padding=padding, name=self.model_name+'_conv6'))
        if self.use_bn:
            model.add(BatchNormalization(name=self.model_name+'_bn6'))
        model.add(Activation('relu', name=self.model_name+'_act6'))
        model.add(MaxPooling2D(pool_size=(2, 2), name=self.model_name+'_mp2'))
        if self.pdrop_conv > 0.:
            model.add(Dropout(self.pdrop_conv, name=self.model_name+'_drop3'))

        # Dense fully connected layers
        model.add(Flatten(name=self.model_name+'_flat1'))

        for idx, n_nodes in enumerate(self.fc_layers):
            model.add(Dense(n_nodes, name=self.model_name+'_fc'+str(idx)))
            if self.use_bn:
                model.add(BatchNormalization(name=self.model_name+'_bn_fc'+str(idx)))
            model.add(Activation('relu', name=self.model_name+'_act_fc'+str(idx)))
            if self.pdrop_fc > 0.:
                model.add(Dropout(self.pdrop_fc, name=self.model_name+'_drop_fc'+str(idx)))

        model.add(Dense(1, name=self.model_name+'_output'))
        model.add(Activation(linear_angle_activation, name=self.model_name+'_act_out'))
        #model.add(Activation('linear', name=self.model_name+'_act_out'))

        return model
