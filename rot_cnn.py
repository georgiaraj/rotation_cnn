#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Flatten, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import math

class cnn(object):
    ###############################################################################
    # Methods currently established as beta/stable
    def __init__(self, model_name, model_path, im_size, use_fullconv=False, batch_size=128, nepochs=100, use_bn=True, lr=0.001, decay=1e-5, pdrop_conv=0., pdrop_fc=0., fc_layers=[512], fcn_head_row=None, fcn_head_col=None, padding='same'):
        # Sanity check for fully convolutional

        if padding != 'same' and padding != 'valid':
            print('ParamError: Conv2D padding can either be \'valid\' or \'same\'.')
            sys.exit(0)

        if use_fullconv:
            mode_fullconv = '_fullconv'
        else:
            mode_fullconv = ''

        if padding == 'same':
            padding_text = '_'+padding
        else:
            padding_text = '_valid'

        self.model_name = model_name
        self.model_path = model_path+mode_text+mode_fullconv+padding_text+'_batch'+str(batch_size)+'_nepochs'+str(nepochs)+'_bn'+str(use_bn)+'_dropconv'+str(pdrop_conv)+'_dropfc'+str(pdrop_fc)+'.cnn'
        self.patch_size = patch_size
        self.multi_offset = multi_offset
        self.batch_size = batch_size # Number of training images used per iteration
        self.nepochs = nepochs # Max number of epochs in training; 1 epoch is when we pass through all training data
        self.use_bn = use_bn # Triggers batch normalization just before the activation functions on the inner layer; NOT applied in readout layer
        self.lr = lr # Learning rate in Adam optimizer
        self.decay = decay # decay learning factor in Adam optimizer
        self.pdrop_conv = pdrop_conv # percentage of neurons dropped in convolution layers, wherever Dropout is used; if set to '0', no dropout occurs
        self.pdrop_fc = pdrop_fc # percentage of neurons dropped in fully connected layer; if set to '0', no dropout occurs
        self.fc_layers = fc_layers # List of the number of nodes per fully connected inner layer
        self.use_fullconv = use_fullconv
        self.fcn_head_row = fcn_head_row
        self.fcn_head_col = fcn_head_col
        self.padding = padding

        self.model = None
        self.scaling_factor = None
        self.kernel_padding = None

    def train(self, images, labels):
        # Divide data into training and validation samples in ratio 4:1
        ntrain = round(patches.shape[0]*4/5)
        patches_train = np.expand_dims(patches[0:ntrain,:,:],axis=3)
        labels_train = labels[0:ntrain,:]
        patches_val = np.expand_dims(patches[ntrain:patches.shape[0],:,:],axis=3)
        labels_val = labels[ntrain:patches.shape[0],:]

        # Setup in the tensorflow session
        sess = tf.Session()
        K.set_session(sess)

        # Build the graph
        #img = tf.placeholder(tf.float32, shape=(None, patch_size))
        nrows = patches.shape[1]
        ncols = patches.shape[2]
        nchannels = 1
        use_custom_fcn_regressor_head = False

        # Experimental code. Inference logic to build FCN graph with dynamic input; Requires manually setting the head of the FCN regressor
        if self.use_fullconv and self.fcn_head_col is not None and self.fcn_head_row is not None:
             nrows = None
             ncols = None
             use_custom_fcn_regressor_head = True

        patch_shape = (nrows, ncols, nchannels)
        if use_custom_fcn_regressor_head:
            model = self.__build_graph(patch_shape, padding=self.padding, fcn_head_row=self.fcn_head_row, fcn_head_col=self.fcn_head_col)
        else:
            model = self.__build_graph(patch_shape, padding=self.padding)

        model.summary() #print neural network summary

        patience = 3

        # Set so that the training stops when the validation loss doesn't improve for 3 epochs
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
        ]

        # Set optimisation and loss functions
        opt = keras.optimizers.Adam(lr=self.lr, decay=self.decay)
        print('Using Adam optimizer')
        print(opt.get_config())
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])

        # Actually do the training; If fully convolutional reshape labels_train
        if self.use_fullconv:
            str_shape_original = '{0}'.format(labels_train.shape)
            labels_train = np.reshape(labels_train, (labels_train.shape[0], 1, 1, labels_train.shape[1]))
            labels_val = np.reshape(labels_val, (labels_val.shape[0], 1, 1, labels_val.shape[1]))
            str_shape_new = '{0}'.format(labels_train.shape)
            print('Using Fully Convolutional Network')
            print('Reshaping \'labels_train\' and \'labels_val\'  from: ' + str_shape_original + ' to: ' + str_shape_new)

        model.fit(patches_train, labels_train, batch_size=self.batch_size, epochs=self.nepochs, validation_data=(patches_val, labels_val), shuffle=True, callbacks=callbacks)

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

    def reshape_input_layer_full_convnet(self, new_patch_shape, padding='same', verbose=False):
        # Check that we are using a fully convolutional network
        if self.is_trained() == False or self.use_fullconv == False:
            print('Error. \'reshape_input_layer_full_convnet\' is only supported for trained Full ConvNets2D')
            sys.exit(0)

        original_shape = self.model.layers[0].input_shape
        # Check that original trained network input layer is trained for dynamic input (this is whan input layer shape (None, None, None, 1)
        # If trained for dynamic input, no reshaped/new network is required
        if original_shape[1] is None and original_shape[2] is None:
            return

        # Check whether it is necessary to reshape based on expected images' rows and cols
        if new_patch_shape[0:2] == original_shape[1:3]:
            return

        # Find head of fcn shape(nrows,ncols) from loaded/trained network. This would be the first Layer2D that is not (None,1,1,None)
        # This is provided that no Maxpooling or strides>1 etc techniques were used on the fcn part of the network
        fcn_head_row = None
        fcn_head_col = None
        for layer in reversed(self.model.layers):
            if layer.output_shape[1] != 1:
                fcn_head_row = layer.output_shape[1]
                fcn_head_col = layer.output_shape[2]
                break
        new_model = self.__build_graph(new_patch_shape, padding, fcn_head_row, fcn_head_col)
        #Replace weights and reset model
        for n, layer in  enumerate(self.model.layers):
            new_model.layers[n].set_weights(layer.get_weights())

        if verbose:
             print('Original Full-ConvNet summary')
             self.model.summary()
             print('Reshaped Full-ConvNet summary')
             new_model.summary()

        self.model = new_model

    def set_session(self):
        sess = tf.Session()
        K.set_session(sess)

    def load_model(self):
        # Load model parameters
        self.model = load_model(self.model_path)

    def summary(self):
        self.model.summary()

    def test(self, patches, use_beta=False):
        # Prepare network for variable size images onthe FullConvNet case
        if self.use_fullconv:
            if use_beta is False:
                print('Error in \'cnn.test\'. Fully ConvNets are still beta. Set \'use_beta=True\' to test.')
                exit(0)
            #Check that input images require for the network to reset its input layer; check is done inside reshape method
            nrows = patches.shape[1]
            ncols = patches.shape[2]
            nchannels = 1
            new_shape = (nrows, ncols, nchannels)
            self.reshape_input_layer_full_convnet(new_shape, padding=self.padding, verbose=False)
        labels = self.model.predict(patches, batch_size=self.batch_size, verbose=1)

        return labels

    def test_shifted_images(self, image):
        if self.use_fullconv is False:
            print('ParamError: \'test_shifted_images\' is only supported for FullyConvNets.')
            sys.exit(0)
        if image.shape[0] != 1:
            print('ParamError: \'test_shifted_images\' currently only supports one query image at a time.')
            sys.exit(0)

        # 1. Check whether the input images require for the network to reset its input layer; check is done inside reshape method
        n_rows_out = image.shape[1]
        n_cols_out = image.shape[2]
        nchannels = 1
        new_shape = (n_rows_out, n_cols_out, nchannels)
        self.reshape_input_layer_full_convnet(new_shape, padding='same', verbose=False)

        # Calculate the scale factor from the network architecture
        scaling_factor = self.get_scaling_factor()

        # Compute all slided regression maps.
        ll_labels = []
        for i_r_slide in range(scaling_factor):
            l_labels = []
            for i_c_slide in range(scaling_factor):
                # Current slide map to query cnn
                cur_image = np.zeros(image.shape)
                cur_image[:, 0:n_rows_out-i_r_slide, 0:n_cols_out-i_c_slide, :] = image[:, i_r_slide:, i_c_slide:, :]
                labels = self.model.predict(cur_image, batch_size=2, verbose=1)
                l_labels.append(labels)
            ll_labels.append(l_labels)

        return ll_labels

    def get_cnn(self):
        return self.model

    def get_scaling_factor(self):
        if self.model is None:
            self.load_model()

        if self.scaling_factor is None:
            scaling = 1
            for layer in self.model.layers:
                config = layer.get_config()
                if 'mp' in config['name']:
                    scaling *= int(config['pool_size'][0])

            print('Calculated scaling factor:',scaling)
            self.scaling_factor = scaling

        return self.scaling_factor

    def get_kernel_padding(self):
        if self.model is None:
            self.load_model()

        if self.kernel_padding is None:
            for layer in self.model.layers:
                config = layer.get_config()
                if 'conv_fcn0' in config['name']:
                    self.kernel_padding = (layer.input_shape[1] - 1)/2
                    break
            print('Calculated padding factor:',self.kernel_padding)

        return self.kernel_padding

    # Method to test the reshaping sanity of trainl labels for fully convolutional
    def __test_reshape_trainlabels_for_fcv(self, labels_train):
        test = np.reshape(labels_train,(labels_train.shape[0], 1, 1, labels_train.shape[1]))
        print('Input shape: {0}'.format(labels_train.shape))
        print('Output shape: {0}'.format(test.shape))
        ok_reshape_logic = test[:, 0, 0, :] == labels_train
        if np.all(ok_reshape_logic) == True:
            print('Reshaped values \'labels_train\' for fully convolutional network successfully')
        else:
            print('Reshaped values \'labels_train\' for fully convolutional network failed')

    # patch_shape = (nrows, ncols, nchannels)
    def __build_graph(self, patch_shape, padding='same', fcn_head_row=None, fcn_head_col=None):
        # Step 1. Build inner convnet layers
        model = self.__build_convnet2d_base(patch_shape, padding)
        # Step 2. Append with outer network convolutional layers; (a). Fully convolutional
        if self.use_fullconv:
            model = self.__add_outer_convnet2d_regression(model, fcn_head_row, fcn_head_col)
        # (b). Fully connected
        else:
            model = self.__add_outer_fcnet_regression(model, patch_shape)

        return model

    def __build_convnet2d_base(self, patch_shape, padding = 'same'):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding=padding, input_shape=patch_shape, name=self.model_name+'_conv1'))
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
        if self.pdrop_conv > 0.:
            model.add(Dropout(self.pdrop_conv, name=self.model_name+'_drop3'))

        model.add(Conv2D(64, (5, 5), padding=padding, name=self.model_name+'_conv7'))
        if self.use_bn:
            model.add(BatchNormalization(name=self.model_name+'_bn7'))
        model.add(Activation('relu', name=self.model_name+'_act7'))
        model.add(Conv2D(64, (5, 5), padding=padding, name=self.model_name+'_conv8'))
        if self.use_bn:
            model.add(BatchNormalization(name=self.model_name+'_bn8'))
        model.add(Activation('relu', name=self.model_name+'_act8'))
        model.add(MaxPooling2D(pool_size=(2, 2), name=self.model_name+'_mp2'))
        if self.pdrop_conv > 0.:
            model.add(Dropout(self.pdrop_conv, name=self.model_name+'_drop4'))

        return model

    # Regressor with fully connected layers. Assumes that model, is inner convnet2d
    def __add_outer_fcnet_regression(self, model, patch_shape):
        # Sanity checks
        self.__checks_outer_layers(model)

        # Dense fully connected layers
        model.add(Flatten(), name=self.model_name+'_flat1')
        for idx, n_nodes in enumerate(self.fc_layers):
            model = self.__add_fc_layer(idx, model, n_nodes)

        # Output layer
        if self.multi_offset:
            model.add(Dense(patch_shape[0] * patch_shape[1] * 2, name=self.model_name+'_output'))
        else:
            model.add(Dense(2, name=self.model_name+'_output'))
        model.add(Activation('linear', name=self.model_name+'_act_out'))

        return model

    def __add_fc_layer(self, idx, model, units):
        model.add(Dense(units, name=self.model_name+'_fc'+str(idx)))
        if self.use_bn:
            model.add(BatchNormalization(name=self.model_name+'_bn_fc'+str(idx)))
        model.add(Activation('relu', name=self.model_name+'_act_fc'+str(idx)))
        if self.pdrop_fc > 0.:
            model.add(Dropout(self.pdrop_fc, name=self.model_name+'_drop_fc'+str(idx)))

        return model

    # Regressor with fully convolutional layers. Assumes that model, is inner convnet2d
    def __add_outer_convnet2d_regression(self, model, fcn_head_row=None, fcn_head_col=None):
        # Sanity checks
        self.__checks_outer_layers(model)

        # Get previous layer shape
        prv_layer = model.layers[-1]
        layer_shape = prv_layer.output_shape
        n_rows_lastfilter = layer_shape[1]
        n_cols_lastfilter = layer_shape[2]

        if not fcn_head_row is None and not fcn_head_col is None:#Just to make sure that the user does not set only one of those
            n_rows_lastfilter = fcn_head_row
            n_cols_lastfilter = fcn_head_col

        # Build fully convolutional second inner part of the network, equivalent to the inner FC layers part
        for index, n_nodes in enumerate(self.fc_layers):
            if index == 0:
                model.add(Conv2D(n_nodes, (n_rows_lastfilter, n_cols_lastfilter), name=self.model_name+'_conv_fcn'+str(index))) # Specially handle conv layer equivallent to the first inner FC layer
            else:
                model.add(Conv2D(n_nodes, (1, 1), name=self.model_name+'_conv_fcn'+str(index)))
            if self.use_bn:
                model.add(BatchNormalization(name=self.model_name+'_bn_fcn'+str(index)))
            model.add(Activation('relu', name=self.model_name+'_act_fcn'+str(index)))
            if self.pdrop_fc > 0.: #Note! Here This convlayer is equivalent to an FC layer; Thus assume the Dropout param for FCs
                model.add(Dropout(self.pdrop_fc, name=self.model_name+'_drop_fcn'+str(index)))

        # Add Output Conv2d layer
        if len(self.fc_layers) == 0:
            # !!We should test if this special case is legit; This is when no second inner layer part of the network is set
            model.add(Conv2D(2, (n_rows_lastfilter, n_cols_lastfilter), name=self.model_name+'_fcn_output'))
        else:
            model.add(Conv2D(2,(1, 1), name=self.model_name+'_fcn_output'))

        model.add(Activation('linear', name=self.model_name+'_fcn_output_act'))

        return model

    # Terminate if outer layer graph build checks fail
    def __checks_outer_layers(self, model):
        # Sanity check
        if len(model.layers) == 0:
            print('Error: Inner ConvNet2D is not initialized.')
            sys.exit(0)
