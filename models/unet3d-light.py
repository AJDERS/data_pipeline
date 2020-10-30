import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, Add, Conv2D
from .unet3d import _down_block, _resnet_block, _activation, _up_block
from tensorflow.keras.models import Model




def create_model(conf):
    target_size = conf['PREPROCESS_TRAIN'].getint('TargetSize')
    duration = conf['DATA'].getint('MovementDuration')
    num_filters = conf['MODEL'].getint('NumFilter')
    padding = conf['MODEL'].get('Padding')
    output_activation = conf['MODEL'].get('OutputActivation')
    up_sampling = conf['MODEL'].get('UpSampling')
    down_sampling = conf['MODEL'].get('DownSampling')
    activation = conf['MODEL'].get('Activation')
    conv_1d_later = conf['MODEL'].getboolean('Conv1DLater')
    initializer = conf['MODEL'].get('Initializer')
    batchnorm = conf['MODEL'].getboolean('BatchNorm')
    shortcut = conf['MODEL'].getboolean('Shortcut')
    dropout_rate = conf['MODEL'].getfloat('DropOutRate')
    dropout_wrn = conf['MODEL'].getboolean('DropOutWarning')
    temporal_down_scaling = conf['MODEL'].getboolean('TemporalDownScaling')
    stacks = conf['DATA'].getboolean('Stacks')
    tracks = conf['DATA'].getboolean('Tracks')
    assert stacks != tracks, 'One and only one output format must be set.'

    if temporal_down_scaling:
        strides = 2
        scale = 2
    else:
        strides = (2,2,1)
        scale = (2,2,1)


    input_size = (target_size, target_size, duration, 1)

    # target_size = ts
    # duration = d
    # Input layer,
    #                           ts x ts x d x 1
    input_data = Input(shape=input_size)

    # Encoder layer 0, 
    #                           ts x ts x 1 > ts/2 x ts/2 x 64
    e0, _ = _down_block(
        input_data,
        num_filters=num_filters,
        activation=activation,
        strides=strides,
        initializer=initializer,
        shortcut=shortcut,
        dropout_rate=dropout_rate,
        dropout_wrn=dropout_wrn,
        down_sampling=down_sampling,
        batchnorm=batchnorm,
        name='down_block_1'
    )

    # Conv layer,
    #                           ts/2 x ts/2 x 64 > ts/2 x ts/2 x 128
    conv0 = _resnet_block(
        e0,
        num_filters=num_filters*2, # 8
        activation=activation,
        initializer=initializer,
        dropout_rate=dropout_rate,
        dropout_wrn=dropout_wrn,
        batchnorm=batchnorm,
        name='middle_resnet'
    )

    # Decoder layer 0,
    #                           ts/2 x ts/2 x 128 > ts x ts x 64
    d0 = _up_block(
        conv0, ### e2???
        num_filters=num_filters*2, # 4
        activation=activation,
        initializer=initializer,
        conv_1d_later=conv_1d_later,
        shortcut=shortcut,
        dropout_rate=dropout_rate,
        dropout_wrn=dropout_wrn,
        up_sampling=up_sampling,
        scale=scale,
        batchnorm=batchnorm,
        name='up_block_1'
    )
    if stacks:
        output_data = Conv3D(
            filters=1,
            kernel_size=1,
            padding=padding,
            kernel_initializer=initializer,
            name='output_layer'
        )(d0)
        output_data = _activation(output_data, output_activation)

    if tracks:
        d0 = tf.keras.layers.Add()(
            [d0[:,:,:,f,:] for f in range(duration)]
        )
        output_data = Conv2D(
            filters=1,
            kernel_size=1,
            padding=padding,
            kernel_initializer=initializer,
            name='output_layer'
        )(d0)
        output_data = _activation(output_data, output_activation)
    return Model(inputs=input_data, outputs=output_data)