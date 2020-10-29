import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, UpSampling2D, BatchNormalization, LeakyReLU, Concatenate, ReLU, Activation, \
    MaxPool3D, Conv3DTranspose, SeparableConv2D, Add, Dropout, AvgPool2D, Conv2D
from tensorflow.keras.models import Model, Sequential
from .layers.layers import SubPixel3D


def _activation(tensor, activation='lrelu', name=''):
    if activation == 'lrelu':
        tensor = LeakyReLU(name='activation_lrelu' + name)(tensor)
    elif activation == 'relu':
        tensor = ReLU(name='activation_relu' + name)(tensor)
    elif activation == 'sigmoid':
        tensor = Activation(
            activation, 
            name='activation_sigmoid' + name
        )(tensor)
    elif activation is None:
        pass
    else:
        raise NotImplementedError(
            'activation should be [sigmoid or relu or lrelu]'
        )
    return tensor


def _resnet_block(
    tensor,
    num_filters,
    kernel_size=3,
    padding='same',
    activation='lrelu',
    initializer='glorot_uniform',
    shortcut=False,
    dropout_rate=None,
    dropout_wrn=False,
    batchnorm=True,
    name=''
    ):

    # shortcut
    if shortcut:
        tensor_skip = tensor

    # BN
    if batchnorm:
        tensor = BatchNormalization()(tensor)

    # Activation
    tensor = _activation(tensor, activation, name+'_1')

    # weight
    tensor = Conv3D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding=padding,
        kernel_initializer=initializer
    )(tensor)

    # wrn dropout
    if dropout_wrn and dropout_rate is not None:
        tensor = Dropout(rate=dropout_rate)(tensor)

    # BN
    if batchnorm:
        tensor = BatchNormalization()(tensor)

    # Activation
    tensor = _activation(tensor, activation, name+'_2')

    # weight
    tensor = Conv3D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding=padding,
        kernel_initializer=initializer
    )(tensor)

    # shortcut
    if shortcut:
        tensor = Add()([tensor, tensor_skip])

    # dropout
    if not dropout_wrn and dropout_rate is not None:
        tensor = Dropout(rate=dropout_rate)(tensor)

    return tensor


def _down_block(
    tensor,
    num_filters,
    kernel_size=3,
    padding='same',
    strides=2,
    shortcut=True,
    activation='lrelu',
    dropout_rate=None,
    dropout_wrn=False,
    down_sampling='strided_conv',
    initializer='orthogonal',
    batchnorm=True,
    name=''
    ):

    tensor = Conv3D(kernel_size=1, filters=num_filters)(tensor)
    tensor = _resnet_block(
        tensor,
        num_filters,
        kernel_size,
        shortcut=shortcut,
        padding=padding,
        activation=activation,
        initializer='orthogonal',
        dropout_rate=dropout_rate,
        dropout_wrn=dropout_wrn,
        batchnorm=batchnorm,
        name=name
        )

    skip_tensor = tensor

    # down-sampling
    if down_sampling == 'strided_conv':
        tensor = Conv3D(
            filters=num_filters,
            kernel_size=kernel_size*2-1,
            strides=strides,
            padding=padding,
            kernel_initializer=initializer
        )(tensor)

    elif down_sampling == 'maxpool':
        tensor = Conv3D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=initializer
        )(tensor)

        tensor = MaxPool3D(strides)(tensor)

    elif down_sampling == 'avgpool':
        tensor = Conv3D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=initializer
        )(tensor)

        tensor = AvgPool2D(strides)(tensor)

    else:
        raise ValueError(
            'down_sampling should be one of [ \'strided_conv\', \'maxpool\' ]'
        )

    return tensor, skip_tensor


def _up_block(
    tensor,
    num_filters,
    kernel_size=3,
    padding='same',
    up_sampling='simple',
    scale=(2, 2, 2),
    activation='lrelu',
    shortcut=True,
    dropout_rate=None,
    dropout_wrn=False,
    initializer='orthogonal',
    conv_1d_later=True,
    batchnorm=True,
    name=''
    ):

    tensor = Conv3D(
        filters=num_filters,
        kernel_size=1,
        padding=padding
    )(tensor)
    
    tensor = _resnet_block(
        tensor,
        num_filters,
        kernel_size,
        padding=padding,
        activation=activation,
        initializer=initializer,
        shortcut=shortcut,
        dropout_rate=dropout_rate,
        dropout_wrn=dropout_wrn,
        batchnorm=batchnorm,
        name=name
    )

    # up-sampling
    if up_sampling == 'simple':
        tensor = UpSampling2D(scale)(tensor)
        tensor = Conv3D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=initializer
        )(tensor)

    elif up_sampling == 'transpose':
        tensor = Conv3DTranspose(
            filters=num_filters,
            kernel_size=kernel_size,
            strides=scale,
            padding=padding,
            kernel_initializer=initializer
        )(tensor)

    elif up_sampling == 'subpixel':
        if not conv_1d_later:
            tensor = SubPixel3D(scale=scale)(tensor)
            tensor = Conv3D(
                filters=num_filters,
                kernel_size=1,
                padding=padding,
                kernel_initializer=initializer
            )(tensor)

        else:
            tensor = Conv3D(
                filters=num_filters*scale[0]*scale[1],
                kernel_size=1,
                padding=padding,
                kernel_initializer=initializer
            )(tensor)

            tensor = SubPixel3D(scale=scale)(tensor)

    elif up_sampling == 'simple_subpixel':
        tensor = SubPixel3D(scale=scale)(tensor)

    elif up_sampling == 'conv_subpixel':
        tensor = Conv3D(
            filters=num_filters*scale[0],
            kernel_size=1,
            padding=padding,
            kernel_initializer=initializer
        )(tensor)

        tensor = SubPixel3D(scale=scale)(tensor)
    else:
        raise ValueError(
            'up_sampling should be one of '
            '[ \'simple\', \'transposed\', \'subpixel\' ]'
        )
    return tensor

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
    skip_connection = conf['MODEL'].get('SkipConnection')
    dropout_encoder_only = conf['MODEL'].getboolean('DropOutEncoderOnly')
    stacks = conf['DATA'].getboolean('Stacks')
    tracks = conf['DATA'].getboolean('Tracks')
    assert stacks != tracks, 'One and only one output format must be set.'


    input_size = (target_size, target_size, duration, 1)

    # target_size = ts
    # duration = d
    # Input layer,
    #                           ts x ts x d x 1
    input_data = Input(shape=input_size)

    # Encoder layer 0, 
    #                           ts x ts x 1 > ts/2 x ts/2 x 64
    e0, s0 = _down_block(
        input_data,
        num_filters=num_filters,
        activation=activation,
        strides=2,
        initializer=initializer,
        shortcut=shortcut,
        dropout_rate=dropout_rate,
        dropout_wrn=dropout_wrn,
        down_sampling=down_sampling,
        batchnorm=batchnorm,
        name='down_block_1'
    )

    # Encoder layer 1,
    #                           ts/2 x ts/2 x 64 > ts/4 x ts/4 x 128
    e1, s1 = _down_block(
        e0,
        num_filters=num_filters*2,
        activation=activation,
        strides=2,
        initializer=initializer,
        shortcut=shortcut,
        dropout_rate=dropout_rate,
        dropout_wrn=dropout_wrn,
        down_sampling=down_sampling,
        batchnorm=batchnorm,
        name='down_block_2'
    )

    # Encoder layer 2,
    #                           ts/4 x ts/4 x 128 > ts/8 x ts/8 x 256
    e2, s2 = _down_block(
        e1,
        num_filters=num_filters*4,
        activation=activation,
        strides=2,
        initializer=initializer,
        shortcut=shortcut,
        dropout_rate=dropout_rate,
        dropout_wrn=dropout_wrn,
        down_sampling=down_sampling,
        batchnorm=batchnorm,
        name='down_block_3'
    )  

    if dropout_encoder_only:
        dropout_rate = None

    # Conv layer,
    #                           ts/8 x ts/8 x 256 > ts/8 x ts/8 x 512
    conv0 = _resnet_block(
        e2,
        num_filters=num_filters*16, # 8
        activation=activation,
        initializer=initializer,
        dropout_rate=dropout_rate,
        dropout_wrn=dropout_wrn,
        batchnorm=batchnorm,
        name='middle_resnet'
    )

    # Decoder layer 0,
    #                           ts/8 x ts/8 x 512 > ts/4 x ts/4 x 128 > ts/4 x ts/4 x 256
    d0 = _up_block(
        e2, ### e2???
        num_filters=num_filters*8, # 4
        activation=activation,
        initializer=initializer,
        conv_1d_later=conv_1d_later,
        shortcut=shortcut,
        dropout_rate=dropout_rate,
        dropout_wrn=dropout_wrn,
        up_sampling=up_sampling,
        batchnorm=batchnorm,
        name='up_block_1'
    )

    if skip_connection == 'concat':
        Concatenate(axis=-1, name='concat_skip_1')([s2, d0])
    elif skip_connection == 'add':
        Add(name='add_skip_1')([s2, d0])

    # Decoder layer 1,
    #                           ts/4 x ts/4 x 256 > ts/2 x ts/2 x 64 > ts/2 x ts/2 x 128
    d1 = _up_block(
        d0,
        num_filters=num_filters*4, # 2
        activation=activation,
        initializer=initializer,
        conv_1d_later=conv_1d_later,
        shortcut=shortcut,
        dropout_rate=dropout_rate,
        dropout_wrn=dropout_wrn,
        up_sampling=up_sampling,
        batchnorm=batchnorm,
        name='up_block_2'
    )
    
    if skip_connection == 'concat':
        Concatenate(axis=-1, name='concat_skip_2')([s1, d1])
    elif skip_connection == 'add':
        Add(name='add_skip_2')([s1, d1])

    # Decoder layer 2,
    #                           ts/2 x ts/2 x 128 > 2*ts x2*ts x 64 > 2*ts x 2*ts x 64
    d2 = _up_block(
        d1,
        num_filters=num_filters*2, # 1
        activation=activation,
        initializer=initializer,
        conv_1d_later=conv_1d_later,
        shortcut=shortcut,
        dropout_rate=dropout_rate,
        dropout_wrn=dropout_wrn,
        up_sampling=up_sampling,
        batchnorm=batchnorm,
        name='up_block_3'
    )

    if skip_connection == 'concat':
        Concatenate(axis=-1, name='concat_skip_3')([s0, d2])
    elif skip_connection == 'add':
        Add(name='add_skip_3')([s0, d2])

    if stacks:
            # Output layer 2*ts x 2*ts x 64 > ts x ts x 1
        output_data = Conv3D(
            filters=1,
            kernel_size=1,
            padding=padding,
            kernel_initializer=initializer,
            name='output_layer'
        )(d2)
        output_data = _activation(output_data, output_activation)

    if tracks:
            # Output layer 2*ts x 2*ts x 64 > ts x ts x 1
        d2 = tf.keras.layers.Add()(
            [d2[:,:,:,f,:] for f in range(duration)]
        )
        output_data = Conv2D(
            filters=1,
            kernel_size=1,
            padding=padding,
            kernel_initializer=initializer,
            name='output_layer'
        )(d2)
        output_data = _activation(output_data, output_activation)
    return Model(inputs=input_data, outputs=output_data)