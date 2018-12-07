from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Dense
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
from random import randrange

def random_model(input_shape, num_classes, l2_regularization=0.01):
    activations = ['elu', 'selu', 'relu', 'tanh', 'sigmoid']
    rate_dropout = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    len_rate = len(rate_dropout)
    len_activations = len(activations)
    use_bias = [False, True]

    regularization = l2(l2_regularization)
    num_layers = randrange(3, 9, 1)
    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation(activations[randrange(0, len_activations,1)])(x)

    # module
    for i in range(num_layers):
        filters = randrange(2, 8, 1) * 8
        kernel_size = randrange(3, 8, 2)
        kernel_size = (kernel_size, kernel_size)
        stride = randrange(1, 3, 1)
        strides_random = (stride, stride)
        use_bias_random = use_bias[randrange(0, 2, 1)]

        residual = Conv2D(filters, kernel_size, strides=strides_random,
                        padding='same', use_bias=use_bias_random)(x)

        if use_bias[randrange(0, 2, 1)]:
            residual = BatchNormalization()(residual)

        x = SeparableConv2D(filters, kernel_size, 
                            padding='same',
                            kernel_regularizer=regularization,
                            use_bias=use_bias_random)(x)

        if use_bias[randrange(0, 2, 1)]:
            x = BatchNormalization()(x)

        x = Activation(activations[randrange(0, len_activations,1)])(x)

        x = SeparableConv2D(filters, kernel_size, 
                            padding='same',
                            kernel_regularizer=regularization,
                            use_bias=use_bias_random)(x)

        if use_bias[randrange(0, 2, 1)]:
            x = BatchNormalization()(x)

        x = MaxPooling2D(kernel_size, strides=strides_random, 
                            padding='same')(x)
        x = layers.add([x, residual])

        if use_bias[randrange(0, 2, 1)]:
            x = Dropout(rate_dropout[randrange(0, len_rate, 1)])(x)

    # Output
    x = Conv2D(num_classes, (3, 3),
            activation = activations[randrange(0, len_activations, 1)], 
            padding='same')(x)
    if use_bias[randrange(0, 2, 1)]:
        x = GlobalAveragePooling2D()(x)
    else:
        x = GlobalMaxPooling2D()(x)

    output = Activation('softmax', name='predictions')(x)
    model = Model(img_input, output)
    return model

if __name__ == "__main__":
    input_shape = (64, 64, 1)
    num_classes = 7
    model = random_model(input_shape, num_classes)
    model.summary()
