import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import clear_session

import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random
from IPython.display import Image

from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

import keras.backend as K
from keras.models import Sequential

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Dense, LSTM
from keras.layers import AveragePooling2D, BatchNormalization
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



def build_model(input_shape, num_classes, l2_regularization=0.01):
    activations = ['elu', 'selu', 'relu', 'tanh', 'sigmoid']
    rate_dropout = [0.4, 0.5, 0.6, 0.7, 0.8]
    len_rate = len(rate_dropout)
    len_activations = len(activations)
    use_bias = [False, True]

    regularization = l2(l2_regularization)
    # base
    img_input = Input(input_shape)
    x = Conv2D(256, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation(activations[randrange(0, len_activations,1)])(x)

    # module 1
    filters = 2048
    kernel_size = randrange(3, 8, 2)
    kernel_size = (kernel_size, kernel_size)
    stride = randrange(1, 3, 1)
    strides_random = (stride, stride)
    use_bias_random = use_bias[randrange(0, 2, 1)]

    residual = Conv2D(filters, kernel_size, strides=strides_random,
                    padding='same', use_bias=use_bias_random)(x)

    residual = BatchNormalization()(residual)

    x = SeparableConv2D(filters, kernel_size, 
                        padding='same',
                        kernel_regularizer=regularization,
                        use_bias=use_bias_random)(x)

    x = BatchNormalization()(x)

    x = Activation(activations[randrange(0, len_activations,1)])(x)

    x = MaxPooling2D(kernel_size, strides=strides_random, 
                        padding='same')(x)
    x = layers.add([x, residual])

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
    
model = build_model((100, 100, 3), 5005)
model.summary()
os.listdir("../input")
# load image data
df = pd.read_csv('../input/train.csv')
Image(filename="../input/train/"+random.choice(df['Image'])) 
training_pts_per_class = df.groupby('Id').size()
training_pts_per_class
print("Min example a class can have: ",training_pts_per_class.min())
print("0.99 quantile: ",training_pts_per_class.quantile(0.99))
print("Max example a class can have: \n",training_pts_per_class.nlargest(5))    
data = training_pts_per_class.copy()
data.loc[data > data.quantile(0.99)] = '22+'

def prepare_images(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    
    for fig in data['Image']:
        #load images into images of size 100x100x3
        img = image.load_img("../input/"+dataset+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train

def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder

X = prepare_images(df, df.shape[0], "train")
X /= 255

y, label_encoder = prepare_labels(df['Id'])

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# config 
batch_size = 32
num_epochs = 100
input_shape = (100, 100, 3)
num_classes = 5005
patience = 50
# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)
# callbacks
log_file_path = 'log_training.csv'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4), verbose=1)
model_names = 'whale.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]
# loading dataset
model.fit_generator(data_generator.flow(X, y,
                                        batch_size),
                    steps_per_epoch=len(X) / batch_size,
                    epochs=num_epochs, verbose=2, callbacks=callbacks)
gc.collect()
test = os.listdir("../input/test/")
col = ['Image']
test_df = pd.DataFrame(test, columns=col)
test_df['Id'] = ''
X_test = prepare_images(test_df, test_df.shape[0], "test")
X_test /= 255
predictions = model.predict(np.array(X_test), verbose=1)
for i, pred in enumerate(predictions):
    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
test_df.to_csv('submission.csv', index=False)