"""
File: train_emotion_classifier.py
Author: HungNT
Base: oarriaga
Description: Train emotion classification model
"""

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import clear_session

from models.auto_cnn import random_model

from utils.datasets import DataManager
from utils.datasets import split_data
from utils.preprocessor import preprocess_input

import hashlib
import requests
import gc

# parameters
batch_size = 256
num_epochs = 300
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 30
base_path = '../trained_models/emotion_models/'
dataset_name = 'fer2013'
url = 'http://tradersupport.club/admin/api/check_exist_and_insert_hash_model'
headers = {'Accept': 'application/json'}

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

# loading dataset
data_loader = DataManager(dataset_name, image_size=input_shape[:2])
faces, emotions = data_loader.get_data()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
train_data, val_data = split_data(faces, emotions, validation_split)
train_faces, train_emotions = train_data
i = 0
while(True):
    # model parameters/compilation
    model = random_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    str_sha256 = hashlib.sha256(model.to_json().encode()).hexdigest()

    data_json = {"sha256": str_sha256}
    response = requests.post(url, json = data_json, headers = headers).json()
    if response['status']:
        continue
    
    # callbacks
    log_file_path = base_path + dataset_name + '_' + str_sha256 + '.csv'
    model_names = base_path + dataset_name + '_' + str_sha256 + '.hdf5'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                    patience=int(patience/4), verbose=1)
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                    save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    model.fit_generator(data_generator.flow(train_faces, train_emotions,
                                            batch_size),
                        steps_per_epoch=len(train_faces) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=val_data)
    del model
    clear_session()
    gc.collect()
