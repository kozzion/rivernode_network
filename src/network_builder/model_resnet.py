import os
import sys

sys.path.append(os.path.abspath('../imagenode_core/'))
import tools_model
import struct_annotation
import struct_labelling

from keras.models import Model
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K

import numpy as np
import cv2
import random


from iterator_image_annotation import ImageAnnotationListIterator
import json

import loader_model


class ModelResnet(object):
    def __init__(self, path_dir_data_cache, config, force_cpu=True):
        super(ModelResnet, self).__init__()
        self.path_dir_data_cache = path_dir_data_cache
        self.config = config

        self.data_format = 'channels_last'
        self.color_mode = 'rgb',
        self.target_size = target_size=(200, 200)
        self.image_shape = self.target_size + (3,)
        self.size_batch = 32

    def prepare(self):
        self.image_data_generator = ImageDataGenerator()
        self.is_running = False

        # load model
        self.labelling = tools_model.load_labelling(self.path_dir_data_cache, self.config['name_labelling'])
        self.model = tools_model.load_model(self.path_dir_data_cache, self.config['name_model']) #TODO move to model loader
        tools_model.load_weight(self.path_dir_data_cache, self.config['name_weight'], self.model)

        # compile model
        self.model.compile(
            optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True),
            loss={'img': 'categorical_crossentropy', 'bbox': 'mean_squared_error'},
            metrics={'img': ['accuracy', 'top_k_categorical_accuracy'], 'bbox': ['mse']})

    def batch_create_path_file(self, path_file):
        img = image.load_img(path_file, grayscale = False, target_size = self.target_size)
        shape = cv2.imread(path_file).shape
        return self.batch_create_instance((img, shape))

    def batch_create_instance(self, instance):
        (img, shape)= instance
        array_image = image.img_to_array(img, data_format = self.data_format)
        array_batch_item = self.image_data_generator.standardize(array_image)

        dict_metadata = {}
        dict_metadata['list_dim_input'] = [shape[1], shape[0]] #opencv has height first (array of arrays notion which we do not care about)
        return array_batch_item, dict_metadata

    def batch_process(self, list_array_batch_item, list_dict_metadata):
        array_batch = np.zeros((len(list_array_batch_item),) + self.image_shape, dtype=K.floatx())
        for index, array_batch_item in enumerate(list_array_batch_item):
            array_batch[index] = array_batch_item

        list_result_batch = self.model.predict_on_batch(array_batch)
        return list_result_batch

    def batch_complete(self, list_result_batch, list_dict_metadata):
        list_image_annotation = self.result_to_list_image_annotation(list_dict_metadata, list_result_batch[0], list_result_batch[1])
        return list_image_annotation


    def result_to_list_image_annotation(self, list_image_shape, list_prediction, list_boundingbox):
        list_image_annotation = []
        for i in range(len(list_image_shape)):
            list_image_annotation.append(self.result_to_image_annotation(list_image_shape[i], list_prediction[i], list_boundingbox[i]))
        return list_image_annotation


    def result_to_image_annotation(self, dict_metadata, array_ecoding, array_boundingbox):
        #regularize the bounding box to make sense
        dim_input_0 = dict_metadata['list_dim_input'][0]
        dim_input_1 = dict_metadata['list_dim_input'][1]
        x2 = min(array_boundingbox[2], 1)
        y2 = min(array_boundingbox[3], 1)
        x1 = min(array_boundingbox[0], array_boundingbox[2])
        y1 = min(array_boundingbox[1], array_boundingbox[3])

        x = int(x1 * dim_input_0)
        y = int(y1 * dim_input_1)
        w = int((x2 -  x1)  * dim_input_0)
        h = int((y2 -  y1)  * dim_input_1)

        annotation_instance = struct_annotation.create_instance(dim_input_0, dim_input_1)
        annotation_instance['list_annotation'].append(struct_labelling.decode(self.labelling, [x, y, w, h], array_ecoding))
        return annotation_instance



###Training from here

    def test(self, list_tuple):
        test_image_data_generator = ImageDataGenerator()
        test_iterator = ImageAnnotationListIterator(list_tuple, self.labelling, test_image_data_generator, target_size=(200, 200))
        scores = self.model.evaluate_generator(test_iterator, steps=2000)

        print('Multi target loss: ' + str(scores[0]))
        print('Image loss: ' + str(scores[1]))
        print('Bounding boxes loss: ' + str(scores[2]))
        print('Image accuracy: ' + str(scores[3]))
        print('Top-5 image accuracy: ' + str(scores[4]))
        print('Bounding boxes error: ' + str(scores[5]))

        # Multi target loss: 7.156928829193115
        # Image loss: 4.873817190408706
        # Bounding boxes loss: 0.44048004437983035
        # Image accuracy: 0.006640625
        # Top-5 image accuracy: 0.058421875
        # Bounding boxes error: 0.44048004437983035

        # 32 epoch
        # Multi target loss: 3.1632959175109865
        # Image loss: 1.7250869936943054
        # Bounding boxes loss: 0.030489844439551234
        # Image accuracy: 0.508828125
        # Top-5 image accuracy: 0.839625
        # Bounding boxes error: 0.030489844439551234

        # Multi target loss: 3.156714767217636
        # Image loss: 1.7183037620782853
        # Bounding boxes loss: 0.03069192047510296
        # Image accuracy: 0.508640625
        # Top-5 image accuracy: 0.84096875
        # Bounding boxes error: 0.03069192047510296


        # 200 epoch
        # Multi target loss: 2.210102091431618
        # Image loss: 1.803655897974968
        # Bounding boxes loss: 0.012509618766140193
        # Image accuracy: 0.512421875
        # Top-5 image accuracy: 0.847265625
        # Bounding boxes error: 0.012509618766140193

def custom_generator(iterator):
    while True:
        batch_x, batch_y = iterator.next()
        yield (batch_x, batch_y)

def train(path_dir_data_cache, name_labelling, name_model_load, name_model_save, name_annotation, count_steps =2000, count_epoch = 2000):

    labelling = tools_model.load_labelling(path_dir_data_cache, name_labelling)
    model = loader_model.load_model(path_dir_data_cache, name_model_load, labelling)

    # generate data
    list_tuple_train, list_tuple_val, list_tuple_test = tools_model.load_list_tuple(path_dir_data_cache, name_annotation)

    train_image_data_generator = ImageDataGenerator(rotation_range=30., shear_range=0.2, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    train_iterator = ImageAnnotationListIterator(list_tuple_train, labelling, train_image_data_generator, target_size=(200, 200))

    val_image_data_generator = ImageDataGenerator()
    val_iterator = ImageAnnotationListIterator(list_tuple_val, labelling, val_image_data_generator, target_size=(200, 200))

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', patience=12, factor=0.5, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join('.','log'))
    early_stopper = EarlyStopping(monitor='val_loss', patience=30, verbose=1)
    checkpoint = ModelCheckpoint(os.path.join('.','checkpoint.h5'))

    model.fit_generator(custom_generator(train_iterator),
                      steps_per_epoch=count_steps,
                      epochs=count_epoch,
                      validation_data=custom_generator(val_iterator),
                      validation_steps=200,
                      verbose=2,
                      shuffle=True,
                      callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard],
                      workers=1)
    # !!!workers was 12

    tools_model.save_model(path_dir_data_cache, name_model_save, model)
    # self.test(list_tuple_test)
