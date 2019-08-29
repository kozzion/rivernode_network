import sys
import os
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
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K

import random
import numpy as np


class ImageAnnotationListIterator(object):
    def __init__(self, list_tuple, labelling, image_data_generator,
    color_mode: str = 'rgb', data_format='channels_last', target_size=(256, 256), batch_size: int = 32, shuffle: bool = True, seed=None):
        super(ImageAnnotationListIterator, self).__init__()
        self.list_tuple = list_tuple
        self.labelling = labelling

        # also needed not default
        self.image_data_generator = image_data_generator

        # also needed default
        self.color_mode = color_mode
        self.data_format = data_format
        self.target_size = target_size
        self.batch_size = batch_size
        if len(list_tuple) < batch_size:
            self.batch_size = len(list_tuple)
        self.shuffle = shuffle
        self.seed = seed

        # self.image_shape
        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

    def next(self):
        while True:
            #TODO add lock aroung random and use seed

            index_array = random.sample(range(0, len(self.list_tuple)), self.batch_size)

            # TODO make bactching paralel
            batch_x = np.zeros((self.batch_size,) + self.image_shape, dtype=K.floatx())
            batch_y = np.zeros((self.batch_size, self.labelling['size_encoding']), dtype=K.floatx())
            bounding_box = np.zeros((self.batch_size,) + (4,), dtype=K.floatx())



            # build batch of image data
            for i, j in enumerate(index_array):
                file_path, image_annotation  = self.list_tuple[j]
                annotation = image_annotation['list_annotation'][0]
                img = image.load_img(file_path,
                                     grayscale = self.color_mode == 'grayscale',
                                     target_size = self.target_size)

                x = image.img_to_array(img, data_format=self.data_format)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i] = x
                batch_y[i,:] = struct_labelling.encode(self.labelling, annotation)

                x1 = annotation['xywh'][0] / image_annotation['width']
                y1 = annotation['xywh'][1] / image_annotation['height']
                x2 = (annotation['xywh'][0] + annotation['xywh'][2]) / image_annotation['width']
                y2 = (annotation['xywh'][1] + annotation['xywh'][3]) / image_annotation['height'] #TODO rounding to 2 decimals?
                bounding_box[i] = np.asarray([x1, y1, x2, y2], dtype=K.floatx())

            return batch_x, [batch_y, bounding_box]
