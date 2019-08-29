import os
import sys
import json
import numpy as np
import urllib
import shutil


sys.path.append(os.path.abspath('../../../../rivernode_core/src/core_core/'))
from persistency_model_s3 import PersistencyModelS3
import struct_format_input
import struct_format_output

from config_maskrcnn import ConfigMaskrcnn
from model import MaskRCNN

persistency_model = PersistencyModelS3()

# Local path to trained weights file
name_model = 'maskrcnnresnet101_coco_full_base'
path_file_weight = 'maskrcnnresnet101_coco_full_base_weight.h5'

# Download COCO trained weights from Releases if needed # Only for top layers!!!
if not os.path.isfile(path_file_weight):
    url_weight_coco = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"
    with urllib.request.urlopen(url_weight_coco) as response, open(path_file_weight, 'wb') as file:
        shutil.copyfileobj(response, file)

#TODO
# from keras.utils.data_utils import get_file
# TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
#                             'releases/download/v0.2/'\
#                             'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
# weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
#                         TF_WEIGHTS_PATH_NO_TOP,
#                         cache_subdir='models',
#                         md5_hash='a268eb855778b3df3c7506639542a6af')
# return weights_path 


# Directory of images to run detection on
mode = 'inference'
count_class = 1 + 80
count_gpu = 1
count_image_per_gpu = 1
json_config_coco = ConfigMaskrcnn.create_json(count_class, count_gpu, count_image_per_gpu)
json_config_coco['BACKBONE'] = "resnet101"   #TODO redundant, Supported values are: resnet50, resnet101. We could also insert other backbones?

# Create model object in inference mode.
config_coco = ConfigMaskrcnn(json_config_coco)
manager_model = MaskRCNN(mode, config_coco)

# Load weights trained on MS-COCO
manager_model.load_weights(path_file_weight, by_name=True)

input_shape = manager_model.keras_model.layers[0].input_shape
format_input = struct_format_input.create([input_shape[1], input_shape[2]])
output_shape = manager_model.keras_model.layers[-1].output_shape
print(manager_model.keras_model.layers[-1].output_shape)
format_output = struct_format_output.create('maskrcnn', output_shape[1])

# COCO Class names #TODO create labelling from this
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')

list_name_class = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

json_metadata = {}
json_metadata['labelling'] = list_name_class
json_metadata['config'] = json_config_coco #TODO this is the dummy field
json_metadata['name_model'] = name_model
json_metadata['type_model'] = 'maskrcnn'
json_metadata['format_input'] = format_input
json_metadata['format_output'] = format_output
persistency_model.save_model(name_model, manager_model.keras_model, json_metadata, False)

   


