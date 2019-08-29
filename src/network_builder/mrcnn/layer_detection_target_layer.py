import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

import utils

class DetectionTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, count_image_per_gpu, train_roi_per_image, shape_mask_0, shape_mask_1, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.count_image_per_gpu = count_image_per_gpu
        self.train_roi_per_image = train_roi_per_image
        self.shape_mask_0 = shape_mask_0
        self.shape_mask_1 = shape_mask_1
     

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z, self.config),
            self.count_image_per_gpu, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.train_roi_per_image, 4),  # rois
            (None, self.train_roi_per_image),  # class_ids
            (None, self.train_roi_per_image, 4),  # deltas
            (None, self.train_roi_per_image, self.shape_mask_0, self.shape_mask_1)  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


    def get_config(self):
        config = {}
        config['count_image_per_gpu'] = self.count_image_per_gpu
        config['train_roi_per_image'] = self.train_roi_per_image
        config['rpn_bbox_std_dev'] = self.rpn_bbox_std_dev
        config['shape_mask_0'] = self.shape_mask_0
        config['shape_mask_1'] = self.shape_mask_1
        base_config = super(DetectionTargetLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))