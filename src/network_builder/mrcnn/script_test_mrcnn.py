import os
import sys



import tools_inspect_maskrcnn

# Local path to trained weights file

sys.path.append(os.path.abspath('../../../../rivernode_core/src/core_core/'))
from persistency_model_s3 import PersistencyModelS3
from coder_maskrcnn import CoderMaskrcnn
import tools_general




path_file_instance = 'test_2.jpg'
name_model = 'maskrcnnresnet101_coco_full_base'

persistency_model = PersistencyModelS3()

# TODO move this to metadata
import tensorflow as tf
# TODO sys.path.append(os.path.abspath('../../../../rivernode_core/src/core_core/'))
from layer_batch_norm import BatchNorm
from layer_detection_layer import DetectionLayer
from layer_detection_target_layer import DetectionTargetLayer
from layer_proposal_layer import ProposalLayer
from layer_pyramid_roi_align import PyramidROIAlign
custom_object_scope ={}
custom_object_scope['tf'] = tf
custom_object_scope['BatchNorm'] = BatchNorm
custom_object_scope['ProposalLayer'] = ProposalLayer
custom_object_scope['DetectionLayer'] = DetectionLayer
custom_object_scope['DetectionTargetLayer'] = DetectionTargetLayer
custom_object_scope['ProposalLayer'] = ProposalLayer
custom_object_scope['PyramidROIAlign'] = PyramidROIAlign

model, json_metadata = persistency_model.load_model(name_model, custom_object_scope=custom_object_scope)
coder = CoderMaskrcnn(json_metadata)
array_image = tools_general.path_file_to_array_image_pil(path_file_instance)


input_predict, metadata_predict = coder.list_array_instance_to_input_predict([array_image])
output_predict = model.predict(input_predict, verbose=0)
list_result = coder.output_predict_to_list_result(output_predict, metadata_predict)

# Visualize results
result = list_result[0]
# tools_inspect_maskrcnn.show(array_image, [], [], [], json_metadata['labelling'], [])
list_volume_mask = []
tools_inspect_maskrcnn.show(array_image, result['rois'], result['masks'], result['class_ids'], json_metadata['labelling'], result['scores'])