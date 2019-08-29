import sys
import os
import base64

sys.path.append(os.path.abspath('../imagenode_core/'))
from route import Route
from manager_model import ManagerModel
import tools_inspect
import struct_annotation
import struct_labelling


from model_resnet import ModelResnet
import loader_config

path_file_0 = "test_0.jpg"
path_file_1 = "test_1.jpg"
path_file_2 = "test_2.jpg"

model = ModelResnet('C:\\models', loader_config.load_config('deep_fashion_category'))

# option a
# manager_model = ManagerModel(model)
# model.prepare()
# annotation_instance = manager_model.process_path_file_no_thread_no_queue(path_file_0)
# tools_inspect.show_annotation_instance(path_file_0, annotation_instance)

# option b
# manager_model = ManagerModel(model)
# model.prepare()
# annotation_instance = manager_model.process_path_file_no_thread(path_file_0)
# tools_inspect.show_annotation_instance(path_file_0, annotation_instance)

# option c
# manager_model = ManagerModel(model)
# manager_model.start()
# annotation_instance = manager_model.process_path_file(path_file_0)
# tools_inspect.show_annotation_instance(path_file_0, annotation_instance)

# option d
# manager_model = ManagerModel(model)
# model.prepare()
# list_path_file = [path_file_0, path_file_1, path_file_2]
# # list_path_file = [path_file_2]
# list_annotation_instance = manager_model.process_list_path_file_no_thread(list_path_file)
# for path_file, annotation_instance in zip(list_path_file, list_annotation_instance):
#     tools_inspect.show_annotation_instance(path_file, annotation_instance)

# option e
with open(path_file_0, "rb") as file:
    image_base64 = base64.b64encode(file.read()).decode('utf-8')
route = Route(model)
succes, annotation_instance = route.image_annotation_for_image_base64(image_base64)
tools_inspect.show_annotation_instance(path_file_0, annotation_instance)
