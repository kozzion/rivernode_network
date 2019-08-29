import os
import sys



from keras.models import Model
from keras.layers import Dense
from keras.regularizers import l2
from keras.applications.resnet50 import ResNet50

dict_name_activation = {'softmax':True, 'linear':True}

def build(persistancy_model, name_activation, size_encoding, name_weight='imagenet', name_pooling='avg'):
    if name_activation not in dict_name_activation:
        raise RuntimeError('illegal name_activation: ' + name_activation)
    name_model_source = 'resnet50' + name_weight + '_notop_pool' + name_pooling + '_base'

    #TODO check if it exist
    model_resnet50_base = persistancy_model.load_model(name_model_source)
    output = model_resnet50_base.output
    output = Dense(512, activation='elu', kernel_regularizer=l2(0.001))(output)
    output = Dense(size_encoding, activation=name_activation, name='img')(output) #TODO do we really need the name?
    return Model(inputs=model_resnet50_base.input, outputs=output)

def prepare(persistancy_model, name_weight='imagenet', name_top='notop', name_pooling='avg'):
    #TODO do more argument checking
    if name_top == 'notop':
        include_top = False
    else:
        include_top = True
    name_model_target = 'resnet50' + name_weight + '_' + name_top + '_pool' + name_pooling + '_base'
    if persistancy_model.has_model(name_model_target):
        return
    model = ResNet50(weights=name_weight, include_top=include_top, pooling=name_pooling)
    persistancy_model.save_model(name_model_target, model)
