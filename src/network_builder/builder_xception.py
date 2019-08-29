from keras.applications.xception import Xception
from keras.models import Model
import numpy as np

# base_model = VGG19(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

# img_path = 'elephant.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# block4_pool_features = model.predict(x)



dict_name_layer = {'avg_pool'}

def create_model_feature(persistancy_model, name_layer='avg_pool'):
    name_weight='imagenet'
    name_top='top'
    name_pooling='none'
    name_model_source = 'xception' + name_weight + '_' + name_top + '_pool' + name_pooling + '_base'
    name_model_target = 'xception' + name_weight +'_feature' + name_layer
    if persistancy_model.has_model(name_model_target):
        print('model: ' + name_model_target + ' found, skipping')
        return

    model_source = persistancy_model.load_model(name_model_source)
    model_target = Model(inputs=model_source.input, outputs=model_source.get_layer(name_layer).output)
    persistancy_model.save_model(name_model_target, model_target)

def prepare(persistancy_model):
    #TODO get input dims (299x299??), get ouput dims
    name_weight='imagenet'
    name_top='top'
    name_pooling='avg'
    name_model_target = 'xception' + name_weight + '_' + name_top + '_pool' + name_pooling + '_base'
    if persistancy_model.has_model(name_model_target):
        print('model: ' + name_model_target + ' found, skipping')
        return

    model_target = Xception(include_top=True, weights=name_weight, input_tensor=None, input_shape=None, pooling=None, classes=1000)

    #TODO
    persistancy_model.save_model(name_model_target, model_target)

def check(persistancy_model):
    #TODO get input dims (299x299??), get ouput dims
    name_weight='imagenet'
    name_top='top'
    name_pooling='avg'
    name_model = 'xception' + name_weight + '_' + name_top + '_pool' + name_pooling + '_base'
    model = persistancy_model.load_model(name_model)
    # model.summary()
    layer_input = model.get_layer(index=0)
    layer_output  = model.get_layer(index=-1)
    print(layer_input.input_shape)
    print(layer_output.output_shape)


