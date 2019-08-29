from keras.layers import Input, Dense
from keras.models import Model, load_model


def build_deep(input_dim, size_encoding, optimizer):
    input_layer = Input(shape=(input_dim,))        
    hidden_one = Dense(size_encoding*4, activation='relu') (input_layer)
    hidden_two = Dense(size_encoding*2, activation='relu') (hidden_one)
    encoder_output = Dense(size_encoding, activation='relu') (hidden_two)
    
    hidden_three = Dense(size_encoding*2, activation='relu') (encoder_output)
    hidden_four = Dense(size_encoding*4, activation='relu') (hidden_three)
    decoder_output = Dense(input_dim, activation='sigmoid') (hidden_four)

    model_encoder = Model(input_layer, encoder_output)
    model_decoder = Model(input_layer, decoder_output)
    model_decoder.compile(optimizer=optimizer, loss='mean_squared_error') 
    return model_encoder, model_decoder



def build_deep2(input_dim, size_encoding, optimizer):
    input_layer = Input(shape=(input_dim,))        
    hidden_one = Dense(size_encoding*4, activation='relu') (input_layer)
    hidden_two = Dense(size_encoding*2, activation='relu') (hidden_one)
    encoder_output = Dense(size_encoding, activation='relu') (hidden_two)
    
    hidden_three = Dense(size_encoding*2, activation='relu') (encoder_output)
    hidden_four = Dense(size_encoding*4, activation='relu') (hidden_three)
    decoder_output = Dense(input_dim, activation='sigmoid') (hidden_four)

    model_encoder = Model(input_layer, encoder_output)
    model_decoder = Model(input_layer, decoder_output)
    model_decoder.compile(optimizer=optimizer, loss='binary_crossentropy') #TODO other loss?
    return model_encoder, model_decoder