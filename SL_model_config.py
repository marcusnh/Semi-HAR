
'''

INFO ABOUT model_config.py

Model to create_model and configurate it after need.
'''
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Embedding, LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import tensorflow as tf

def create_DNN_model(hp,input_shape=(50, 4), output_shape=3, 
                    model_name='test_model'):
    '''
    Create_model for HAR recognition:
    Simple DNN model 

    '''
    # hyperparameters:
    hp_neurons = hp.Int('units_hp', min_value = 32, max_value = 512, step = 32)
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])

    # inputs = tf.keras.Input(shape=input_shape, name='input')
    # x = inputs
    # x = Dense(units=hp_neurons, activation='relu', )(x)
    # x = Dropout(0.5)(x)
    # x = Dense(units=hp_neurons, activation='relu', )(x)
    # x = Flatten()(x)
    # output = Dense(3, activation='softmax')(x)
    # model =tf.keras.Model(inputs=inputs, outputs=output, name=model_name)

    model = Sequential()
    
    model.add(Dense(units=hp_neurons, activation='relu', ))
    model.add(Dropout(0.5))
    model.add(Dense(units=hp_neurons, activation='relu'))

    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    if hp.Choice('optimizer', ['adam', 'sgd']) == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
        print('USING ADAM')
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=hp_learning_rate)
        print('USING SGD')

    
    model.compile(loss='categorical_crossentropy', optimizer=opt, 
                 metrics = ['accuracy'])
    return model

    


def create_LSTM_model(N_NODES, N_CLASSES, TIME_PERIODS, N_FEATURES):
    model = Sequential()
    # RNN layer
    # Bias regularizer value - we will use elasticnet
    model.add(LSTM(units =60, return_sequences=True,
                   input_shape =(TIME_PERIODS,N_FEATURES)))
    model.add(LSTM(units = 60, return_sequences=True,
                     input_shape =(TIME_PERIODS,N_FEATURES)))
    # Dropout layer
    model.add(Dropout(0.5)) 
    # Dense layer with ReLu
    model.add(Dense(units = 64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(N_CLASSES, activation='softmax'))

def load_model(hp=None, path='test_runs/models/test.h5'):
    model = keras.models.load_model(path)

    return model