'''
code to help with building and training NNs with TensorFlow.Keras
'''
### globals
import os, sys
import numpy as np
import tensorflow as tf


def build(num_input_features):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(num_input_features,), name="Input_Layer"),
        tf.keras.layers.Dense(num_input_features//4*2, activation='relu', name="Hidden_Layer_1"),
        tf.keras.layers.Dense(num_input_features//4, activation='relu', name="Hidden_Layer_2"),
        tf.keras.layers.Dense(1, activation='linear', name="Output_Layer") # linear activation for regression
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error', # For regression
                  metrics=['mae', 'mse']) # track 'mean absolute error' and 'mean squared error'
    
    return model


def train(model, X_train, y_train, epochs=10, batch_size=25):
    history = model.fit(X_train,
                        y_train,
                        batch_size,
                        epochs,
                        validation_split=0.2,
                        verbose=2) # 0 for silent, 1 for progress bar, 2 for one line per epoch
    
    return model, history


def save(model, output_dir, model_name):
    # NOTE that model.save will make this model_path into a directory to store a bunch of stuff
    model_path = os.path.join(output_dir, 'model', model_name)
    model.save(model_path, save_format='tf', include_optimizer=False)
    return
