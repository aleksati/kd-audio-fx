import tensorflow as tf
from Layers import FiLM

"""
Initializes a data generator object
  :param data_dir: the directory in which data are stored
  :param output_size: output size
  :param batch_size: The size of each batch returned by __getitem__
"""


def create_LSTM_DK_model(units, mini_batch_size=2048, input_dim=1, b_size=8, stateful=False):
  #unconditioned 
    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, mini_batch_size, input_dim), name='input')

    outputs = tf.keras.layers.LSTM(
                units, stateful=stateful, return_sequences=True, name="LSTM")(inputs)
    outputs = tf.keras.layers.LSTM(
                8, stateful=stateful, return_sequences=True, name="LSTM2")(outputs)
    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)
    outputs = outputs + inputs
    
    model = tf.keras.models.Model(inputs, outputs)

    model.summary()

    return model


def create_cond_LSTM_DK_model(units, mini_batch_size=2048, input_dim=1, b_size=8, stateful=False):
  # conditioned 
    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, mini_batch_size, input_dim), name='input')
    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 1), name='cond_inputs')

    outputs = tf.keras.layers.LSTM(
                units, stateful=stateful, return_sequences=True, name="LSTM")(inputs)

    outputs = FiLM(in_size=units)(outputs, cond_inputs)


    outputs = tf.keras.layers.LSTM(
                8, stateful=stateful, return_sequences=True, name="LSTM2")(outputs)
    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)
    outputs = outputs + inputs
    
    model = tf.keras.models.Model([cond_inputs, inputs], outputs)

    model.summary()

    return model