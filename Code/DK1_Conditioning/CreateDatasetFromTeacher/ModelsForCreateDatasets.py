import tensorflow as tf
from Layers import FiLM

"""
Initializes a data generator object
  :param data_dir: the directory in which data are stored
  :param output_size: output size
  :param batch_size: The size of each batch returned by __getitem__
"""


def create_model_LSTM_DK(input_dim=1, conditioning_size=1, b_size=2400):
    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, 1, input_dim), name='input')

    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, conditioning_size), name='cond_inputs')

    outputs = tf.keras.layers.LSTM(8, stateful=True, return_sequences=True, name='LSTM')(inputs)

    outputs = tf.keras.layers.LSTM(32, stateful=True, return_sequences=True, name='LSTM3')(outputs)

    outputs = tf.keras.layers.LSTM(32, stateful=True, return_sequences=True, name='LSTM5')(outputs)

    outputs6 = tf.keras.layers.LSTM(8, stateful=True, return_sequences=False, name='LSTM7')(outputs)

    outputs_film = FiLM(in_size=8)(outputs6, cond_inputs)

    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)

    model = tf.keras.models.Model([cond_inputs, inputs], [outputs, outputs6, outputs_film])

    model.summary()

    return model