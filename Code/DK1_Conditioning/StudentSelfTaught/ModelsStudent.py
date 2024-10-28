import tensorflow as tf
from Layers import FiLM

"""
Initializes a data generator object
  :param data_dir: the directory in which data are stored
  :param output_size: output size
  :param batch_size: The size of each batch returned by __getitem__
"""


def create_model_LSTM_DK1(units, input_dim=1, conditioning_size=0, b_size=2400):
    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, 1, input_dim), name='input')

    outputs = tf.keras.layers.LSTM(
        units, stateful=True, return_sequences=True, name="LSTM")(inputs)

    cond_inputs = tf.keras.layers.Input(batch_shape=(
        b_size, conditioning_size), name='cond_inputs')

    outputs = FiLM(in_size=units)(outputs[:, :, 0], cond_inputs)
    outputs = tf.expand_dims(outputs, axis=1)

    outputs = tf.keras.layers.LSTM(
        8, stateful=True, return_sequences=False, name="LSTM2")(outputs)

    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)

    model = tf.keras.models.Model([cond_inputs, inputs], outputs)

    model.summary()

    return model
