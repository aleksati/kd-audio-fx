import tensorflow as tf
from Layers import FiLM

"""
Initializes a data generator object
  :param data_dir: the directory in which data are stored
  :param output_size: output size
  :param batch_size: The size of each batch returned by __getitem__
"""


def create_model_LSTM_DK(units=[8, 16, 32, 64, 32, 16, 8], input_dim=1, conditioning_size=0, b_size=2399):

    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, 1, input_dim), name='input')

    # add layers to the model dynamically with the units from the trial.
    outputs0 = tf.keras.layers.LSTM(
        units[0], stateful=True, return_sequences=True, name="LSTM0")(inputs)
    outputs1 = tf.keras.layers.LSTM(
        units[1], stateful=True, return_sequences=True, name="LSTM1")(outputs0)
    outputs2 = tf.keras.layers.LSTM(
        units[2], stateful=True, return_sequences=True, name="LSTM2")(outputs1)
    outputs3 = tf.keras.layers.LSTM(
        units[3], stateful=True, return_sequences=True, name="LSTM3")(outputs2)
    outputs4 = tf.keras.layers.LSTM(
        units[4], stateful=True, return_sequences=True, name="LSTM4")(outputs3)
    outputs5 = tf.keras.layers.LSTM(
        units[5], stateful=True, return_sequences=True, name="LSTM5")(outputs4)
    outputs6 = tf.keras.layers.LSTM(
        units[6], stateful=True, return_sequences=False, name="LastLSTM")(outputs5)

    cond_inputs = tf.keras.layers.Input(batch_shape=(
            b_size, conditioning_size), name='cond_inputs')

    outputs = FiLM(in_size=units)(outputs6, cond_inputs)

    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs6)

    model = tf.keras.models.Model([inputs, cond_inputs], [
                                      outputs, outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6])

    model.summary()

    return model
