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

    cond_inputs = tf.keras.layers.Input(batch_shape=(
            b_size, conditioning_size), name='cond_inputs')

    # add layers to the model dynamically with the units from the trial.
    outputs0 = tf.keras.layers.LSTM(
        8, stateful=True, return_sequences=True, name="LSTM0")(inputs)
    outputs1 = tf.keras.layers.LSTM(
        16, stateful=True, return_sequences=True, name="LSTM1")(outputs0)
    outputs2 = tf.keras.layers.LSTM(
        32, stateful=True, return_sequences=True, name="LSTM2")(outputs1)

    outputs3 = tf.keras.layers.LSTM(
        64, stateful=True, return_sequences=True, name="LSTM3")(outputs2)
    outputs4 = tf.keras.layers.LSTM(
        32, stateful=True, return_sequences=True, name="LSTM4")(outputs3)

    outputs4 = FiLM(in_size=32)(outputs4[:, :, 0], cond_inputs)
    outputs4 = tf.expand_dims(outputs4, axis=-1)

    outputs5 = tf.keras.layers.LSTM(
        16, stateful=True, return_sequences=True, name="LSTM5")(outputs4)
    outputs6 = tf.keras.layers.LSTM(
        8, stateful=True, return_sequences=False, name="LastLSTM")(outputs5)

    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs6)

    model = tf.keras.models.Model([inputs, cond_inputs], [
                                      outputs, outputs6])

    model.summary()

    return model