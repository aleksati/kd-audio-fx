import tensorflow as tf

"""
Initializes a data generator object
  :param data_dir: the directory in which data are stored
  :param output_size: output size
  :param batch_size: The size of each batch returned by __getitem__
"""


def create_model_LSTM_DK1(units=512, mini_batch_size=2048, input_dim=1, b_size=2399):

    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, mini_batch_size, input_dim), name='input')

    outputs = tf.keras.layers.LSTM(units, stateful=False, return_sequences=True, name='LSTM')(
        inputs)

    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)
    model = tf.keras.models.Model(inputs, outputs)

    model.summary()

    return model

def create_model_LSTM_DK(units=[8, 16, 32, 64, 32, 16, 8], mini_batch_size=2048, input_dim=1, b_size=2399):

    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, mini_batch_size, input_dim), name='input')

    outputs0 = tf.keras.layers.LSTM(units[0], return_sequences=True, name="LSTM0")(inputs)
    outputs1 = tf.keras.layers.LSTM(units[1], return_sequences=True, name="LSTM1")(outputs0)
    outputs2 = tf.keras.layers.LSTM(units[2], return_sequences=True, name="LSTM2")(outputs1)
    outputs3 = tf.keras.layers.LSTM(units[3], return_sequences=True, name="LSTM3")(outputs2)
    outputs4 = tf.keras.layers.LSTM(units[4], return_sequences=True, name="LSTM4")(outputs3)
    outputs5 = tf.keras.layers.LSTM(units[5], return_sequences=True, name="LSTM5")(outputs4)
    outputs6 = tf.keras.layers.LSTM(units[6], return_sequences=True, name="LastLSTM")(outputs5)

    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs6)

    model = tf.keras.models.Model(inputs, [outputs, outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6])

    model.summary()

    return model
