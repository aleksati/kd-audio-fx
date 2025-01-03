import tensorflow as tf

"""
Initializes a data generator object
  :param data_dir: the directory in which data are stored
  :param output_size: output size
  :param batch_size: The size of each batch returned by __getitem__
"""


def create_model_LSTM_DK(units=512, mini_batch_size=2048, input_dim=1, b_size=2399, stateful=False):

    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, mini_batch_size, input_dim), name='input')

    outputs = tf.keras.layers.LSTM(units, stateful=stateful, return_sequences=True, name='LSTM')(
        inputs)
    outputs = tf.keras.layers.LSTM(8, stateful=stateful, return_sequences=True, name='LSTM')(
        outputs)

    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)
    model = tf.keras.models.Model(inputs, outputs)


    model.summary()

    return model

def create_model_LSTM_DK_morelay(units=[8, 16, 32, 64, 32, 16, 8], mini_batch_size=2048, input_dim=1, b_size=2399, stateful=False):

    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, mini_batch_size, input_dim), name='input')

    outputs = tf.keras.layers.LSTM(units[0], stateful=stateful, return_sequences=True, name="LSTM0")(inputs)
    outputs = tf.keras.layers.LSTM(units[1], stateful=stateful, return_sequences=True, name="LSTM1")(outputs)
    outputs = tf.keras.layers.LSTM(units[2], stateful=stateful, return_sequences=True, name="LSTM2")(outputs)
    outputs = tf.keras.layers.LSTM(units[3], stateful=stateful, return_sequences=True, name="LSTM3")(outputs)
    outputs = tf.keras.layers.LSTM(units[4], stateful=stateful, return_sequences=True, name="LSTM4")(outputs)
    outputs = tf.keras.layers.LSTM(units[5], stateful=stateful, return_sequences=True, name="LSTM5")(outputs)
    outputs = tf.keras.layers.LSTM(units[6], stateful=stateful, return_sequences=True, name="LastLSTM")(outputs)

    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)

    model = tf.keras.models.Model(inputs, outputs)

    model.summary()

    return model
