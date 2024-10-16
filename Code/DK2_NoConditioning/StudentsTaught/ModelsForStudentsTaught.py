import tensorflow as tf

"""
Initializes a data generator object
  :param data_dir: the directory in which data are stored
  :param output_size: output size
  :param batch_size: The size of each batch returned by __getitem__
"""


def create_model_LSTM_DK2(units, input_dim=1, b_size=2400, training=True):

    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, 1, input_dim), name='input')

    outputs = tf.keras.layers.LSTM(
                units, stateful=True, return_sequences=True, name="LSTM")(inputs)
    outputs = tf.keras.layers.LSTM(
                8, stateful=True, return_sequences=False, name="LSTM2")(outputs)

    if training == False:

        outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)
        model = tf.keras.models.Model(inputs, outputs)
    else:
        model = tf.keras.models.Model(inputs, outputs)

    model.summary()

    return model