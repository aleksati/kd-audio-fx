import tensorflow as tf

"""
Initializes a data generator object
  :param data_dir: the directory in which data are stored
  :param output_size: output size
  :param batch_size: The size of each batch returned by __getitem__
"""


def create_model_LSTM_DK1(unit, input_dim=1, conditioning_size=0, b_size=2399):

    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, 1, input_dim), name='input')

    outputs = tf.keras.layers.LSTM(
                unit, stateful=True, return_sequences=False, name="LSTM")(inputs)

    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)

    model = tf.keras.models.Model(inputs, outputs)

    model.summary()

    return model
