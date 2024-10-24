import tensorflow as tf
from Layers import FiLM
"""
Initializes a data generator object
  :param data_dir: the directory in which data are stored
  :param output_size: output size
  :param batch_size: The size of each batch returned by __getitem__
"""


def create_model_LSTM_DK1(input_dim=1, conditioning_size=0, b_size=2399):

    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, 1, input_dim), name='input')
    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, conditioning_size), name='cond_inputs')


    outputs = tf.keras.layers.LSTM(8, stateful=True, return_sequences=True, return_state=False, name='LSTM')(
        inputs)

    outputs = tf.keras.layers.LSTM(16, stateful=True, return_sequences=True, return_state=False, name='LSTM1')(
        outputs)

    outputs = tf.keras.layers.LSTM(32, stateful=True, return_sequences=True, return_state=False, name='LSTM2')(
        outputs)

    outputs = tf.keras.layers.LSTM(64, stateful=True, return_sequences=True, return_state=False, name='LSTM3')(
        outputs)

    outputs = tf.keras.layers.LSTM(32, stateful=True, return_sequences=True, return_state=False, name='LSTM4')(
        outputs)

    outputs = FiLM(in_size=32)(outputs[:, :, 0], cond_inputs)
    outputs = tf.expand_dims(outputs, axis=-1)

    outputs = tf.keras.layers.LSTM(16, stateful=True, return_sequences=True, return_state=False, name='LSTM5')(
        outputs)

    outputs = tf.keras.layers.LSTM(8, stateful=True, return_sequences=False, return_state=False, name='LastLSTM')(
        outputs)

    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)
    model = tf.keras.models.Model([inputs, cond_inputs], outputs)


    model.summary()

    return model