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
        batch_shape=(b_size, input_dim), name='input')
    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, conditioning_size), name='cond_inputs')
    inputs = tf.expand_dims(inputs, axis=1)


    outputs = tf.keras.layers.LSTM(16, stateful=True, return_sequences=True, name='LSTM1')(
        inputs)

    outputs = FiLM(in_size=16)(outputs[:, :, 0], cond_inputs)
    outputs = tf.expand_dims(outputs, axis=1)

    outputs = tf.keras.layers.LSTM(32, stateful=True, return_sequences=True, name='LSTM2')(
        outputs)

    outputs = FiLM(in_size=32)(outputs[:, :, 0], cond_inputs)
    outputs = tf.expand_dims(outputs, axis=1)

    outputs = tf.keras.layers.LSTM(32, stateful=True, return_sequences=True, name='LSTM4')(
        outputs)

    outputs = FiLM(in_size=32)(outputs[:, :, 0], cond_inputs)
    outputs = tf.expand_dims(outputs, axis=1)

    outputs = tf.keras.layers.LSTM(16, stateful=True, return_sequences=False, name='LSTM5')(
        outputs)

    outputs = FiLM(in_size=16)(outputs, cond_inputs)

    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)
    model = tf.keras.models.Model([cond_inputs, inputs], outputs)


    model.summary()

    return model



def __create_model_LSTM_DK1(input_dim=1, conditioning_size=0, b_size=2399):

    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, input_dim), name='input')
    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, conditioning_size), name='cond_inputs')

    inputs =  tf.expand_dims(inputs, axis=1)

    outputs = tf.keras.layers.LSTM(8, stateful=True, return_sequences=True, name='LSTM')(
        inputs)
    outputs = FiLM(in_size=8)(outputs[:, :, 0], cond_inputs)
    outputs = tf.expand_dims(outputs, axis=1)

    outputs = tf.keras.layers.LSTM(16, stateful=True, return_sequences=True, name='LSTM1')(
        outputs)

    outputs = FiLM(in_size=16)(outputs[:, :, 0], cond_inputs)
    outputs = tf.expand_dims(outputs, axis=1)


    outputs = tf.keras.layers.LSTM(32, stateful=True, return_sequences=True, name='LSTM2')(
        outputs)

    outputs = FiLM(in_size=32)(outputs[:, :, 0], cond_inputs)
    outputs = tf.expand_dims(outputs, axis=1)

    outputs = tf.keras.layers.LSTM(64, stateful=True, return_sequences=True, name='LSTM3')(
        outputs)

    outputs = FiLM(in_size=64)(outputs[:, :, 0], cond_inputs)
    outputs = tf.expand_dims(outputs, axis=1)

    outputs = tf.keras.layers.LSTM(32, stateful=True, return_sequences=True, name='LSTM4')(
        outputs)

    outputs = FiLM(in_size=32)(outputs[:, :, 0], cond_inputs)
    outputs = tf.expand_dims(outputs, axis=1)

    outputs = tf.keras.layers.LSTM(16, stateful=True, return_sequences=True, name='LSTM5')(
        outputs)

    outputs = FiLM(in_size=16)(outputs[:, :, 0], cond_inputs)
    outputs = tf.expand_dims(outputs, axis=1)

    outputs = tf.keras.layers.LSTM(8, stateful=True, return_sequences=False, name='LastLSTM')(
        outputs)

    outputs = FiLM(in_size=8)(outputs, cond_inputs)

    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)
    model = tf.keras.models.Model([cond_inputs, inputs], outputs)


    model.summary()

    return model