import tensorflow as tf

"""
Initializes a data generator object
  :param data_dir: the directory in which data are stored
  :param output_size: output size
  :param batch_size: The size of each batch returned by __getitem__
"""


def create_model_LSTM_DK1(trial, input_dim=1, conditioning_size=0, b_size=2399):

    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, 1, input_dim), name='input')

    # add layers to the model dynamically with the units from the trial.
    for i, unit in enumerate(trial):
        # for the first loop, attach the layer to the input layer, otherwise, attach to previous layer.
        if i == 0:
            tail = inputs
        else:
            tail = outputs

        # add new layer
        outputs = tf.keras.layers.LSTM(
            unit, stateful=True, return_sequences=True, return_state=False, name="LSTM{}".format(i))(tail)

    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)

    model = tf.keras.models.Model(inputs, outputs)

    model.summary()

    return model
