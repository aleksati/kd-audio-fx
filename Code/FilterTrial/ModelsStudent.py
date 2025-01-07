import tensorflow as tf

"""
Initializes a data generator object
  :param data_dir: the directory in which data are stored
  :param output_size: output size
  :param batch_size: The size of each batch returned by __getitem__
"""


class DelayAndFeedbackNN(tf.keras.layers.Layer):
    def __init__(self, num_of_steps, units, batch_size, stateful=False):
        super(DelayAndFeedbackNN, self).__init__()
        self.num_of_steps = num_of_steps
        self.batch_size = batch_size
        self.lstm = tf.keras.layers.LSTM(units, stateful=stateful, return_sequences=True, return_state=False,
                                               name='LSTM')

        self.dense = tf.keras.layers.Dense(1, name='OutLayer')
        self.gain_feedback = tf.Variable(tf.ones(1), trainable=True, name='GainFeed')

        #self.residuals = (tf.Variable(tf.zeros([batch_size, num_of_steps, 1]), trainable=False, name='res'))

        #inputs = (tf.Variable(tf.ones([batch_size, num_of_steps, 1], dtype=tf.float32), trainable=False, name='inp'))
        #inputs = tf.expand_dims(inputs, axis=-1)

    def call(self, inputs):

        out = self.lstm(inputs)
        out = self.dense(out)

        #residuals = []
        #rolls = tf.concat((out, self.residuals), axis=1)
        rolls = tf.pad(out, [[0, 0], [self.num_of_steps, 0], [0, 0]])
        for o in range(self.num_of_steps):
            rolls = tf.roll(rolls, shift=1, axis=1)
            rolls = tf.multiply(rolls, self.gain_feedback)
            out = tf.add(rolls[:, self.num_of_steps:, :], out)
            #residuals.append(outs)

        #residuals = tf.stack(residuals)
        #residuals = tf.reduce_sum(residuals, axis=0)
        #residuals = tf.expand_dims(residuals, axis=-1)
        #y = out + residuals[:self.batch_size]

        #self.residuals.assign(y[:self.num_of_steps])

        return out

def create_model_REV(input_dim, mini_batch_size, units=32, b_size=2399, stateful=False):

    # Defining inputs
    inputs = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, input_dim), name='input')

    outputs = DelayAndFeedbackNN(mini_batch_size, units, b_size, stateful)(inputs)

    model = tf.keras.models.Model(inputs, outputs)

    model.summary()
    return model

def create_model_LSTM_DK1(units, mini_batch_size=2048, input_dim=1, b_size=2400, stateful=False):
    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, mini_batch_size, input_dim), name='input')

    outputs = tf.keras.layers.LSTM(
        units, stateful=stateful, return_sequences=True, name="LSTM")(inputs)
    outputs = tf.keras.layers.LSTM(
        8, stateful=stateful, return_sequences=True, name="LSTM2")(outputs)
    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)
    outputs = outputs + inputs

    model = tf.keras.models.Model(inputs, outputs)

    model.summary()

    return model
