import tensorflow as tf


def create_model_LSTM_DK2(units, input_dim=1, conditioning_size=0, enable_second_output=False, training_just_out=False, b_size=2399):

    training_lstms = not training_just_out


    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, 1, input_dim), name='input')

    if units == 64:
        outputs = tf.keras.layers.LSTM(4*2, stateful=True, return_sequences=True, return_state=False, name='LSTM')(
            inputs)

        outputs = tf.keras.layers.LSTM(8*2, stateful=True, return_sequences=True, return_state=False, name='LSTM1')(
            outputs)

        outputs = tf.keras.layers.LSTM(16*2, stateful=True, return_sequences=True, return_state=False, name='LSTM2')(
            outputs)

        outputs = tf.keras.layers.LSTM(32*2, stateful=True, return_sequences=True, return_state=False, name='LSTM3')(
            outputs)

        outputs = tf.keras.layers.LSTM(16*2, stateful=True, return_sequences=True, return_state=False, name='LSTM4')(
            outputs)

        outputs = tf.keras.layers.LSTM(8*2, stateful=True, return_sequences=True, return_state=False, name='LSTM5')(
            outputs)

        outputs = tf.keras.layers.LSTM(4*2, stateful=True, return_sequences=False, return_state=False, name='LastLSTM')(
            outputs)

    elif units == 8:
        outputs = tf.keras.layers.LSTM(8*2, stateful=True, return_sequences=True, return_state=False, trainable=training_lstms, name='LSTM')(
            inputs)
        outputs = tf.keras.layers.LSTM(4*2, stateful=True, return_sequences=False, return_state=False, trainable=training_lstms, name='LastLSTM')(
            outputs)

    if conditioning_size != 0:
        cond_inputs = tf.keras.layers.Input(batch_shape=(
            b_size, conditioning_size), name='cond_inputs')

    outputs_ = tf.keras.layers.Dense(1, trainable=training_just_out, name='OutLayer')(outputs)

    if conditioning_size != 0:
        if enable_second_output: # we want only the latent space of last layer
            model = tf.keras.models.Model(
                [inputs, cond_inputs], outputs)

        else:

            model = tf.keras.models.Model(
                [inputs, cond_inputs], outputs_)
    else:
        if enable_second_output: # we want only the latent space of last layer
            model = tf.keras.models.Model(
                inputs, outputs)

        else:
            model = tf.keras.models.Model(
                inputs, outputs_)

    model.summary()
    return model
