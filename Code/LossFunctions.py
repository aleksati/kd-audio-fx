import tensorflow as tf
from tensorflow.keras import backend as K


class DCloss(tf.keras.losses.Loss):
    def __init__(self, delta=1e-6, name="ESR", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = 0.00001
    def call(self, y_true, y_pred):

        #loss = tf.divide(K.mean(K.square(y_true - y_pred)), K.mean(K.square(y_true) + self.delta))
        #loss = (K.mean(K.square(y_true - y_pred)))
        loss = K.abs((K.mean((y_true))) - (K.mean((y_pred))))
        return loss

    def get_config(self):
        config = {
            'delta': self.delta
        }
        base_config = super().get_config()
        return {**base_config, **config}


class ESRloss(tf.keras.losses.Loss):
    def __init__(self, delta=1e-6, name="ESR", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = 0.00001
    def call(self, y_true, y_pred):

        loss = tf.divide(K.square(y_true - y_pred), K.square(y_true) + self.delta)
        return loss

    def get_config(self):
        config = {
            'delta': self.delta
        }
        base_config = super().get_config()
        return {**base_config, **config}



class NormMSELoss(tf.keras.losses.Loss):
    def __init__(self, delta=1e-6, name="NMSE", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = delta
    def call(self, y_true, y_pred):

        loss = tf.divide(tf.keras.metrics.mean_squared_error(y_true, y_pred), tf.norm(y_true, ord=1) + self.delta)
        return loss

    def get_config(self):
        config = {
            'delta': self.delta
        }
        base_config = super().get_config()
        return {**base_config, **config}



class NormMAELoss(tf.keras.losses.Loss):
    def __init__(self, delta=1e-6, name="NMAE", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = delta
    def call(self, y_true, y_pred):

        loss = tf.divide(tf.keras.metrics.mean_absolute_error(y_true, y_pred), tf.norm(y_true, ord=1) + self.delta)
        return loss

    def get_config(self):
        config = {
            'delta': self.delta
        }
        base_config = super().get_config()
        return {**base_config, **config}