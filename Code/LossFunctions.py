import tensorflow as tf
from tensorflow.keras import backend as K


class DC_PreEmph(nn.Module):
    def __init__(self, R=0.995):
        super(DC_PreEmph, self).__init__()

        t, ir = signal.dimpulse(signal.dlti([1, -1], [1, -R]), n=2000)
        ir = ir[0][:, 0]

        self.zPad = len(ir) - 1
        self.pars = torch.flipud(torch.tensor(ir, requires_grad=False, dtype=torch.FloatTensor.dtype)).unsqueeze(
            0).unsqueeze(0)

    def forward(self, output, target):
        output = output.permute(0, 2, 1)
        target = target.permute(0, 2, 1)

        # zero pad the input/target so the filtered signal is the same length
        output = torch.cat((torch.zeros(output.shape[0], 1, self.zPad).type_as(output), output), dim=2)
        target = torch.cat((torch.zeros(output.shape[0], 1, self.zPad).type_as(output), target), dim=2)
        # Apply pre-emph filter, permute because the dimension order is different for RNNs and Convs in pytorch...
        output = nn.functional.conv1d(output, self.pars.type_as(output), bias=None)
        target = nn.functional.conv1d(target, self.pars.type_as(target), bias=None)

        return output.permute(0, 2, 1), target.permute(0, 2, 1)

class ESRloss(tf.keras.losses.Loss):
    def __init__(self, delta=1e-6, name="ESR", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = 0.00001
    def call(self, y_true, y_pred):

        loss =tf.divide(K.mean(K.square(y_pred - y_true)), K.mean(K.square(y_true) + self.delta))
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