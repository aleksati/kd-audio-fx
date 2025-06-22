import torch as tr
import torch.nn as nn
import tensorflow as tf

from torch import Tensor

from typing import Dict, List
import logging
import os

from neutone_sdk import WaveformToWaveformBase

# -------------------------------
# PyTorch student model
# -------------------------------
class DK_LSTM_Student_Pytorch(nn.Module):
    def __init__(self, input_dim=1, units=64, hidden2=8):
        # Edit the units argument to try different models (8, 16, 32, 64)
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=units, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=units, hidden_size=hidden2, batch_first=True)
        self.out = nn.Linear(hidden2, 1)

    def forward(self, x):
        residual = x
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.out(x)
        x = x + residual  # residual connection
        return x

# -------------------------------
# H5 Keras student model
# -------------------------------
def DK_LSTM_Student_Keras(units, input_dim=1, b_size=1, name=""):
    inputs = tf.keras.layers.Input(shape=(None, input_dim), batch_size=b_size, name='input')
    
    x = tf.keras.layers.LSTM(units, stateful=False, return_sequences=True, name="LSTM")(inputs)
    x = tf.keras.layers.LSTM(8, stateful=False, return_sequences=True, name="LSTM2")(x)
    x = tf.keras.layers.Dense(1, name='OutLayer')(x)
    
    # residual connection
    outputs = tf.keras.layers.Add(name="residual_add")([x, inputs])

    model = tf.keras.models.Model(inputs, outputs, name=name)

    return model


logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

class DK_LSTM_Student_Pytorch_Wrapper(WaveformToWaveformBase):
    '''
    Neutone student model wrapper
    '''

    def get_model_name(self) -> str:
        return "distilled_student"

    def get_model_authors(self) -> List[str]:
        return ["Aleksander Tidemann, Riccardo Simionato"]

    def get_model_short_description(self) -> str:
        return "Audio distortion effect using Knowledge Distillation for compression."

    def get_model_long_description(self) -> str:
        return "Audio distortion effect using RNN LSTM neural networks and Knowledge Distillation for compression."

    def get_technical_description(self) -> str:
        return "Knowledge distillation is an ML compression technique where knowledge from larger teacher networks are distilled into smaller student networks. We explore how knowledge distillation can be used to optimize virtual-analog models of audio distortion effects. In particular, we propose an audio-to-audio LSTM architecture for real-time regression tasks where student networks are trained to mimic the internal representations of teachers, known as feature-based knowledge distillation."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "code" : "https://github.com/aleksati/kd-audio-fx"
        }

    def get_tags(self) -> List[str]:
        return ["knowledge-distillation", "LSTM", "RNN", "distortion", "va-modeling"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return True

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return True

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        #return [48000]  # Update if you want to support more
        return []  # support all

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        #return [512]  # This should match what your model was trained with
        return [] # support all buffer sizes

    def aggregate_params(self, params: Tensor) -> Tensor:
        return params  # No parameters used

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        # Input: (batch, samples)
        x = x.unsqueeze(-1)  # -> (batch, samples, 1)
        y = self.model(x)
        return y.squeeze(-1)  # -> (batch, samples)