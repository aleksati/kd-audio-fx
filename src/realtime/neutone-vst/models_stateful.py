import torch as tr
import torch.nn as nn
import tensorflow as tf
from typing import Optional, Tuple

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
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=units, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=units, hidden_size=hidden2, batch_first=True)
        self.out = nn.Linear(hidden2, 1)

    def forward(self, x: Tensor, state: Optional[Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]] = None
            ) -> Tuple[Tensor, Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]]:

        residual = x

        h1: Optional[Tensor] = None
        c1: Optional[Tensor] = None
        h2: Optional[Tensor] = None
        c2: Optional[Tensor] = None

        if state is None:
            h1 = c1 = h2 = c2 = None
        else:
            h1, c1, h2, c2 = state

        if h1 is None or c1 is None:
            x, (h1, c1) = self.lstm1(x)
        else:
            x, (h1, c1) = self.lstm1(x, (h1, c1))

        if h2 is None or c2 is None:
            x, (h2, c2) = self.lstm2(x)
        else:
            x, (h2, c2) = self.lstm2(x, (h2, c2))

        x = self.out(x)
        x = x + residual

        return x, (h1, c1, h2, c2)

# NEW FROM RICCARDO

# NEW FROM RICCARDO
import torch.nn.functional as F

class EDModel(nn.Module):
    """
    Encoder-Decoder model with optional conditioning

    Args:
        D: number of conditioning parameters
        T: input sequence length
        units: number of LSTM units
    """

    def __init__(self, D, T, units):
        super(EDModel, self).__init__()

        self.D = D
        self.T = T
        self.units = units

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=1,
            hidden_size=units,
            batch_first=True
        )

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=1,
            hidden_size=units,
            batch_first=True
        )

        self.film_layer = nn.Linear(D, units * 2)
        self.glu = nn.Linear(units, units * 2)

        # Output layer
        self.output_layer = nn.Linear(units, 1)
        # Store batch_size for state initialization
        # Initialize hidden states as buffers (not parameters)
    #     self.register_buffer('hidden1', torch.zeros((1, 1, units)))
    #     self.register_buffer('hidden2', torch.zeros((1, 1, units)))
    #
    # def reset_states(self):
    #     """Reset hidden states (equivalent to resetting stateful LSTM states)"""
    #     self.hidden1 = torch.zeros((1, 1, units))
    #     self.hidden2 = torch.zeros((1, 1, units))

    def forward(self, encoder_inputs, decoder_inputs, cond_inputs, h, c):
        """
        Forward pass

        Args:
            encoder_inputs: (batch_size, T-1, 1) - encoder input sequence
            decoder_inputs: (batch_size, 1, 1) - decoder input
            cond_inputs: (batch_size, D) - conditioning inputs (optional)

        Returns:
            output: (batch_size, 1) - model output
        """
        # Encoder
        encoder_outputs, (new_h, new_c) = self.encoder_lstm(encoder_inputs, (h, c))
        # Update hidden states for next call (detached to avoid gradient accumulation)
        # self.hidden1 = hidden1.detach()
        # self.hidden2 = hidden2.detach()

        # Decoder with encoder hidden states as initial state
        decoder_outputs, _ = self.decoder_lstm(decoder_inputs, (new_h, new_c))

        # Remove sequence dimension from decoder output
        decoder_outputs = decoder_outputs.squeeze(1)  # (batch_size, units)

        # FiLM (Feature-wise Linear Modulation)
        film = self.film_layer(cond_inputs)  # (batch_size, units * 2)
        g, b = torch.split(film, self.units, dim=-1)

        # Apply modulation
        decoder_outputs = decoder_outputs * g + b

        # Apply GLU
        decoder_outputs = self.glu(decoder_outputs)
        # Split input in half along the last dimension
        value, gate = torch.split(decoder_outputs, 8, dim=-1)
        decoder_outputs = value * torch.nn.Softsign()(gate)

        # Final output layer
        output = self.output_layer(decoder_outputs)

        return output, (new_h, new_c)


def create_model_ED(D, T, units):
    """
    Create ED model (PyTorch version)

    Args:
        D: number of conditioning parameters
        T: input sequence length
        units: number of LSTM units

    Returns:
        model: EDModel instance
    """

    model = EDModel(D, T, units)

    # Print model summary
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model

# -------------------------------
# H5 Keras student model
# -------------------------------
def DK_LSTM_Student_Keras(units, input_dim=1, b_size=1, name=""):
    inputs = tf.keras.layers.Input(shape=(None, input_dim), batch_size=b_size, name='input')

    x = tf.keras.layers.LSTM(units, return_sequences=True, stateful=True, name="LSTM1")(inputs)
    x = tf.keras.layers.LSTM(8, return_sequences=True, stateful=True, name="LSTM2")(x)
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
    
    def __init__(self, model: DK_LSTM_Student_Pytorch):
        super().__init__(model)
        self.state: Optional[Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]] = None

    def reset_stream_state(self):
        # Called automatically by Neutone when stream resets
        self.state = None

    def get_model_name(self) -> str:
        return "distilled_student"

    def get_model_authors(self) -> List[str]:
        return ["Aleksander Tidemann, Riccardo Simionato"]

    def get_model_short_description(self) -> str:
        return "Audio distortion effect using Knowledge Distillation for compression."

    def get_model_long_description(self) -> str:
        return (
            "Audio distortion effect using RNN LSTM neural networks and "
            "Knowledge Distillation for compression."
        )

    def get_technical_description(self) -> str:
        return (
            "Knowledge distillation is an ML compression technique where knowledge from "
            "larger teacher networks are distilled into smaller student networks. "
            "We explore how knowledge distillation can be used to optimize virtual-analog "
            "models of audio distortion effects. In particular, we propose an audio-to-audio "
            "LSTM architecture for real-time regression tasks where student networks are "
            "trained to mimic the internal representations of teachers, known as feature-based "
            "knowledge distillation."
        )

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "code": "https://github.com/aleksati/kd-audio-fx"
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
        return []

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []

    def aggregate_params(self, params: Tensor) -> Tensor:
        return params

    @tr.jit.export
    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        # x shape: (batch, samples)
        x = x.unsqueeze(-1)  # -> (batch, samples, 1)

        # Use a local state variable — TorchScript does not allow assigning to self.state
        local_state = self.state
        y, new_state = self.model(x, local_state)

        # Only assign to self.state outside scripting — at runtime
        if not tr.jit.is_scripting():
            self.state = new_state

        return y.squeeze(-1)  # -> (batch, samples)
