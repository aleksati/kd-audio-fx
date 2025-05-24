import logging
import os
import pathlib
from argparse import ArgumentParser
from typing import Dict, List

import torch as tr
import torch.nn as nn
from torch import Tensor

from neutone_sdk import WaveformToWaveformBase
from neutone_sdk.utils import save_neutone_model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


# Your student model
class DK_LSTM_Student_Pytorch(nn.Module):
    def __init__(self, input_dim=1, units=64, hidden2=8):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=units, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=units, hidden_size=hidden2, batch_first=True)
        self.out = nn.Linear(hidden2, 1)

    def forward(self, x):
        residual = x
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.out(x)
        x = x + residual
        return x


# Neutone student model wrapper
class DK_LSTM_Wrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "drdr_student_non_distilled_64"

    def get_model_authors(self) -> List[str]:
        return ["Aleksander Tidemann, Riccardo Simionato"]

    def get_model_short_description(self) -> str:
        return "Audio distortion effect using Knowledge Distillation for compression."

    def get_model_long_description(self) -> str:
        return "Audio distortion effect thats modeled on analog DrDrive distortion pedal using RNN LSTM neural networks and Knowledge Distillation for compression."

    def get_technical_description(self) -> str:
        return "knowledge distillation for compressing RNN models of audio distortion effects. In particular, we propose an audio-to-audio LSTM architecture for realtime regression tasks where small audio effect networks are trained to mimic the internal representations of more extensive networks, known as feature-based knowledge distillation."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "code" : "https://github.com/aleksati/kd-audio-fx"
        }

    def get_tags(self) -> List[str]:
        return ["knowledge-distillation", "LSTM", "RNN", "distortion", "va-modeling", "drdrive"]

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

################################################################
# Check if model accepts different Buffer sizes.
# model = DK_LSTM_Student_Pytorch()
# model.load_state_dict(tr.load("./models/drdrive_student_distilled_64.pth", map_location="cpu"))
# model.eval()
# wrapper = DK_LSTM_Wrapper(model)

# for T in [256, 512, 1024]:
#     # Test with mono input (batch, time)
#     mono_input = tr.randn(1, T)  # (1, T)
#     output = wrapper.do_forward_pass(mono_input, {})
#     print(f"Input shape: {mono_input.shape}, Output shape: {output.shape}")
##############################################################

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, help="Path to save neutone model", default="./models/")
    parser.add_argument("--weights", type=str, help="Path to model weights (.pth)", default="./models/drdrive_student_non_distilled_64.pth")
    args = parser.parse_args()
    save_dir = pathlib.Path(args.save_dir)

    model = DK_LSTM_Student_Pytorch()
    model.load_state_dict(tr.load(args.weights, map_location="cpu"))
    model.eval()

    wrapper = DK_LSTM_Wrapper(model)
    save_neutone_model(wrapper, save_dir, dump_samples=True, submission=True)