import torch
import h5py
import numpy as np
import tensorflow as tf
from models_stateful import DK_LSTM_Student_Pytorch, DK_LSTM_Student_Keras, DK_LSTM_Student_Pytorch_Wrapper

from neutone_sdk.utils import save_neutone_model
import torch as tr

import pathlib


def save_best_keras_weights(units=64, model_name="", models_dir="./models", ckpt_dir=""):

    '''
    Load and save Keras student model and weights from KD experiment as H5 model.

    args:
    units = the number of units in the first LSTM layer
    model_name = name of the keras model
    model_dir = directory where the models are.
    ckpt_dir = path to best checkpoint weights in KD experiment
    '''
    
    # Create model, load the best training weights, and save to local
    model = DK_LSTM_Student_Keras(units=units, name=model_name, b_size=1)
    model.load_weights(ckpt_dir).expect_partial()
    model.save(f'{models_dir}/{model_name}/{model_name}.h5')
    model.save_weights(f'{models_dir}/{model_name}/{model_name}_weights.h5')


def transfer_weights_from_keras_to_pytorch(h5_weights_path, pytorch_model):
    '''
    Loads Keras H5 student model (2 LSTM layers) and maps weights to match Pytorch models.

    args:
    h5_path = path to h5 keras model weights.
    pytorch_model = the equivelent pytorch model
    '''

    # Optional - Use this code first to inspect the weight structure to know how to edit the below code()
    # import h5py
    # with h5py.File(f'./models/drdrive_student_distilled_8_weights.h5', "r") as f:
    #     f.visit(lambda x: print(x))

    with h5py.File(h5_weights_path, "r") as f:

        # LSTM1
        lstm1 = f["LSTM1"]["LSTM1"]["lstm_cell"]
        kernel = lstm1["kernel:0"][()]               # (in, 4*out)
        recurrent_kernel = lstm1["recurrent_kernel:0"][()]  # (out, 4*out)
        bias = lstm1["bias:0"][()]                   # (4*out,)

        pytorch_model.lstm1.weight_ih_l0.data.copy_(torch.tensor(kernel.T))
        pytorch_model.lstm1.weight_hh_l0.data.copy_(torch.tensor(recurrent_kernel.T))
        pytorch_model.lstm1.bias_ih_l0.data.copy_(torch.tensor(bias))
        pytorch_model.lstm1.bias_hh_l0.data.zero_()  # PyTorch uses two bias vectors

        print("LSTM1 loaded")

        # LSTM2
        lstm2 = f["LSTM2"]["LSTM2"]["lstm_cell"]
        kernel = lstm2["kernel:0"][()]
        recurrent_kernel = lstm2["recurrent_kernel:0"][()]
        bias = lstm2["bias:0"][()]

        pytorch_model.lstm2.weight_ih_l0.data.copy_(torch.tensor(kernel.T))
        pytorch_model.lstm2.weight_hh_l0.data.copy_(torch.tensor(recurrent_kernel.T))
        pytorch_model.lstm2.bias_ih_l0.data.copy_(torch.tensor(bias))
        pytorch_model.lstm2.bias_hh_l0.data.zero_()

        print("LSTM2 loaded")

        # Dense layer
        dense = f["OutLayer"]["OutLayer"]
        dense_w = dense["kernel:0"][()]  # (in, out)
        dense_b = dense["bias:0"][()]    # (out,)

        pytorch_model.out.weight.data.copy_(torch.tensor(dense_w.T))  # PyTorch: (out, in)
        pytorch_model.out.bias.data.copy_(torch.tensor(dense_b))

        print("Dense output layer loaded")


def convert_keras_to_pytorch(model_name="", models_dir="./models", units=64):
    '''
    Convert Keras model and weights to Pytorch model.

    args:
    model_name = name of Keras model and weights to convert.
    models_dir = directory where the models are.
    units = the number of units in the first LSTM layer
    '''
    # Paths
    # h5_model_path = f'./models/{NAME}.h5'
    h5_model_path = f'{models_dir}/{model_name}/{model_name}_weights.h5'
    torch_model_path = f'{models_dir}/{model_name}/{model_name}.pth'

    # Initialize model
    pytorch_model = DK_LSTM_Student_Pytorch(units=units)

    # Transfer weights
    transfer_weights_from_keras_to_pytorch(h5_model_path, pytorch_model)

    # Save the PyTorch weights
    torch.save(pytorch_model.state_dict(), torch_model_path)
    print(f"\n Model successfully saved to '{torch_model_path}'")


def validate(model_name="", models_dir="./models", units=64):
    '''
    Validate that the Keras and PyTorch models produce the same output when given the same input

    args:
    model_name = name of pytorch and keras models to compare
    models_dir = directory where the models are.
    units = the number of units in the first LSTM layer
    '''
    # Load PyTorch model
    pytorch_model = DK_LSTM_Student_Pytorch(units=units)
    pytorch_model.load_state_dict(torch.load(f'{models_dir}/{model_name}/{model_name}.pth'))
    pytorch_model.eval()

    # Load Keras model
    keras_model = tf.keras.models.load_model(f'./models/{model_name}/{model_name}.h5')

    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # Input parameters
    batch_size = 1
    seq_len = 10
    input_dim = 1

    # Generate same input
    input_np = np.random.rand(batch_size, seq_len, input_dim).astype(np.float32)

    # PyTorch inference
    input_pt = torch.tensor(input_np)
    with torch.no_grad():
        output_pt, _ = pytorch_model(input_pt, state=None)  # pass initial state=None
        output_pt = output_pt.numpy()

    # Keras inference (stateful)
    keras_model.reset_states()  # ensure clean hidden state
    input_keras = tf.convert_to_tensor(input_np)
    output_keras = keras_model(input_keras).numpy()

    # Compare
    diff = np.abs(output_pt - output_keras)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"[Validation] Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    if max_diff < 1e-4:
        print("Outputs match closely! Success")
    else:
        print("Outputs differ — check for precision or conversion issues..")


def create_neutone_model(models_dir="./models", units=64, model_name=""):
    '''
    Create a neutone model from an existing Pytorch model/weights (.pth)

    args:
    model_name = name of the model and weights.
    models_dir = Dir to save the neutone model
    units = the number of units in the first LSTM layer
    '''

    # parser = ArgumentParser()
    # parser.add_argument("--save_dir", type=str, help="Path to save neutone model", default="./models/")
    # parser.add_argument("--weights", type=str, help="Path to model weights (.pth)", default="./models/drdrive_student_non_distilled_64.pth")
    # args = parser.parse_args()

    model = DK_LSTM_Student_Pytorch(units=units)
    model.load_state_dict(tr.load(f'{models_dir}/{model_name}/{model_name}.pth', map_location="cpu"))
    model.eval()

    # create full path
    save_dir = pathlib.Path(f'{models_dir}/{model_name}')

    wrapper = DK_LSTM_Student_Pytorch_Wrapper(model)
    save_neutone_model(wrapper, save_dir, dump_samples=True, submission=True)