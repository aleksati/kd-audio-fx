import torch
import torch.nn as nn
import h5py
import numpy as np
import tensorflow as tf

# Config
UNITS = 64
DEVICE = "drdrive"
MODEL_TYPE = "non_distilled"
# Path to best weights
ckpt_dir = f'../../models/students_{MODEL_TYPE}/LSTM_{DEVICE}_dk_{UNITS}/checkpoints/best/best.ckpt'
NAME = f'{DEVICE}_student_{MODEL_TYPE}_{UNITS}'

# -------------------------------
# H5 Keras student model
# -------------------------------
def DK_LSTM_Student_Keras(units, input_dim=1, b_size=1):
    inputs = tf.keras.layers.Input(shape=(None, input_dim), batch_size=b_size, name='input')
    
    x = tf.keras.layers.LSTM(units, stateful=False, return_sequences=True, name="LSTM")(inputs)
    x = tf.keras.layers.LSTM(8, stateful=False, return_sequences=True, name="LSTM2")(x)
    x = tf.keras.layers.Dense(1, name='OutLayer')(x)
    
    outputs = tf.keras.layers.Add(name="residual_add")([x, inputs])  # residual connection

    model = tf.keras.models.Model(inputs, outputs, name=NAME)
    return model

# -------------------------------
# Create Keras model, load weights from KD experiment, and save h5 model
# -------------------------------
def create_and_save_keras_model_and_weights(model):
    # Create model, load the best training weights, and save to local
    model = model(units=UNITS)
    model.load_weights(ckpt_dir).expect_partial()
    model.save(f'./models/{NAME}.h5')
    model.save_weights(f'./models/{NAME}_weights.h5')

# -------------------------------
# PyTorch student model
# -------------------------------
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
        x = x + residual  # residual connection
        return x

# -------------------------------
# Load from Keras H5 and map weights
# -------------------------------
def transfer_weights_from_keras_to_pytorch(h5_path, model):

    # Optional - Use this code first to inspect the weight structure to know how to edit the below code()
    # import h5py
    # with h5py.File(f'./models/{NAME}_weights.h5', "r") as f:
    #     f.visit(lambda x: print(x))

    with h5py.File(h5_path, "r") as f:
        # LSTM1
        lstm1 = f["LSTM"]["LSTM"]["lstm_cell"]
        kernel = lstm1["kernel:0"][()]               # (in, 4*out)
        recurrent_kernel = lstm1["recurrent_kernel:0"][()]  # (out, 4*out)
        bias = lstm1["bias:0"][()]                   # (4*out,)

        model.lstm1.weight_ih_l0.data.copy_(torch.tensor(kernel.T))
        model.lstm1.weight_hh_l0.data.copy_(torch.tensor(recurrent_kernel.T))
        model.lstm1.bias_ih_l0.data.copy_(torch.tensor(bias))
        model.lstm1.bias_hh_l0.data.zero_()  # PyTorch uses two bias vectors

        print("LSTM1 loaded")

        # LSTM2
        lstm2 = f["LSTM2"]["LSTM2"]["lstm_cell"]
        kernel = lstm2["kernel:0"][()]
        recurrent_kernel = lstm2["recurrent_kernel:0"][()]
        bias = lstm2["bias:0"][()]

        model.lstm2.weight_ih_l0.data.copy_(torch.tensor(kernel.T))
        model.lstm2.weight_hh_l0.data.copy_(torch.tensor(recurrent_kernel.T))
        model.lstm2.bias_ih_l0.data.copy_(torch.tensor(bias))
        model.lstm2.bias_hh_l0.data.zero_()

        print("LSTM2 loaded")

        # Dense layer
        dense = f["OutLayer"]["OutLayer"]
        dense_w = dense["kernel:0"][()]  # (in, out)
        dense_b = dense["bias:0"][()]    # (out,)

        model.out.weight.data.copy_(torch.tensor(dense_w.T))  # PyTorch: (out, in)
        model.out.bias.data.copy_(torch.tensor(dense_b))

        print("Dense output layer loaded")

# -------------------------------
# Convert Keres model to Pytorch
# -------------------------------
def convert_keras_model_to_pytorch():
    # Paths
    # h5_model_path = f'./models/{NAME}.h5'
    h5_model_path = f'./models/{NAME}_weights.h5'
    torch_model_path = f'./models/{NAME}.pth'

    # Initialize model
    pytorch_model = DK_LSTM_Student_Pytorch()

    # Transfer weights
    transfer_weights_from_keras_to_pytorch(h5_model_path, pytorch_model)

    # Save the PyTorch weights
    torch.save(pytorch_model.state_dict(), torch_model_path)
    print(f"\n Model successfully saved to '{torch_model_path}'")

# -------------------------------
# Validate that the Keras and PyTorch models produce the same output when given the same input
# -------------------------------
def validate(): 
    # Load PyTorch model
    pytorch_model = DK_LSTM_Student_Pytorch()
    pytorch_model.load_state_dict(torch.load(f'./models/{NAME}.pth'))
    pytorch_model.eval()

    # Load Keras model
    keras_model = tf.keras.models.load_model(f'./models/{NAME}.h5')

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
        output_pt = pytorch_model(input_pt).numpy()

    # Keras inference
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

# Step 1
#create_and_save_keras_model_and_weights(DK_LSTM_Student_Keras)

# Step 2
#convert_keras_model_to_pytorch()

# Step 3 
# Validation between keras and pytorch
#validate()

# Step 4
# Wrap PyTorch model in Neutone wrapper and export as Neutone model