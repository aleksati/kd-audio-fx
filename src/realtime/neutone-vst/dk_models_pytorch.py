import torch
import torch.nn as nn


# 64 unit student models
class DK_LSTM_Student(nn.Module):
    def __init__(self, input_dim=1, units=64, hidden2=8):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=units, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=units, hidden_size=8, batch_first=True)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        residual = x
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.out(x)
        x = x + residual  # residual connection
        return x
    
# Teacher model
class DK_LSTM_Teacher(nn.Module):
    def __init__(self, units=[8, 16, 32, 64, 32, 16, 8], input_dim=1):
        super(DK_LSTM_Teacher, self).__init__()
        self.input_dim = input_dim
        self.units = units

        # Define stacked LSTMs
        self.lstm_layers = nn.ModuleList()
        in_dim = input_dim
        for u in units:
            self.lstm_layers.append(nn.LSTM(input_size=in_dim, hidden_size=u, batch_first=True))
            in_dim = u  # next input is current output size

        # Final dense layer
        self.dense = nn.Linear(units[-1], 1)

    def forward(self, x):
        out = x
        for lstm in self.lstm_layers:
            out, _ = lstm(out)  # ignore hidden state

        out = self.dense(out)
        # Residual connection: match input and output shapes
        if out.shape[-1] == x.shape[-1]:
            out = out + x
        else:
            # Broadcast or pad input if needed
            out = out + x[..., :1]  # assuming input_dim = 1
        return out

# Teacher tester
#model = DK_LSTM_Teacher()
#x = torch.randn(2399, 2048, 1)  # [batch_size, seq_len, input_dim]
#y = model(x)
#print(y.shape)  # should be [2399, 2048, 1]

model = DK_LSTM_Student()
model.eval()

# Export to TorchScript
example_input = torch.randn(1, 100, 1)
traced = torch.jit.trace(model, example_input)
traced.save("neutone_lstm_model.pt")