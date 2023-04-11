import torch.nn as nn
from models.utils import PositionalEncoding, MaskedNorm, WeightsLoader

class HeadNetwork(nn.Module):
    def __init__(self, CFG, n_classes, input_size, hidden_size, ff_size, ff_kernel_size, residual_connection) -> None:
        super().__init__()
        self.residual_connection = residual_connection
        self.layer_norm1 = nn.LayerNorm(input_size, eps=1e-06)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.relu1 = nn.ReLU()

        self.PE = PositionalEncoding(d_model=hidden_size, N=10000)
        self.dropout1 = nn.Dropout(0.1)
        self.pe = nn.Sequential(
                    PositionalEncoding(d_model=hidden_size, N=10000),
                    nn.Dropout(p=0.1))
        
        self.temp_conv_block = nn.Sequential(
                                nn.Conv1d(hidden_size, ff_size, kernel_size=3, stride=1, padding='same'),
                                nn.ReLU(),
                                nn.Dropout(p=0.1),
                                nn.Conv1d(ff_size, hidden_size, kernel_size=3, stride=1, padding='same'),
                                nn.Dropout(p=0.1))

        
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-06)
        self.layer_norm3 = nn.LayerNorm(hidden_size, eps=1e-06)
        self.translation_layer = nn.Linear(hidden_size, n_classes)

        self.weightsLoader = WeightsLoader(self.state_dict(), CFG.weights_filename)
    
    def load_weights(self):
        print(f"Loading weights from {self.CFG.weights_filename.split('/')[0]}")
        self.weightsLoader.load(verbose=True)
        
        

    def forward(self, x, mask):
        #Input: x = [N x T/4 x 832]

        # Head
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.bn1(x, mask)
        x = self.relu1(x)

        x = self.PE(x)
        x = self.dropout1(x)

        # temporal convolutional block
        print("BEFORE CONV", x.size())
        if self.residual_connection:
            x = self.temp_conv_block(self.layer_norm2(x).transpose(1, 2)).transpose(1, 2) + x 
        else:
            x = self.temp_conv_block(self.layer_norm2(x).transpose(1, 2)).transpose(1, 2)
        print("AFTER CONV", x.size())
        x = self.layer_norm3(x)
        # gloss translation layer
        logits = self.translation_layer(x)
        
        print("Logits: ", logits.size())
        return logits
