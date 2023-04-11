import torch
import torch.nn as nn
from models.utils import PositionalEncoding, WeightsLoader

class HeadNetwork(nn.Module):
    def __init__(self, CFG) -> None:
        super().__init__()
        self.residual_connection = CFG.residual_connection
        self.layer_norm1 = nn.LayerNorm(CFG.input_size, eps=1e-06)
        self.fc1 = nn.Linear(CFG.input_size, CFG.hidden_size)
        self.bn1 = nn.BatchNorm1d(num_features=CFG.hidden_size)
        self.relu1 = nn.ReLU()

        self.PE = PositionalEncoding(d_model=CFG.hidden_size, N=10000)
        self.dropout1 = nn.Dropout(0.1)
        self.pe = nn.Sequential(
                    PositionalEncoding(d_model=CFG.hidden_size, N=10000),
                    nn.Dropout(p=0.1))
        
        self.temp_conv_block = nn.Sequential(
                                nn.Conv1d(CFG.hidden_size, CFG.ff_size, kernel_size=CFG.ff_kernel_size, stride=1, padding='same'),
                                nn.ReLU(),
                                nn.Dropout(p=0.1),
                                nn.Conv1d(CFG.ff_size, CFG.hidden_size, kernel_size=CFG.ff_kernel_size, stride=1, padding='same'),
                                nn.Dropout(p=0.1))

        
        self.layer_norm2 = nn.LayerNorm(CFG.hidden_size, eps=1e-06)
        self.layer_norm3 = nn.LayerNorm(CFG.hidden_size, eps=1e-06)

        self.translation_layer = nn.Linear(CFG.hidden_size, CFG.n_classes)
        self.Softmax = nn.Softmax(dim=-1)

        self.weightsLoader = WeightsLoader(self.state_dict(), CFG.head_weights_filename)
    
    def load_weights(self):
        print(f"Loading weights from {self.CFG.head_weights_filename.split('/')[0]}")
        self.weightsLoader.load(verbose=True)

    def forward(self, x):
        #Input: x = [N x T/4 x 832]

        # Head
        x = self.layer_norm1(x)
        x = self.fc1(x)
        
        x = self.bn1(x.transpose(1, 2)) # N x 512 x T/4
        x = self.relu1(x)

        x = self.PE(x.transpose(1, 2)) # N x T/4 x 512
        x = self.dropout1(x)
        # temporal convolutional block
        if self.residual_connection:
            x = self.temp_conv_block(self.layer_norm2(x).transpose(1, 2)).transpose(1, 2) + x 
        else:
            x = self.temp_conv_block(self.layer_norm2(x).transpose(1, 2)).transpose(1, 2)
        x = self.layer_norm3(x)

        # gloss translation layer
        logits = self.translation_layer(x)
        gloss_probs = self.Softmax(logits)
        return gloss_probs