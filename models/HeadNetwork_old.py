import torch
import torch.nn as nn
from models.utils import PositionalEncoding, WeightsLoader, MaskedNorm

class HeadNetwork(nn.Module):
    def __init__(self, CFG) -> None:
        super().__init__()
        self.CFG = CFG
        self.residual_connection = CFG.residual_connection
        self.layer_norm1 = nn.LayerNorm(CFG.input_size, eps=1e-06) ### NOT USED!
        self.fc1 = nn.Linear(CFG.input_size, CFG.hidden_size)
        self.bn1 = nn.BatchNorm1d(num_features=CFG.hidden_size)
        self.relu1 = nn.ReLU()

        self.PE = PositionalEncoding(d_model=CFG.hidden_size, N=10000)
        self.dropout1 = nn.Dropout(CFG.head_dropout)
        
        self.temp_conv_block = nn.Sequential(
                                nn.Conv1d(CFG.hidden_size, CFG.ff_size, kernel_size=CFG.ff_kernel_size, stride=1, padding='same'),
                                nn.ReLU(),
                                nn.Dropout(p=CFG.head_dropout),
                                nn.Conv1d(CFG.ff_size, CFG.hidden_size, kernel_size=CFG.ff_kernel_size, stride=1, padding='same'),
                                nn.Dropout(p=CFG.head_dropout))
        
        self.layer_norm2 = nn.LayerNorm(CFG.hidden_size, eps=1e-06)
        self.layer_norm3 = nn.LayerNorm(CFG.hidden_size, eps=1e-06)

        self.translation_layer = nn.Linear(CFG.hidden_size, CFG.n_classes)
        self.Softmax = nn.Softmax(dim=-1)

        self.weightsLoader = WeightsLoader(self.state_dict(), CFG.head_weights_filename)
    
    def load_weights(self, verbose):
        print(f"Loading weights from {self.CFG.head_weights_filename.split('/')[0]}")
        self.weightsLoader.load(verbose=verbose)

    def forward(self, x, mask):
        # Input: x = [N x T/4 x 832]

        # Head
        x = self.layer_norm1(x)
        x = self.fc1(x)
        
        x = self.bn1(x.transpose(1, 2))
        x = self.relu1(x)
        x = self.PE(x.transpose(1, 2))
        x = self.dropout1(x)
        
        # temporal convolutional block
        if self.residual_connection:
            gloss_reps = self.temp_conv_block(self.layer_norm2(x).transpose(1, 2)).transpose(1, 2) + x 
        else:
            gloss_reps = self.temp_conv_block(self.layer_norm2(x).transpose(1, 2)).transpose(1, 2)

        # gloss translation layer
        logits = self.translation_layer(self.layer_norm3(gloss_reps)) 
        gloss_probs = self.Softmax(logits) 
        return gloss_probs, gloss_reps