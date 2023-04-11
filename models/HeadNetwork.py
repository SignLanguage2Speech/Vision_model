import torch.nn as nn


class HeadNetwork(nn.Module):
    def __init__(self, n_classes, input_size, hidden_size, ff_size, ff_kernel_size, residual_connection=True) -> None:
        super().__init__()
        self.residual_connection = residual_connection
        self.head1 = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    MaskedNorm(num_features=hidden_size),
                    nn.ReLU(),
                    PositionalEncoding(hidden_size),
                    nn.Droput(p=0.1)
                    )
        
        self.temp_conv_block = nn.Sequential(
                                nn.Conv1d(hidden_size, ff_size, kernel_size=3, stride=1, padding='same'),
                                nn.ReLU(),
                                nn.Dropout(p=0.1),
                                nn.Conv1d(ff_size, hidden_size, kernel_size=3, stride=1, padding='same'),
                                nn.Dropout(p=0.1))
        
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-06)

        self.translation_layer = nn.Linear(hidden_size, n_classes)
    
    def load_weights(self, model, ckpt='WLASL'):
        ckpts = ['phoenix', 'wlasl', 'kinetics', 'how2sign']
        assert(ckpt.lower() in ckpts, print(f"{ckpt} is not a valid checkpoint!\n Valid ones are:\n{ckpts}"))
        print(f"Loading weights for {ckpt}")
        
    def forward(self, x):
        # x = [N x T x 832]
        x2 = self.head1(x)
        if self.residual_connection:
            x2 = self.temp_conv_block(x2) + x
        else:
            x2 = self.temp_conv_block(x2)
        logits = self.translation_layer(x2)

        return logits
