import torch
import torch.nn as nn

class VGG(nn.Module):    
    
    def __init__(self, architecture, in_channels=3, n_classes=1000):
        super().__init__()
        
        # Convolutional layers:
                
        n_convs = {'A': [1, 1, 2, 2, 2],
                   'B': [2, 2, 2, 2, 2],
                   'C': [2, 2, 3, 3, 3],
                   'D': [2, 2, 3, 3, 3],
                   'E': [2, 2, 4, 4, 4]}[architecture]
        channels = [in_channels, 64, 128, 256, 512] + [512]
        
        for n_block in range(5):
            block = []
            for conv in range(n_convs[n_block]):
                block += [nn.Conv2d(in_channels=channels[n_block] if conv == 0 else channels[n_block+1],
                                    out_channels=channels[n_block+1],
                                    kernel_size = 3 if not (architecture == 'C' and conv == 2) else 1,
                                    padding='same'), 
                          nn.ReLU()]
            block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.add_module(f'block{n_block+1}', nn.Sequential(*block))
        
        # Fully connected layers:
        
        features = [7*7*512, 4096, 4096] + [n_classes]
        dense = []
        for n_fc in range(3):
            dense += [nn.Linear(in_features=features[n_fc], out_features=features[n_fc + 1]),
                      nn.ReLU()]
        self.dense = nn.Sequential(*dense[:-1])
            
    def forward(self, x):  # net input: Cx224x224 image.
        
        for block in [self.block1, self.block2, self.block3, self.block4, self.block5]: 
            x = block(x)
        x = nn.Flatten()(x)
        x = self.dense(x)
        return x