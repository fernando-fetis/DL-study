import torch
import torch.nn as nn

class ResNet(nn.Module):
    """ResNet architecture based on the paper 'Deep Residual Learning for Image Recognition' by He et al."""

    def __init__(self,
                 architecture: int,  # number of layers. Available: 18, 34, 50, 101, 152.
                 in_channels: int = 3,  # RGB image as default.
                 n_classes: int = 1000):  # original implementation for ImageNet.
        
        super().__init__()
        assert architecture in [18, 34, 50, 101, 152], 'Undefined architecture.'
        
        self.initial_conv = nn.Sequential(nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
                                          nn.BatchNorm2d(64))
        
        resblocks_per_group = {18:  [2, 2, 2,  2],
                               34:  [3, 4, 6,  3],
                               50:  [3, 4, 6,  3],
                               101: [3, 4, 23, 3],
                               152: [3, 8, 36, 3]}
        
        # Residual blocks groups:
        for n_group in range(4):
            n_blocks = resblocks_per_group[architecture][n_group]
            group = [_ResidualBlock(architecture, n_group, n_block) for n_block in range(n_blocks)]
            self.add_module(f'group{n_group}', nn.Sequential(*group))
        
        final_pooling_channels = 512 if architecture in [18, 34] else 2048
        self.final_dense = nn.Linear(final_pooling_channels, n_classes)

    def forward(self, x: torch.Tensor):  # expected input shape: (-1, in_channels, 224, 224).
        
        x = nn.ReLU()(self.initial_conv(x))
        x = nn.MaxPool2d(2, stride=2)(x)
        
        for group in [self.group0, self.group1, self.group2, self.group3]:
            x = group(x)
            
        x = nn.AvgPool2d(7)(x)  # global average pooling.
        x = self.final_dense(nn.Flatten()(x))
        return x

class _ResidualBlock(nn.Module):
    """Internal class to build the residual blocks of the network."""

    def __init__(self,
                 architecture: int,  # chosen architecture.
                 n_group: int,  # blocks group being built (0 to 3).
                 n_block: int):  # block number in group being built (to recognize the first block).
        
        super().__init__()
        
        if architecture in [18, 34]:  # 2 convolutions per block.
            mode = 'shallow'
        elif architecture in [50, 101, 152]:  # 3 convolutions per block (bottleneck).
            mode = 'deep'
            
        # Resblocks parameters:        group0           group1           group2            group3
        out_channels = {'shallow': [[64, 64     ], [128, 128     ], [256, 256      ], [512, 512      ]],
                        'deep':    [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]}[mode]
        filter_sizes = {'shallow': [[3,  3      ], [3,   3       ], [3,   3        ], [3,   3        ]],
                        'deep':    [[1,  3,  1  ], [1,   3,   1  ], [1,   3,   1   ], [1,   3,   1   ]]}[mode]
        
        # Residual block convolutions:
        block_convs = []
        n_convs = len(out_channels[n_group])
        for n_conv in range(n_convs):
            
            # Convolution in-channels:
            if n_block == n_conv == 0:  # first convolution of the first block in the group.
                in_channels = out_channels[n_group - 1][-1]
            else:
                in_channels = out_channels[n_group][n_conv - 1]
            if n_group == n_block == n_conv == 0:  # first convolution of the first block in the first group.
                in_channels = 64  # the output of the first network convolution is 64-channels.
                
            conv = nn.Conv2d(in_channels,
                             out_channels[n_group][n_conv],
                             filter_sizes[n_group][n_conv],
                             # (First convolution of the first block of each group reduces resolution, except in the first group)
                             stride = 2 if (n_block == n_conv == 0 and n_group != 0) else 1,
                             padding = 1 if filter_sizes[n_group][n_conv] == 3 else 0)
            
            block_convs += [conv, nn.BatchNorm2d(out_channels[n_group][n_conv]), nn.ReLU()]
            
        self.add_module(f'block_convs', nn.Sequential(*block_convs[:-1]))  # last ReLU omited (activation after the residual adittion).
        
        # Projection for residual connection (only in the first block of each group, except in the first group):
        if n_block == 0 and n_group != 0:
            self.projection = nn.Conv2d(out_channels[n_group - 1][-1], out_channels[n_group][n_conv], 1, stride=2)
            self.do_projection = True
        else:
            self.do_projection = False
                
            
    def forward(self, x: torch.Tensor):  # variable input shape.
        
        resblock_output = self.block_convs(x)
        
        if self.do_projection:
            residual_connection = self.projection(x)
        else:
            residual_connection = x
            
        return nn.ReLU()(resblock_output + residual_connection)