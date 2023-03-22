import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    
    def __init__(self, in_channels, conv1x1, conv3x3_reduce, conv3x3, conv5x5_reduce, conv5x5, pool_reduce):
        super().__init__()
        
        block1 = nn.Sequential(nn.Conv2d(in_channels, conv1x1, 1), nn.ReLU())
        block2 = nn.Sequential(nn.Conv2d(in_channels, conv3x3_reduce, 1), nn.ReLU(),
                               nn.Conv2d(conv3x3_reduce, conv3x3, 3, padding=1), nn.ReLU())
        block3 = nn.Sequential(nn.Conv2d(in_channels, conv5x5_reduce, 1), nn.ReLU(),
                               nn.Conv2d(conv5x5_reduce, conv5x5, 5, padding=2), nn.ReLU())
        block4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                               nn.Conv2d(in_channels, pool_reduce, 1), nn.ReLU())
        
        self.blocks = nn.ModuleList([block1, block2, block3, block4])
        
    def forward(self, x):
        projections = []
        for block in self.blocks:
            projections.append(block(x))
        concatenated_filters = torch.cat(projections, axis=1)
        return concatenated_filters
    
class InnerClassifier(nn.Module):
    
    def __init__(self, in_channels, n_classes):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, 128, 1)
        self.dense1 = nn.Linear(4*4*128, 1024)  # avgpool output resolution: 4x4.
        self.dense2 = nn.Linear(1024, n_classes)
        
    def forward(self, x):
        x = nn.AvgPool2d(5, stride=3)(x)
        x = nn.ReLU()(self.conv(x))
        x = nn.Flatten()(x)
        x = nn.ReLU()(self.dense1(x))
        x = nn.Dropout(0.7)(x)
        x = self.dense2(x)
        return x

class GoogLeNet(nn.Module):
    
    def __init__(self, in_channels=3, n_classes=1000):
        super().__init__()
        
        channels = [192, 256, 480, 512, 512, 512, 528, 832, 832] + [1024]
        
        inception_params = [[64,  96,  128, 16, 32,  32],
                            [128, 128, 192, 32, 96,  64],
                            [192, 96,  208, 16, 48,  64],
                            [160, 112, 224, 24, 64,  64],
                            [128, 128, 256, 24, 64,  64],
                            [112, 144, 288, 32, 64,  64],
                            [256, 160, 320, 32, 128, 128],
                            [256, 160, 320, 32, 128, 128],
                            [384, 192, 384, 48, 128, 128]]
        
        self.initial_conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3)
        self.initial_conv2 = nn.Conv2d(64, 64, 1)
        self.initial_conv3 = nn.Conv2d(64, 192, 3, padding=1)
        
        self.inception = nn.ModuleList([InceptionModule(channels[i], *inception_params[i])
                                        for i in range(len(inception_params))])
        
        self.aux_classifier1 = InnerClassifier(512, n_classes)
        self.aux_classifier2 = InnerClassifier(528, n_classes)
        
        self.final_dense = nn.Linear(1*1*1024, n_classes)
    
    def forward(self, x):  # net input: Cx224x224
        outputs = 0
        
        x = nn.ReLU()(self.initial_conv1(x))
        x = nn.MaxPool2d(3, stride=2, ceil_mode=True)(x)
        x = nn.LocalResponseNorm(5)(x)
        x = nn.ReLU()(self.initial_conv2(x))
        x = nn.ReLU()(self.initial_conv3(x))
        x = nn.LocalResponseNorm(5)(x)
        x = nn.MaxPool2d(3, stride=2, ceil_mode=True)(x)
                
        for i, block in enumerate(self.inception):
            x = block(x)
            if i in [1, 6]:
                x = nn.MaxPool2d(3, stride=2, ceil_mode=True)(x)
            if self.training:
                if i==2:
                    outputs += 0.3 * self.aux_classifier1(x)
                if i==5:
                    outputs += 0.3 * self.aux_classifier2(x)
                    
        x = nn.AvgPool2d(kernel_size=7, stride=1, ceil_mode=True)(x)
        x = nn.Flatten()(x)
        x = nn.Dropout(0.4)(x)
        x = nn.Softmax(dim=1)(self.final_dense(x))
        outputs += x
        return outputs