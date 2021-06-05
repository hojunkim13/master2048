import torch.nn as nn
import torch.nn.functional as F

class conv_layer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(conv_layer, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

        
    def forward(self, x):
        output = self.block(x)
        return output
        
# class residual_layer(nn.Module):
#     def __init__(self, channel):
#         super(residual_layer, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(channel, channel, 3, 1, 1),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(),
#             nn.Conv2d(channel, channel, 3, 1, 1),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(),
#         )
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         connect = self.block(x)
#         output = F.relu(connect + x)
#         return output


class Network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Network, self).__init__()    
        self.conv = nn.Sequential(
            conv_layer(state_dim[0], 32),
            conv_layer(32, 64),
            conv_layer(64, 128),
            conv_layer(128, 128),
        )
    
        self.policy = nn.Sequential(nn.Conv2d(128, 2, 1, 1, 0),
                                    nn.BatchNorm2d(2),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(4*4*2, action_dim),
                                    nn.Softmax(-1),
                                    )

        self.value = nn.Sequential(nn.Conv2d(128, 1, 1, 1, 0),
                                    nn.BatchNorm2d(1),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(4*4*1, 256),
                                    nn.ReLU(),                                    
                                    nn.Linear(256, 1),
                                    nn.Tanh(),
        )

        self.cuda()

    def forward(self, x):
        x = self.conv(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value
    