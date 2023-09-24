from torch import nn
import torch.nn.functional as F

class Yournet1(nn.Module):
    ###################### Begin #########################
    # You can create your own network here or copy our reference model (LeNet5)
    # We will conduct a unified test on this network to calculate your score

    def __init__(self):
        super(Yournet1, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        #28x28
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 8, 3)
        # self.fc1 = nn.Linear(16 * 5 * 5, 99)# 5x5 image dimension
        self.drop = nn.Dropout(0.3)
        # self.fc2 = nn.Linear(99, 10)
        self.ft=nn.Flatten()
        self.fc=nn.Linear(8*5*5,10)


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x=self.ft(x)
        x= self.drop(x)
        x = self.fc(x)
        return x

    ######################  End  #########################