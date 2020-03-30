import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import conv_dims, conv1d_len

'''
General guideline:
* Input is thought as a Matrix of dimensions (batch_size, in_char_n, in_word_len)
* Output is thought as a Matrix of dimensions (batch_size, in_char_n, in_word_len + 1 (reading_time))
'''

class NConv(nn.Module):
    '''
    Module takes a tensor of shape (batch_size, channels, lenght)
    Applies a 1D convolution with a kernel of varying size from 1 up to lenght
    '''
    def __init__(self, input_shape):
        super(NConv, self).__init__()
        self.in_char_n = input_shape[0]
        self.in_word_len = input_shape[1]
        self.out_len = conv_dims(self.in_word_len)

        self.convs=nn.ModuleList([nn.Conv1d(in_channels=self.in_char_n,
                                            out_channels=i,
                                            kernel_size=i,
                                            stride=1)
                                    for i in range(1,self.in_word_len+1)])

    def forward(self, x):
        self.batch_size = x.shape[0]
        out=[]
        for i, conv in enumerate(self.convs):
            out.append(conv(x).view(self.batch_size,-1).unsqueeze(1))
        return torch.cat(out, dim=2)


class CLRM(NConv, nn.Module):
    '''
    Creates a Conv-Deconv system, extendign N-Conv taking (batch_size, channels, lenght)
    and returning a tensor of the shape specified in labels
    '''
    def __init__(self, input_shape, label_shape):
        super(CLRM, self).__init__(input_shape)

        assert len(label_shape) == 3

        self.channels = label_shape[0]
        self.n_measure = label_shape[1]
        self.bin_measure_lenght = label_shape[2]

        self.convs = NConv(input_shape)
        self.conv2 = nn.Conv1d(1, 2, 5)
        self.lin1 = nn.Linear(conv1d_len(self.out_len, 5), self.out_len)
        self.lin2 = nn.Linear(self.out_len, self.n_measure * self.bin_measure_lenght) 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.convs(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = x.view(-1, self.channels, self.n_measure, self.bin_measure_lenght)
        return F.softmax(x, dim=1)



if __name__ == "__main__":
    T = torch.rand(100,26,8)
    model = CLRM((26,8), (2,2,22))
    print(model(T))