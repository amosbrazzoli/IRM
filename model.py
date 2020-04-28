import torch
import torch.nn as nn

from utils import conv_dims, exp_corr

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
        self.channels = input_shape[0]
        self.lenght = input_shape[1]
        
        self.pool = nn.MaxPool1d(kernel_size=self.channels)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.channels,
                                                out_channels=exp_corr(self.channels, self.channels, i),
                                                kernel_size=i,
                                                padding=0,
                                                stride=1,
                                                dilation=1)for i in range(1,self.lenght+1)])

        self.out_len = sum([(self.lenght - k +1)*exp_corr(self.channels, self.channels, k) for k in range(1, self.lenght+1)])//self.channels
        

    def forward(self, x):
        self.batch_size = x.shape[0]
        out = []
        for conv in self.convs:
            out.append(self.pool(conv(x).permute(0, 2, 1)).reshape(self.batch_size, -1))
        return torch.cat(out, 1)

class CLRM(NConv, nn.Module):
    '''
    Creates a Conv-Deconv system, extendign N-Conv taking (batch_size, channels, lenght)
    and returning a tensor of the shape specified in labels
    '''
    def __init__(self, input_shape, label_shape):
        super(CLRM, self).__init__(input_shape)


        self.channels = label_shape[0]
        self.n_measure = label_shape[1]
        self.out_shape = label_shape[0] * label_shape[1]

        self.convs = NConv(input_shape)
        self.lin1 = nn.Linear(self.out_len, self.out_len*4)
        self.lin2 = nn.Linear(self.out_len*4, self.out_len*2)
        self.lin3 = nn.Linear(self.out_len*2, self.out_shape) 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        self.batch_size = x.shape[0]
        #print(x.shape)
        x = self.relu(self.convs(x))
        #print(x.shape)
        x = self.relu(self.lin1(x))
        #print(x.shape)
        x = self.relu(self.lin2(x))
        #print(x.shape)
        x = self.relu(self.lin3(x))
        #print(x.shape)
        return x

if __name__ == "__main__":
    T = torch.rand(1, 26, 8)
    model = CLRM((26, 8),(21,1))
    out = model(T)
    print(out)