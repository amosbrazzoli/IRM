import torch
import torch.nn as nn

from utils import conv_dims

'''
General guideline:
* Input is thought as a Matrix of dimensions (batch_size, in_char_n, in_word_len)
* Output is thought as a Matrix of dimensions (batch_size, in_char_n, in_word_len + 1 (reading_time))
'''

class NConv(nn.Module):
    '''
    Module takes a tensor of shape (batch_size, channels, lenght)
    Applies a 1D convolution with a kernel ok varying size from 1 up to lenght
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

        self.out_phon_n = label_shape[0]
        self.out_pron_len = label_shape[1]

        self.convs = NConv(input_shape)
        self.lin1 = nn.Linear(self.out_len, self.out_len)
        self.lin2 = nn.Linear(self.out_len, self.out_phon_n * self.out_pron_len) 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.convs(x))
        x = self.relu(self.lin1(x))
        x = self.sigmoid(self.lin2(x))
        return x.view(-1, self.out_phon_n, self.out_pron_len)