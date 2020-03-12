import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import conv_dims

'''
General guideline:
* Input is thought as a Matrix of dimensions (batch_size, in_char_n, in_word_len)
* Output is thought as a Matrix of dimensions (batch_size, in_char_n, in_word_len + 2 (lexical decision, naming time))
'''

class NConv(nn.Module):
    '''
    Module takes a tensor of shape (batch_size, channels, lenght)
    Applies a 1D convolution with a kernel of varying size from 1 up to lenght
    '''
    def __init__(self, input_shape):
        super(NConv, self).__init__()
        # extrats dimensions form the given shape
        self.in_char_n = input_shape[0]
        self.in_word_len = input_shape[1]

        # calculates the final lenght after the convolution (Gauss' formula)
        self.out_len = conv_dims(self.in_word_len)

        # creates ta list of convolutions augmenting kernel size each time
        self.convs=nn.ModuleList([nn.Conv1d(in_channels=self.in_char_n,
                                            out_channels=i,
                                            kernel_size=i,
                                            stride=1)
                                    for i in range(1,self.in_word_len+1)])

    def forward(self, x):
        # extracts batch size from input
        self.batch_size = x.shape[0]
        # passes through each elment in the convolution list and passes the data
        out=[]
        for i, conv in enumerate(self.convs):
            out.append(conv(x).view(self.batch_size,-1).unsqueeze(1))
        # returns the concatenated output
        return torch.cat(out, dim=2)


class CLRM(NConv, nn.Module):
    '''
    Creates a Conv-Deconv system, extendign N-Conv taking (batch_size, channels, lenght)
    and returning a tensor of the shape specified in labels
    '''
    def __init__(self, input_shape, label_shape):
        super(CLRM, self).__init__(input_shape)
        # extrats dimensions form the given shape
        self.out_phon_n = label_shape[0]
        self.out_pron_len = label_shape[1]

        # Sets up the NConvolutiona layer
        # 2 dense layers
        # A transposed convolution
        self.convs = NConv(input_shape)
        self.lin1 = nn.Linear(self.out_len, self.out_len//4)
        self.lin2 = nn.Linear(self.out_len//4 ,self.out_pron_len)
        self.deconv = nn.ConvTranspose1d(in_channels=1,
                                            out_channels=self.out_phon_n,
                                            kernel_size=1)
    def forward(self, x):
        # passes data throug the layers and 
        x = torch.tanh(self.convs(x))
        x = torch.tanh(self.lin1(x))
        x = torch.tanh(self.lin2(x))
        # hardshrink with a very high threshold is used to zero out noise
        return F.hardshrink(self.deconv(x), lambd=0.7)