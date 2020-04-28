import math
import torch
import torch.nn.functional as F
from random import random

def conv1d_dims(lenght_in, kernel_size, padding=0, dilation=0, stride=1):
    return (((lenght_in+2)*(padding - dilation)*(kernel_size-1))-1)//(stride)

def conv_dims(lenght):
    '''
    Calculates the number of nodes after the n-convs

    $$ \sum_{i=0}^\lambda i(\lambda-i+1) $$
    '''
    out = 0
    for i in range(0, lenght+1):
        out += (i*(lenght-i+1))
    return out

def pad(word, max_len, symmetric=False, pad_char='_'):
    '''
    Pads a word with '_' up to max_len on both sides
    if odd chooses randomly
    '''
    if symmetric:
        right = random() > 0.5
        while len(word) < max_len:
            if right:
                word = ''.join([word, pad_char])
                right = False
            else:
                word = ''.join([pad_char, word])
                right = True
    else:
        word = ''.join([pad_char*(max_len - len(word)), word])
    return word

def hot_seq(string_roll):
    '''
    Takes a list of strings and returns the set string of all characters
    '''
    lexicon = set()
    counter = 0
    for word in string_roll:
        for l in list(word):
            if l not in lexicon:
                lexicon.add(l)
    lexicon = sorted(lexicon)
    return {char: i for (i, char) in enumerate(lexicon)}

def string_vectoriser(string, alfab):
    '''
    Uses an alphabet string to convert a string
    in a one hot vector
    '''
    #alfab = f'{alfab}_'
    return torch.tensor([[0 if char != letter else 1 for letter in string]
                    for char in alfab])



def int_to_padd_bin(n, max_len):
    '''
    Converts a number_10 into a number_2 and pads it to lenght
    '''
    strBin = bin(int(n))[2:]
    lisBin = [int(i) for i in strBin]
    while len(lisBin) < max_len:
        lisBin.insert(0, 0)
    return torch.tensor(lisBin).unsqueeze_(1)


def opposite(binary_list):
    "returns the opposite of the binary list"
    return [ (int(i)+1)%2 for i in binary_list ]


def double_binary(numeral):
    "returns binary and it's oppositi in joint tensor form"
    numeral = int(numeral)
    return torch.tensor([int(i) for i in list(bin(numeral))[2:]])

def padding(tensor, max_shape):
    return F.pad(tensor, (max_shape-tensor.shape[-1], 0), "constant", 0.5)

def exp_corr(base, rounder, index):
    exp =  2.54196 - 3.34921/(449942*index**15.1505 + 1)**0.0603755
    subtractor = (base**exp) % rounder
    return int((base**exp) - subtractor)