import torch
from random import random


def conv_dims(lenght):
    '''
    Calculates the number of nodes after the n-convs
    $$ \sum_{i=0}^\lambda i(\lambda-i+1) $$
    '''
    out = 0
    for i in range(0, lenght+1):
        out += (i*(lenght-i+1))
    return out

def pad(word, max_len, pad_char='_'):
    # NOTE: padd on one side and cut max lenght to improve pron performance
    '''
    Pads a word with '_' up to max_len on both sides
    if odd chooses randomly
    '''
    right = random() > 0.5
    while len(word) < max_len:
        if right:
            word = ''.join([word, pad_char])
            right = False
        else:
            word = ''.join([pad_char, word])
            right = True
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
    return ''.join(lexicon)

def string_vectoriser(string, alfab):
    '''
    Uses an alpphabet string to convert a string
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



def binary_column_to_natural(column):
    '''
    turns binary column into a natural number
    '''
    column = column.tolist()
    c = 0
    for i, value in enumerate(column[::-1]):
        c += value * 2**i 
    return c


def word_from_onehot(matrix, alphabet):
    matrix = matrix.argmax(dim=0).tolist()
    return ''.join([alphabet[i] for i in matrix])

import torch
print(binary_column_to_natural(torch.tensor([0,1,1])))
print(word_from_onehot(torch.tensor([[1,0,0],[0,0,1],[0,1,0]]),"abc"))