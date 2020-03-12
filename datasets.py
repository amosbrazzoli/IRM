import pandas as pd

import torch
import torch.utils.data as d

from utils import pad, hot_seq, string_vectoriser, int_to_padd_bin

class ENG_WUsL(d.Dataset):
    '''
    Implementa a class for the English Lexicon Project Datase from the Washington University in Saint Louis
    available at: https://elexicon.wustl.edu/query13/query13.html
    '''
    def __init__(self, max_input):
        super().__init__()
        self.max_input = max_input
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = 'Datasets/ENG/WUsLData.csv'
        self.col_names = ["Word", "Length", "Freq_HAL", "Ortho_N", "Phono_N", "Freq_N", "Freq_N_P","Pron"]
        self.data = pd.read_csv(self.path, delimiter=',', na_values='#')
        self.data.Pron = self.data.Pron.replace('[""\.]','',regex=True)

        # Preprocessing data, cleaning for missing data
        self.data.dropna(axis=0,
                        subset=["Word","Pron","Freq_HAL","I_Mean_RT","I_NMG_Mean_RT"],
                        how='any',
                        inplace=True)
        # make a word lenght column
        self.data = self.data[self.data['Word'].map(len) <= max_input]
        # reset the index column
        self.data.reset_index(inplace=True)


        self.len = len(self.data.index)

        self.word_len = self.data.Length.max()
        self.word_abc = hot_seq(self.data.Word.to_list())

        self.pron_len = self.data.Pron.apply(len).max()
        self.pron_abc = hot_seq(self.data.Pron.to_list())


        self.in_shape = len(self.word_abc), self.word_len
        self.out_shape = len(self.pron_abc), self.pron_len + 2 # added for naming and lexical decision


    def __getitem__(self, index):
        # Gets data, padds it to shape, then turns it into a vector
        #padding character (default "_") is added if not present
        x = self.data.Word.iloc[index]
        x = string_vectoriser(pad(x, self.word_len), self.word_abc)

        y = self.data.Pron.iloc[index]
        y = string_vectoriser(pad(y, self.pron_len), self.pron_abc)

        # Gets numerical data, multiplies it by 1000 to remove decimal
        # then turns it in a one-side padded torch vector
        t_decision = int_to_padd_bin(float(self.data.I_Mean_RT.iloc[index].replace(',',''))*1000, self.out_shape[0])
        t_naming = int_to_padd_bin(float(self.data.I_NMG_Mean_RT.iloc[index].replace(',',''))*1000, self.out_shape[0])

        # Concateneates the label data in a single matrix
        y_def = torch.cat((y,t_naming, t_decision), dim=1).float()

        # Returns the input, label couple
        return x.float().to(self.device), y_def.float().to(self.device)

    def __len__(self):
        return self.len
