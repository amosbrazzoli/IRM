import pandas as pd

import torch
import torch.utils.data as d

from utils import pad, hot_seq, string_vectoriser, double_binary,padding

BIN_LOG = 22

class ENG_WUsL(d.Dataset):
    '''
    Implementa a class for the English Lexicon Project Datase from the Washington University in Saint Louis
    available at: https://elexicon.wustl.edu/query13/query13.html
    '''
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = 'Datasets/ENG/WUsLData.csv'
        self.col_names = ["Word", "Length", "Freq_HAL", "Ortho_N", "Phono_N", "Freq_N", "Freq_N_P","Pron"]
        self.data = pd.read_csv(self.path, delimiter=',', na_values='#')
        self.data.dropna(axis=0,
                        subset=["Word","I_Mean_RT","I_NMG_Mean_RT"],
                        how='any',
                        inplace=True)
        self.data.reset_index(inplace=True)
        # Dataset is ready

        self.len = len(self.data.index)

        self.word_len = self.data.Length.max()
        self.word_abc = hot_seq(self.data.Word.to_list())

        self.in_shape = len(self.word_abc), self.word_len
        self.out_shape = 2, 2, BIN_LOG # 24 is ceil(Bin_Log(max(times))), numbers



    def __getitem__(self, index):

        x = self.data.Word.iloc[index]
        x = string_vectoriser(pad(x, self.word_len), self.word_abc)


        t_decision = padding(double_binary(self.data.I_Mean_RT.iloc[index]),BIN_LOG).unsqueeze(0)
        t_naming = padding(double_binary(self.data.I_NMG_Mean_RT.iloc[index]), BIN_LOG).unsqueeze(0)

        y_def = torch.cat((t_naming, t_decision), dim=0).float()
        return x.float().to(self.device), y_def.float().to(self.device)

    def __len__(self):
        return self.len

if __name__ == "__main__":
    data = ENG_WUsL()
    for _, i in data:
        break