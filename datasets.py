import pandas as pd
import string
import torch
import torch.utils.data as d

from utils import pad, hot_seq, string_vectoriser, double_binary,padding

class ENG_WUsL(d.Dataset):
    '''
    Implementa a class for the English Lexicon Project Datase from the Washington University in Saint Louis
    available at: https://elexicon.wustl.edu/query13/query13.html
    '''
    def __init__(self, max_lenght=False):
        super().__init__()
        self.max_lenght = max_lenght
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = 'Datasets/ENG/WUsLData.csv'
        self.col_names = ["Word", "Length", "LgSUBTLCD", "Ortho_N", "Phono_N", "Freq_N", "Freq_N_P","Pron"]
        self.data = pd.read_csv(self.path, delimiter=',', na_values='#')
        self.data.dropna(axis=0,
                        subset=["Word","Length","I_Mean_RT","I_NMG_Mean_RT"],
                        how='any',
                        inplace=True)
        self.data.Word = self.data.Word.str.lower()
        if self.max_lenght:
            self.data = self.data[self.data.Length <= self.max_lenght]
        self.data.reset_index(inplace=True)
        # Dataset is ready

        self.len = len(self.data.index)

        self.word_len = self.data.Length.max()
        self.word_abc = hot_seq(self.data.Word.to_list())

        self.in_shape = len(self.word_abc), self.word_len
        self.out_shape =  2, 1  



    def __getitem__(self, index):
        word = self.data.Word.iloc[index]
        indexes = [self.word_abc[i] for i in word]
        x = torch.zeros(self.in_shape)
        for i, ind in enumerate(indexes):
            x[ind, i] = 1
        t_naming = self.data.I_Mean_RT.iloc[index] / 10**6
        t_decision = self.data.I_NMG_Mean_RT.iloc[index] / 10**6

        return x.float().to(self.device), torch.tensor((t_naming, t_decision)).to(self.device)

    def __len__(self):
        return self.len

class Test_ENG_WUsL(d.Dataset):
    """
    Test data of max_lenght = 5 based on the English Lexicon Project Datase from the Washington University in Saint Louis
    available at: https://elexicon.wustl.edu/query13/query13.html
    The data is a Random stratified sample, selected over a dummy strata variable encoding frequency(Q1-2,Q3-4) x
    orthographic neighbourhood(Q1-2,Q3-4), phonologic neigbourhood(Q1-2,Q3-4)
    so that:
    | binary | decimal | meaning         |
    |--------|---------|-----------------|
    | 000    | 0       | L_F, S_ON, S_PN |
    | 001    | 1       | H_F, S_ON, S_PN |
    | 010    | 2       | L_F, D_ON, S_PN |
    | 011    | 3       | H_F, D_ON, S_PN |
    | 100    | 4       | L_F, S_ON, D_PN |
    | 101    | 5       | H_F, S_ON, D_PN |
    | 110    | 6       | L_F, D_ON, D_PN |
    | 111    | 7       | H_F, D_ON, D_PN |
    The dummy variable is saved in the "Cat" column
    Frequency wise, all cathegories have similar frequency, except 7 which is several times the others

    """
    
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = 'Datasets/ENG/Test_set_r5.csv'
        self.col_names = ["ID","Word", "Length", "LgSUBTLCD", "Ortho_N", "Phono_N", "Freq_N_PH",
                            "Freq_N_OGH","BG_Sum","I_Mean_RT", "I_SD", "I_NMG_Mean_RT", "I_NMG_SD", "Cat"]
        self.data = pd.read_csv(self.path, delimiter=',', na_values='#')
        self.data.dropna(axis=0,
                        subset=["Word","Length","I_Mean_RT","I_NMG_Mean_RT"],
                        how='any',
                        inplace=True)
        self.data.reset_index(inplace=True)
        # Dataset is ready
        self.len = len(self.data.index)
        
        self.word_len = self.data.Length.max()
        self.word_abc = hot_seq(self.data.Word.to_list())
        self.word_abc["'"] = 26

        self.in_shape = len(self.word_abc), self.word_len
        self.out_shape = 2, 1

    def __getitem__(self, index):
        word = self.data.Word.iloc[index]
        indexes = [self.word_abc[i] for i in word]
        x = torch.zeros(self.in_shape)
        for i, ind in enumerate(indexes):
            x[ind, i] = 1
        t_naming = self.data.I_Mean_RT.iloc[index] / 10**3
        t_decision = self.data.I_NMG_Mean_RT.iloc[index] / 10**3

        return x.float().to(self.device), torch.tensor((t_naming, t_decision)).to(self.device)

    def __len__(self):
        return self.len

if __name__ == "__main__":
    data = Test_ENG_WUsL()
    print(data[100])