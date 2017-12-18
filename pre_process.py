import pandas as pd
import numpy as np

class Reader(object):
    '''
        数据读取和预处理类
    '''

    def __init__(self, filename):
        self.df = pd.read_excel(filename)
        self.data = self.df.iloc[:,1:]
        self.data.index = self.df.iloc[:,0].values

    def fill_empty(self):
        mean_col = self.data.mean()
        self.data = self.data.fillna(mean_col)

    def delete_duplicate(self):
        


    def pre_process(self):
        fill_empty()
        delete_duplicate()



if __name__ == "__main__":
    filename = "训练.xlsx"
    r = Reader(filename)
    r.pre_process()
    
    
        