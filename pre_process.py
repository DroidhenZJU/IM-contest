import pandas as pd
import numpy as np

class Reader(object):
    '''
        数据读取和预处理类
    '''

    def __init__(self, train_filename, test_filename):
        self.train_df = pd.read_excel(train_filename)
        self.test_df = pd.read_excel(test_filename)
        self.train_data = self.train_df.iloc[:,1:]
        self.train_data.index = self.train_df.iloc[:,0].values
        self.test_data = self.test_df.iloc[:,1:]
        self.test_data.index = self.test_df.iloc[:,0].values
        self.selected_features = [] #保留的特征
        self.dropped_features = [] #删除的特征

    def fill_empty(self):
        mean_col = self.train_data.mean()
        self.train_data = self.train_data.fillna(mean_col)
        mean_col = self.test_data.mean()
        self.test_data = self.test_data.fillna(mean_col)

    def delete_duplicate(self):
        for col in self.test_data.columns:
            x = np.hstack((self.train_data[col].values, self.test_data[col].values))
            x_set_list = list(set(x))
            if isinstance(x_set_list[0], str):
                encode_dict = {}
                for i in range(len(x_set_list)):
                    encode_dict[x_set_list[i]] = i
                c_train = self.train_data[col]
                for i in range(len(c_train)):
                    c_train.iat[i] = encode_dict[c_train.iat[i]]
                c_test = self.test_data[col]
                for i in range(len(c_test)):
                    c_test.iat[i] = encode_dict[c_test.iat[i]]
            x = self.train_data[col].values
            x_set_list = list(set(x))
            if len(x_set_list) >= 2 and not all([True if str(n) == "nan" else False for n in x]) and int(x_set_list[0] / 10000000000) != 2017 and int(x_set_list[0] / 1000000000000) != 2017 and int(x_set_list[0] / 10000) != 2017:
                self.selected_features.append(col)
            else:
                self.dropped_features.append(col)
        self.train_data = self.train_data.loc[:, self.selected_features + ['Y']]
        self.test_data = self.test_data.loc[:, self.selected_features]
                    

    def pre_process(self, after_file):
        self.fill_empty()
        self.delete_duplicate()
        with pd.ExcelWriter(after_file) as writer:
            self.train_data.to_excel(writer,sheet_name = "train_data")
            self.test_data.to_excel(writer, sheet_name = "test_data")
            # self.deleted_feature.to_excel(writer, sheet_name = "deleted")

        return self.train_data



if __name__ == "__main__":
    trainfile_name = "训练.xlsx"
    testfile_name = "测试A.xlsx"

    after_file = "after_pre_process_A.xlsx"
    r = Reader(trainfile_name, testfile_name)
    r.pre_process(after_file)
    
        