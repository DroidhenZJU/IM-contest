import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import config
import os

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
            # if col == '210X24':
            #     print(x_set_list[0] / 10000000000)
            #     exit()
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

            if len(x_set_list) >= 2 and not all([True if str(n) == "nan" else False for n in x]):
                # if col == "210X24":
                #     print(x_set_list[:20])
                #     print(str(x_set_list[0]))
                #     print([str(e).startswith("2017") or str(e).startswith("2016") for e in x_set_list[:20]])
                if col == "520X171":
                    self.dropped_features.append(col)
                elif not all([str(e).startswith("2017") or str(e).startswith("2016") for e in x_set_list[:20]]):
                    self.selected_features.append(col)
                else:
                    self.dropped_features.append(col)
            else:
                self.dropped_features.append(col)
        self.train_data = self.train_data.loc[:, self.selected_features + ['Y']]
        self.test_data = self.test_data.loc[:, self.selected_features]
                    
    def coef_selection(self, k = 2000):
        corr_values = []

        for col in self.test_data.columns:
            corr_values.append(abs(pearsonr(self.train_data[col].values,self.train_data['Y'])[0]))
        corr_df = pd.DataFrame({'col':self.test_data.columns, 'corr_value':corr_values})
        corr_df = corr_df.sort_values(by='corr_value',ascending=False)
        selected = corr_df['col'].values[:k]

        self.train_data = self.train_data.loc[:, list(selected) + ['Y']]
        self.test_data = self.test_data.loc[:, list(selected)]
        # print(self.train_data)
        # print(self.test_data)

    def tree_selection(self, k = 2000):
        X_train = self.train_data.values[:, 0:-1]
        Y_train = self.train_data.values[:,-1]
        X_test = self.test_data.values[:,:]
        # x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, random_state = 1024, test_size=0.1)

        # reg = GradientBoostingRegressor(random_state = 0,
        #                                 learning_rate = 0.1,
        #                                 n_estimators= 29,
        #                                 min_samples_split=2,
        #                                 min_samples_leaf=5,
        #                                 max_features=0.81,
        #                                 subsample= 0.8,
        #                                 max_depth= 6,
        # )
        reg = XGBRegressor(random_state = 0,
                            learning_rate = 0.1,
                            n_estimators= 49,
                            subsample= 0.78,
                            colsample_bytree= 0.62,
                            max_depth= 3,
        )
        reg.fit(X_train, Y_train)
        print("---GBDT feature importance: %dth---" %k)
        print(sorted(reg.feature_importances_, reverse=True)[:k])
        importance_df = pd.DataFrame({'col':self.test_data.columns, 'importance':reg.feature_importances_})
        importance_df = importance_df.sort_values(by="importance", ascending=False)
        selected = importance_df['col'].values[:k]
        self.train_data = self.train_data.loc[:, list(selected) + ['Y']]
        self.test_data = self.test_data.loc[:, list(selected)]


    def save_to_file(self, after_file):
        with pd.ExcelWriter(after_file) as writer:
            self.train_data.to_excel(writer,sheet_name = "train_data")
            self.test_data.to_excel(writer, sheet_name = "test_data")
            # self.deleted_feature.to_excel(writer, sheet_name = "deleted")

        return self.train_data

    def pre_process(self, after_file):
        self.fill_empty()
        self.delete_duplicate()
        # self.coef_selection(k = 2500)
        self.tree_selection(300)
        self.save_to_file(after_file)

        


if __name__ == "__main__":
    trainfile_name = os.path.join(config.base_path,"data","训练.xlsx")
    testfile_name = os.path.join(config.base_path,"data","测试A.xlsx")

    after_file = os.path.join(config.base_path,"data","tree_feature_selected_A_300.xlsx")
    r = Reader(trainfile_name, testfile_name)
    r.pre_process(after_file)
    
        