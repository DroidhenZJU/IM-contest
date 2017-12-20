from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import math
import pandas as pd
import numpy as np
# from pre_process import Reader
# from sklearn import cross_validation, metrics
# from sklearn.grid_search import GridSearchCV


class Predictor(object):
    
    def __init__(self, data_file):
        self.train_data = pd.read_excel(data_file, sheet_name = 0)
        self.X_train = self.train_data.values[:, 0:-1]
        self.Y_train = self.train_data.values[:,-1]
        self.test_data = pd.read_excel(data_file, sheet_name = 1)
        self.X_test = self.test_data.values[:,:]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_train, self.Y_train, test_size=0.3)
        

        # print(self.X_train.shape)
        # print(self.X_test.shape)
        # print(self.Y_train.shape)
        print(self.X_test[0])
        print(self.X_train[0])
        
        self.pipe = Pipeline([('feature_select', SelectKBest(lambda X, Y: np.array(list(map(lambda x:pearsonr(x, Y)[0], X.T))).T, k=6000)),('clf', GradientBoostingRegressor(max_depth = 5))])

    
    def train(self):
        self.pipe.fit(self.x_train, self.y_train)
        y_predict = self.pipe.predict(self.x_test)
        print(y_predict)
        print(self.y_test)
        print((y_predict - self.y_test)**2)
        mse = 0
        mse = ((y_predict - self.y_test)**2).mean()
        print("mse: %.4f" %(mse))

if __name__ == "__main__":
    p = Predictor("after_pre_process.xlsx")
    p.train()