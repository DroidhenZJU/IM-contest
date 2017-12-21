from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
import math
import pandas as pd
import numpy as np
import csv
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
# from pre_process import Reader
# from sklearn import cross_validation, metrics
# from sklearn.grid_search import GridSearchCV


class Predictor(object):
    
    def __init__(self, data_file):
        self.train_data = pd.read_excel(data_file, sheet_name = 0)
        self.X_train = self.train_data.values[:, 0:-1]
        self.Y_train = self.train_data.values[:,-1]
        self.test_data = pd.read_excel(data_file, sheet_name = 1)
        self.test_index = self.test_data.index
        self.X_test = self.test_data.values[:,:]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_train, self.Y_train, random_state = 1026, test_size=0.3)
        

        # print(self.X_train.shape)
        # print(self.X_test.shape)
        # print(self.Y_train.shape)
        print(self.X_test[0])
        print(self.X_train[0])
        #self.pipe = Pipeline([('feature_select', SelectKBest(lambda X, Y: np.array(list(map(lambda x:pearsonr(x, Y)[0], X.T))).T, k=6000)),('clf', GradientBoostingRegressor(max_depth = 5))])

        self.pipe = Pipeline([('feature_select', SelectKBest(lambda X, Y: np.array(list(map(lambda x:pearsonr(x, Y)[0], X.T))).T, k=3000)),('clf', XGBRegressor(max_depth = 2))])



        
    def train(self):
        self.pipe.fit(self.x_train, self.y_train)
        y_predict = self.pipe.predict(self.x_test)
        mse = 0
        mse = ((y_predict - self.y_test)**2).mean()
        print("mse: %.4f" %(mse))


    def train_Xgboost(self):

        param_grid = {
            'subsample': [0.2, 0.4, 0.6, 0.8],
            'n_estimators':[100, 200, 400],
            'colsample_bytree': [0.2, 0.4, 0.6, 0.8],
            'max_depth': [2,4,6,8,10],
        }
        model = GridSearchCV(estimator = XGBRegressor(learning_rate = 0.1), param_grid = param_grid, n_jobs = 2, cv=10, verbose=20, scoring = "neg_mean_squared_error")
        model.fit(self.x_train, self.y_train)
        print(model.best_params_, model.best_score_)

        # dtrain = xgb.DMatrix(self.x_train, self.y_train)
        # deval = xgb.DMatrix(self.x_test, self.y_test)
        # watchlist = [(deval, 'eval')]
        # params = {
        #     'num_boost_round ':1000,
        #     'booster': 'gbtree',
        #     'objective': 'reg:linear',
        #     'subsample': 0.8,
        #     'colsample_bytree': 0.85,
        #     'eta': 0.01,
        #     'max_depth': 7,
        #     'seed': 2016,
        #     'silent': 0,
        #     'eval_metric': 'rmse'
        # }
        # self.clf_xgb = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds = 50)

    
    def predict(self):
        
        self.pipe.fit(self.X_train, self.Y_train)
        y_predict = self.pipe.predict(self.X_test)


        result = np.vstack((self.test_index, y_predict)).T        
        with open('测试A答案.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(result)

class Xgboost_Predictor(object):
    def __init__(self, data_file):
        self.train_data = pd.read_excel(data_file, sheet_name = 0)
        self.X_train = self.train_data.values[:, 0:-1]
        self.Y_train = self.train_data.values[:,-1]
        self.test_data = pd.read_excel(data_file, sheet_name = 1)
        self.test_index = self.test_data.index
        self.X_test = self.test_data.values[:,:]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_train, self.Y_train, random_state = 1026, test_size=0.3)
        
    def parameter_search(self):

        param_grid = {
            'subsample': [0.2, 0.4, 0.6, 0.8],
            'n_estimators':[100, 200, 400],
            'colsample_bytree': [0.2, 0.4, 0.6, 0.8],
            'max_depth': [2,4,6,8,10],
        }
        model = GridSearchCV(estimator = XGBRegressor(learning_rate = 0.1), param_grid = param_grid, n_jobs = 1, cv=5, verbose=20, scoring = "neg_mean_squared_error")
        model.fit(self.X_train, self.Y_train)
        print(model.best_params_, model.best_score_)

    def cv_train(self):
        dtrain = xgb.DMatrix(self.x_train, self.y_train)
        deval = xgb.DMatrix(self.x_test, self.y_test)
        watchlist = [(deval, 'eval')]
        params = {
            'num_boost_round ':1000,
            'booster': 'gbtree',
            'objective': 'reg:linear',
            'subsample': 0.8,
            'colsample_bytree': 0.85,
            'eta': 0.01,
            'max_depth': 7,
            'seed': 2016,
            'silent': 0,
            'eval_metric': 'rmse'
        }
        cv_reg = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds = 50)

    def predict(self):
        dtrain = xgb.DMatrix(self.X_train, self.Y_train)
        # deval = xgb.DMatrix(self.x_test, self.y_test)
        # watchlist = [(deval, 'eval')]
        params = {
            'num_boost_round ':1000,
            'booster': 'gbtree',
            'objective': 'reg:linear',
            'subsample': 0.8,
            'colsample_bytree': 0.85,
            'eta': 0.01,
            'max_depth': 7,
            'seed': 2016,
            'silent': 0,
            'eval_metric': 'rmse'
        }
        reg = xgb.train(params, dtrain, 500, early_stopping_rounds = 50)
        y_predict = reg.predict(self.X_test)

        result = np.vstack((self.test_index, y_predict)).T      
        with open('测试A答案.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(result)

if __name__ == "__main__":
    p = Xgboost_Predictor("after_pre_process.xlsx")
    p.parameter_search()
