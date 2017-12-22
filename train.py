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
import config
from imp import reload  
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

        self.pipe = Pipeline([('feature_select', SelectKBest(lambda X, Y: np.array(list(map(lambda x:abs(pearsonr(x, Y)[0]), X.T))).T, k=3000)),('clf', XGBRegressor(max_depth = 2))])



        
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


        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_train, self.Y_train, random_state = 1024, test_size=0.1)
        # print(self.x_train.shape)
        # print(self.x_test.shape)
        # print(self.y_train.shape)
        # print(self.y_test.shape)
        # print(self.test_index.shape)
        
    def parameter_search(self):

        param_grid = {
            'subsample': [0.8],
            'colsample_bytree': [0.65, 0.7, 0.75],
            'max_depth': [3],
        }
        model = GridSearchCV(estimator = XGBRegressor(learning_rate = 0.05, random_state = 0, n_estimators = 500), param_grid = param_grid, n_jobs = 1, cv=5, verbose=20, scoring = "neg_mean_squared_error")
        model.fit(self.X_train, self.Y_train)
        print(model.best_params_, model.best_score_)

    def cv_train(self):
        dtrain = xgb.DMatrix(self.x_train, self.y_train)
        deval = xgb.DMatrix(self.x_test, self.y_test)
        watchlist = [(deval, 'eval')]
        # params = {
        #     'num_boost_round ':500,
        #     'booster': 'gbtree',
        #     'objective': 'reg:linear',
        #     'subsample': 0.8,
        #     'colsample_bytree': 0.6,
        #     'eta': 0.1,
        #     'max_depth': 2,
        #     'seed': 0,
        #     'silent': 0,
        #     'eval_metric': 'rmse'
        # }

        while True:
            params = config.Xgboost_config
            cv_reg = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds = 50)
            # cv_reg = xgb.cv(params, dtrain, 500, nfold = 5, metrics = "rmse")
            print(cv_reg)
            str = input("Modify the parameter: ")
            if str ==  's':
                break
            else:
                reload(config)



    def predict(self):
        dtrain = xgb.DMatrix(self.x_train, self.y_train)
        deval = xgb.DMatrix(self.x_test, self.y_test)
        watchlist = [(deval, 'eval')]
        params = {
            'num_boost_round ':500,
            'booster': 'gbtree',
            'objective': 'reg:linear',
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'eta': 0.1,
            'max_depth': 3,
            'seed': 0,
            'silent': 0,
            'eval_metric': 'rmse'
        }
        reg = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds = 50)
        y_predict = reg.predict(xgb.DMatrix(self.X_test),ntree_limit = reg.best_ntree_limit)

        # reg = XGBRegressor(learning_rate = 0.1, subsample = 0.8, colsample_bytree=0.6, max_depth = 2, n_estimators=100)
        # reg.fit(self.X_train, self.Y_train)
        # y_predict = reg.predict(self.X_test)
        result = np.vstack((self.test_index, y_predict)).T      
        with open('answerA.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(result)

class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(1)
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred

if __name__ == "__main__":
    p = Xgboost_Predictor("after_pre_process_A.xlsx")
    # p.parameter_search()
    p.cv_train()
    # p.predict()
