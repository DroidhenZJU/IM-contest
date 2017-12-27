from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
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
from sklearn.metrics import mean_squared_error, make_scorer

# from pre_process import Reader
# from sklearn import cross_validation, metrics
# from sklearn.grid_search import GridSearchCV

def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5

RMSE = make_scorer(mean_squared_error_, greater_is_better=False)

class GBDT_Predictor(object):
    
    def __init__(self, data_file):
        self.train_data = pd.read_excel(data_file, sheet_name = 0)
        self.X_train = self.train_data.values[:, 0:-1]
        self.Y_train = self.train_data.values[:,-1]
        self.test_data = pd.read_excel(data_file, sheet_name = 1)
        self.test_index = self.test_data.index
        self.X_test = self.test_data.values[:,:]


        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_train, self.Y_train, random_state = 1024, test_size=0.1)

    def parameter_search(self):

        param_grid = {
            'min_samples_split':[2],
            'min_samples_leaf':[6],
            'max_features':[0.8],
            'subsample': [0.8],
            'max_depth': [6],
            'n_estimators': [500]
        }
        model = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate = 0.1, random_state = 0), param_grid = param_grid, n_jobs = 2, cv=5, verbose=20, scoring = RMSE)
        model.fit(self.X_train, self.Y_train)
        print(model.best_params_, model.best_score_)

    
    def eval(self):
        reg = GradientBoostingRegressor(learning_rate = 0.1, 
                                        random_state = 0, 
                                        n_estimators=500, 
                                        min_samples_split = 2,
                                        min_samples_leaf = 6,
                                        max_features = 0.8,
                                        subsample = 0.8,
                                        max_depth = 6)

        reg.fit(self.x_train, self.y_train)
        y_predict = reg.predict(self.x_test)
        mse = ((y_predict - self.y_test)**2).mean()
        print("mse: %.4f" %(mse))

    def predict(self):
        
        self.pipe.fit(self.X_train, self.Y_train)
        y_predict = self.pipe.predict(self.X_test)


        result = np.vstack((self.test_index, y_predict)).T        
        with open('测试A答案.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(result)

class RF_Predictor(object):
    
    def __init__(self, data_file):
        self.train_data = pd.read_excel(data_file, sheet_name = 0)
        self.X_train = self.train_data.values[:, 0:-1]
        self.Y_train = self.train_data.values[:,-1]
        self.test_data = pd.read_excel(data_file, sheet_name = 1)
        self.test_index = self.test_data.index
        self.X_test = self.test_data.values[:,:]


        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_train, self.Y_train, random_state = 1024, test_size=0.1)

    def parameter_search(self):

        param_grid = {
            'min_samples_split':[2, 4, 6],
            'min_samples_leaf':[1, 2, 4, 6],
            'max_features':["sqrt", 0.5, 0.6, 0.8, 0.9],
            'max_depth': [2, 4, 6, 8],
        }
        model = GridSearchCV(estimator = RandomForestRegressor(learning_rate = 0.1, random_state = 0, n_estimators = 500), param_grid = param_grid, n_jobs = 2, cv=5, verbose=20, scoring = "neg_mean_squared_error")
        model.fit(self.X_train, self.Y_train)
        print(model.best_params_, model.best_score_)

    
    def eval(self):
        reg = RandomForestRegressor(random_state = 0, 
                                    n_estimators=500, 
                                    min_samples_split = 2,
                                    min_samples_leaf = 2,
                                    max_features = 0.8,
                                    max_depth = 8)
                                        
        reg.fit(self.x_train, self.y_train)
        y_predict = reg.predict(self.x_test)
        mse = ((y_predict - self.y_test)**2).mean()
        print("mse: %.4f" %(mse))

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
        self.answer_file = "answer_" + data_file.split('.')[0] + ".csv"

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_train, self.Y_train, random_state = 1024, test_size=0.1)
        # print(self.x_train.shape)
        # print(self.x_test.shape)
        # print(self.y_train.shape)
        # print(self.y_test.shape)
        # print(self.test_index.shape)

    def feature_select(self):
        SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)
        
    def parameter_search(self):

        param_grid = {
            'subsample': [0.8],
            'colsample_bytree': [0.65, 0.7, 0.75],
            'max_depth': [3],
        }
        model = GridSearchCV(estimator = XGBRegressor(learning_rate = 0.05, random_state = 0, n_estimators = 500), param_grid = param_grid, n_jobs = 1, cv=5, verbose=20, scoring = "neg_mean_squared_error")
        model.fit(self.X_train, self.Y_train)
        print(model.best_params_, model.best_score_)

    def eval(self):
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
            #cv_reg = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds = 50)
            reg = XGBRegressor(learning_rate = 0.1, subsample = 0.8, colsample_bytree=0.6, max_depth = 3, n_estimators=100)
            reg.fit(self.x_train, self.y_train)
            y_predict = reg.predict(self.x_test)
            print(((y_predict - self.y_test)**2).mean())

            # cv_reg = xgb.cv(params, dtrain, 500, nfold = 5, metrics = "rmse")
            # print(cv_reg)
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
            'colsample_bytree': 0.6,
            'eta': 0.1,
            'max_depth': 3,
            'seed': 0,
            'silent': 0,
            'eval_metric': 'rmse'
        }
        reg = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds = 50)
        y_predict = reg.predict(xgb.DMatrix(self.X_test),ntree_limit = reg.best_ntree_limit)

        # reg = XGBRegressor(learning_rate = 0.1, subsample = 0.8, colsample_bytree=0.6, max_depth = 3, n_estimators=100)
        # reg.fit(self.x_train, self.y_train)
        # y_predict = reg.predict(self.x_test)
        # print(((y_predict - self.y_test)**2).mean())
        # y_predict= reg.predict(self.X_test)

        result = np.vstack((self.test_index, y_predict)).T      
        with open(self.answer_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(result)


if __name__ == "__main__":
    # p = Xgboost_Predictor("after_pre_process_A.xlsx")
    # # p.parameter_search()
    # # p.cv_train()
    # p.predict()

    p = Xgboost_Predictor("feature_selected_A_3000.xlsx")
    # p.eval()
    # p.cv_train()
    p.predict()
    
