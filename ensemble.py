import time
start_time = time.time()
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
import os
import config


def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions)

MSE = make_scorer(mean_squared_error_, greater_is_better=False)


class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))
        S_train = np.zeros((X.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):

            print('Fitting For Base Model #%d / %d ---', i+1, len(self.base_models))
            for j, (train_idx, test_idx) in enumerate(folds):

                print('--- Fitting For Fold %d / %d ---', j+1, self.n_folds)

                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred

                print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

            print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        print('--- Base Models Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        clf = self.stacker
        clf.fit(S_train, y)

        print('--- Stacker Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

    def preidct(self, X):
        X = np.array(X)
        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state = 1026)
        S_test = np.zeros((X.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((X.shape[0], folds.get_n_splits(X)))
            for j, (train_idx, test_idx) in enumerate(folds.split(X)):
                S_test_i[:, j] = clf.predict(X)[:]
            S_test[:, i] = S_test_i.mean(1)

        clf = self.stacker
        y_pred = clf.predict(S_test)[:]
        return y_pred

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=2016)

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):

            print('Fitting For Base Model #{0} / {1} ---'.format(i+1, len(self.base_models)))

            S_test_i = np.zeros((T.shape[0], folds.get_n_splits(X)))

            for j, (train_idx, test_idx) in enumerate(folds.split(X)):

                print('--- Fitting For Fold #{0} / {1} ---'.format(j+1, self.n_folds))

                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

                print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

            S_test[:, i] = S_test_i.mean(1)

            print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        print('--- Base Models Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        # param_grid = {
        #     'n_estimators': [100],
        #     'learning_rate': [0.45, 0.05, 0.055],
        #     'subsample': [0.72, 0.75, 0.78]
        # }
        param_grid = {
            'n_estimators': [54],
            'learning_rate': [0.1],
            'subsample': [0.81],
            'colsample_bytree' : [0.67],
            'max_depth' : [2]
        }
        grid = GridSearchCV(estimator=self.stacker, param_grid=param_grid, n_jobs=1, cv=5, verbose=20, scoring=MSE)
        grid.fit(S_train, y)

        # a little memo
        message = 'to determine local CV score of #28'

        try:
            print('Param grid:')
            print(param_grid)
            print('Best Params:')
            print(grid.best_params_)
            print('Best CV Score:')
            print(-grid.best_score_)
            print('Best estimator:')
            print(grid.best_estimator_)
            print(message)
        except:
            pass

         #手动调参
        # from imp import reload 
        # while True:
        #     param_grid = config.ensemble_config
        #     model = GridSearchCV(estimator = self.stacker, param_grid = param_grid, n_jobs = 1, cv=5, verbose=20, scoring = MSE)
        #     model.fit(S_train, y)
        
        #     # print('--- Grid Search Completed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
        #     print('Param grid:')
        #     print(param_grid)
        #     print('Best Params:')
        #     print(model.best_params_)
        #     print('Best CV Score:')
        #     print(-model.best_score_)
            
        #     str = input("Modify the parameter: ")
        #     if str ==  's':
        #         break
        #     else:
        #         reload(config)

        print('--- Stacker Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        y_pred = grid.predict(S_test)[:]

        return y_pred


def main(data_file ='feature_selected_3000.xlsx'):
    train_data = pd.read_excel(data_file, sheet_name = 0)
    X_train = train_data.values[:, 0:-1]
    Y_train = train_data.values[:,-1]
    test_data = pd.read_excel(data_file, sheet_name = 1)
    test_index = test_data.index
    X_test = test_data.values[:,:]
    answer_file = "ensemble_answer_" + os.path.basename(data_file).split('.')[0] + ".csv"

    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, random_state = 1024, test_size=0.1)

    print('--- Features Set: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Number of Features: ', len(X_train[0]))

    base_models = [
        RandomForestRegressor(random_state = 0, 
                            n_estimators=151, 
                            min_samples_split = 2,
                            min_samples_leaf = 2,
                            max_features = 0.79,
                            max_depth = 8
        ),
        # ExtraTreesRegressor(
        #     n_jobs=1, random_state=2016, verbose=1,
        #     n_estimators=500, max_features=12
        # ),
        GradientBoostingRegressor(learning_rate = 0.1, 
                                random_state = 0, 
                                n_estimators=30, 
                                min_samples_split = 2,
                                min_samples_leaf = 8,
                                max_features = 0.79,
                                subsample = 0.78,
                                max_depth = 5
        ),
        XGBRegressor(n_estimators = 66,
                        gamma = 0, 
                        learning_rate = 0.1,
                        subsample = 0.81,
                        colsample_bytree = 0.61,
                        max_depth = 3,
                        random_state=0
    )
    ]
    ensemble = Ensemble(
        n_folds = 5,
        stacker = XGBRegressor(learning_rate = 0.1),
        base_models = base_models
    )

    y_pred = ensemble.fit_predict(X=x_train, y=y_train, T=X_test)

    pd.DataFrame({'id': test_index, 'y': y_pred}).to_csv(os.path.join(config.base_path, "answer", answer_file), index=False, header = False)

    print('--- Submission Generated: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

if __name__ == '__main__':
    data_file = "E:\\天池比赛\\IM-contest\\data\\feature_selected_A_3000.xlsx"
    main(data_file)