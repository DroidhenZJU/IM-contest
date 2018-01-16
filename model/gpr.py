from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import math
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import sys
head_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.append(head_path)

import config
def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions)

MSE = make_scorer(mean_squared_error_, greater_is_better=False)


def main(data_file = "../data/feature_selected_A_3000.xlsx"):
    start_time = time.time()
    train_data = pd.read_excel(data_file, sheet_name = 0)
    X_train = train_data.values[:, 0:-1]
    Y_train = train_data.values[:,-1]
    test_data = pd.read_excel(data_file, sheet_name = 1)
    test_index = test_data.index
    X_test = test_data.values[:,:]
    answer_file = "gpr_answer_" + os.path.basename(data_file).split('.')[0] + ".csv"

    print('--- Features Set: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Number of Features: ', len(X_train[0]))

    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, random_state = 1024, test_size=0.1)

    # kernel = C(0.1, (0.001,0.1))*RBF(0.5,(1e-4,10)) 
    reg = GaussianProcessRegressor(random_state = 0)

    #手动调参
    from imp import reload 
    while True:
        param_grid = config.gpr_config
        model = GridSearchCV(estimator = reg, param_grid = param_grid, n_jobs = 1, cv=10, verbose=20, scoring = MSE)
        model.fit(X_train, Y_train)
    
        # print('--- Grid Search Completed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
        print('Param grid:')
        print(param_grid)
        print('Best Params:')
        print(model.best_params_)
        print('Best CV Score:')
        print(-model.best_score_)
        
        str = input("Modify the parameter: ")
        if str ==  's':
            break
        else:
            reload(config)
  
    # y_p = model.predict(x_test)
    # print("-----eval MSE-----")
    # print(mean_squared_error_(y_test, y_p))
    y_pred = model.predict(X_test)
    print("----Average of Result : %f----" %y_pred.mean())
    pd.DataFrame({'id': test_index, 'y': y_pred}).to_csv(os.path.join(config.base_path,"answer", "all_data_" + answer_file), index=False, header=False)

    print('--- Result Generated: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

if  __name__ == "__main__":
    data_file = "E:\\天池比赛\\IM-contest\\data\\feature_selected_A_1000_category.xlsx"
    main(data_file)