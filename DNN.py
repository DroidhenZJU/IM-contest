from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
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

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_train, self.Y_train, test_size=0.1)
        # scaler = MinMaxScaler()
        # self.x_train = scaler.fit_transform(self.x_train)
        # self.x_test = scaler.fit_transform(self.x_test)

        

        # print(self.X_train.shape[1])
        # print(self.X_test.shape)
        # print(self.Y_train.shape)
        # print(self.X_test[0])
        # print(self.X_train[0])
        
        # self.pipe = Pipeline([('feature_select', SelectKBest(lambda X, Y: np.array(list(map(lambda x:pearsonr(x, Y)[0], X.T))).T, k=6000)),('clf', GradientBoostingRegressor(max_depth = 5))])

    
    def add_layer(self,inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


    def train(self):
        self.pipe.fit(self.x_train, self.y_train)
        y_predict = self.pipe.predict(self.x_test)
        print(y_predict)
        print(self.y_test)
        print((y_predict - self.y_test)**2)
        mse = 0
        mse = ((y_predict - self.y_test)**2).mean()
        print("mse: %.4f" %(mse))

    def DNN_train(self):
        x_data = self.x_train
        y_data = self.y_train.reshape((-1,1))
        print(x_data.shape)
        print(y_data.shape)
        
        xs = tf.placeholder(tf.float32, [None, len(x_data[0])])
        ys = tf.placeholder(tf.float32, [None,1])
        l1 = self.add_layer(xs, len(x_data[0]), 500, activation_function=tf.nn.softmax)
        l2 = self.add_layer(l1, 500, 50, activation_function=tf.nn.softmax)

        prediction = self.add_layer(l2, 50, 1, activation_function=None)
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                    reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)
        for i in range(50):
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            if i % 10 == 0:
                print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

        print(sess.run(loss, feed_dict={xs: self.x_test, ys: self.y_test.reshape((-1,1))}))



    


if __name__ == "__main__":
    p = Predictor("after_pre_process.xlsx")
    
    p.DNN_train()
