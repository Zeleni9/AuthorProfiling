from __future__ import  division
from sklearn import svm
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
import math


class RegressionModel():

    def __init__(self, features, iterations):
        self.features = features
        self.iterations = iterations


    def evaluate_models(self):

        # Init mean scores
        score_svr_mean = 0
        score_svr_default_mean=0
        score_lr_mean = 0

        for iteration in xrange(0, self.iterations):

            # Data is shuffled every call on get_train_test_data()
            train_x, train_y, test_x, test_y = self.features.get_train_test_data()

            # SVM with linear kernel
            svr_clf = svm.SVR(kernel='linear')
            svr_clf.fit(train_x, train_y)
            predict_svr_linear = svr_clf.predict(test_x)
            score_svm_linear   = mean_squared_error(test_y, predict_svr_linear)

            # SVM with rbf kernel
            svr_default_clf = svm.SVR(kernel='rbf')
            svr_default_clf.fit(train_x, train_y)
            predict_svr_default = svr_default_clf.predict(test_x)
            score_svr_default   = mean_squared_error(test_y, predict_svr_default)

            # Linear Regression
            linear_clf = linear_model.LinearRegression()
            linear_clf.fit(train_x, train_y)
            lr_predict = linear_clf.predict(test_x)
            score_lr   = mean_squared_error(test_y, lr_predict)

            score_svr_mean         +=   math.sqrt(score_svm_linear)
            score_svr_default_mean +=   math.sqrt(score_svr_default)
            score_lr_mean          +=   math.sqrt(score_lr)


        # Print scores
        print "Evaulation scores using RMSE"
        print "SVM with linear kernel               : ", score_svr_mean/self.iterations
        print "SVM with rbf kernel                  : ", score_svr_default_mean/self.iterations
        print "Linear regression                    : ", score_lr_mean/self.iterations
