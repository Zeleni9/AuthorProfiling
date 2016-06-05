import os
import numpy as np
from sklearn.metrics import accuracy_score
from preprocessing import Preprocess
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from ageFeatureExtraction import AgeFeatureExtraction
from genderFeatureExtraction import GenderFeatureExtraction
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression

STOP_WORDS_PATH='/Users/filipzelic/Documents/Minesweepers_AuthorProfiling/AuthorProfiling/stopwords.txt'
SWAG_WORDS_PATH='/Users/filipzelic/Documents/Minesweepers_AuthorProfiling/AuthorProfiling/swag_words.txt'
FREQUENT_MALE_WORDS_PATH="/Users/filipzelic/Documents/Minesweepers_AuthorProfiling/AuthorProfiling/frequent_male_words.txt"
FREQUENT_FEMALE_WORDS_PATH='/Users/filipzelic/Documents/Minesweepers_AuthorProfiling/AuthorProfiling/frequent_female_words.txt'

def main():

    path = os.getcwd()
    pre_process = Preprocess(path)
    pre_process.load_data()
    pre_process.truth_data()
    users, truth_users = pre_process.get_data()

    #features = AgeFeatureExtraction(users, truth_users, STOP_WORDS_PATH, SWAG_WORDS_PATH)
    features = GenderFeatureExtraction(users, truth_users, STOP_WORDS_PATH, FREQUENT_MALE_WORDS_PATH, FREQUENT_FEMALE_WORDS_PATH)
    features.extract_features()


    iterations = 10
    score_svm_accumulate = 0
    score_svm_best_parameters = 0
    score_log_accumulate = 0
    score_log_default = 0
    for iter in xrange(0, iterations):

        train_x, train_y, test_x, test_y = features.get_train_test_data()
        n_iter_search = 25

        # Model selection SVM [hiperparameter optimization]
        print "Support Vector Machine model: "
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        kernel_list = ['linear', 'rbf']
        param_grid_SVC = dict(kernel=kernel_list, C=C_range, gamma=gamma_range)
        validation = StratifiedKFold(train_y, n_folds=10)
        grid_svm = RandomizedSearchCV(svm.SVC(), param_distributions=param_grid_SVC, cv=validation, n_iter=n_iter_search)
        grid_svm.fit(train_x, train_y)

        print ""
        print "Best parameters for svm.SVC(): ", grid_svm.best_params_


        # Evaluation on test set with found best hiperparameters of SVM model
        predict_y_grid = grid_svm.predict(test_x)
        score_grid_svm = accuracy_score(test_y, predict_y_grid)

        # Evaluation on test set with best hiperparameters earlier estimated to perform best
        svm_clf = svm.SVC(kernel='linear', C=0.01, gamma=(1**(-9)))
        svm_clf.fit(train_x, train_y)
        predict_y_svm = svm_clf.predict(test_x)
        score_svm = accuracy_score(test_y, predict_y_svm)

        # Accumulate for mean score on 10 iterations
        score_svm_accumulate += score_grid_svm
        score_svm_best_parameters += score_svm

        # Output of scores
        print "Evalutation on test set using Grid search optimized model SVM:                   ", score_grid_svm
        print "Evalutation on test set using best hiperparameters found eralier:                ", score_svm
        print ""


        # Model selection Logistic Regression [hiperparameter optimization]
        print "Logistic Regression model: "
        C_range = np.linspace(0.001, 1000, 100)
        param_grid = {'C': C_range , 'penalty': ['l1', 'l2']}
        validation = StratifiedKFold(train_y, n_folds=10)
        grid_log = RandomizedSearchCV(LogisticRegression(), param_distributions=param_grid, cv=validation, n_iter=n_iter_search)
        grid_log.fit(train_x, train_y)

        print "Best params for logistic regression: ", grid_log.best_params_

        # Evaluation on test set with found best hiperparameters of LogisticRegression model
        predict_y_log_grid = grid_log.predict(test_x)
        score_grid_log = accuracy_score(test_y, predict_y_log_grid)

        # Evaluation on test set with best hiperparameters earlier estimated to perform best
        log_reg_clf = LogisticRegression()
        log_reg_clf.fit(train_x,train_y)

        predict_y_log = log_reg_clf.predict(test_x)
        score_log = accuracy_score(test_y, predict_y_log)

        # Accumulate for mean score on 10 iterations
        score_log_accumulate += score_grid_log
        score_log_default += score_log

        # Output of scores
        print "Evalutation on test set using Grid search optimized model LogisticRegression:    ", score_grid_log
        print "Evalutation on test set using default LogisticRegression:                        ", score_log
        print ""


    print "Iterations : ", iterations
    print "SVM score gridSearch         : ",    score_svm_accumulate / iterations
    print "SVM score bestParamFixed     : ",    score_svm_best_parameters/iterations
    print "----------------------------------------------------"
    print "LogReg score gridSearch      : ",    score_log_accumulate / iterations
    print "LogReg score default         : ",    score_log_default / iterations


    # #feature selection tool - doesn't work for SVM.svc
    # rfe = RFE(log_reg, 6)
    # rfe = rfe.fit(train_x, train_y)
    # # summarize the selection of the attributes
    # print(rfe.support_)
    # print(rfe.ranking_)
    #
    # model = ExtraTreesClassifier()
    # model.fit(train_x, train_y)
    # # display the relative importance of each attribute
    # print(model.feature_importances_)

# Starting point of program
main()