import os
import numpy as np
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, grid_search
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from preprocessing import Preprocess
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from ageFeatureExtraction import AgeFeatureExtraction
from genderFeatureExtraction import GenderFeatureExtraction
from sklearn.linear_model import RandomizedLasso

STOP_WORDS_PATH='/Users/filipzelic/Documents/Minesweepers_AuthorProfiling/AuthorProfiling/stopwords.txt'
SWAG_WORDS_PATH='/Users/filipzelic/Documents/Minesweepers_AuthorProfiling/AuthorProfiling/swag_words.txt'
FREQUENT_MALE_WORDS_PATH="/Users/filipzelic/Documents/Minesweepers_AuthorProfiling/AuthorProfiling/frequent_male_words.txt"
FREQUENT_FEMALE_WORDS_PATH='/Users/filipzelic/Documents/Minesweepers_AuthorProfiling/AuthorProfiling/frequent_female_words.txt'

def main():

    path = os.getcwd()
    #print path
    pre_process = Preprocess(path)
    pre_process.load_data()
    pre_process.truth_data()
    users, truth_users = pre_process.get_data()

    features = AgeFeatureExtraction(users, truth_users, STOP_WORDS_PATH, SWAG_WORDS_PATH)
    #features = GenderFeatureExtraction(users, truth_users, STOP_WORDS_PATH, FREQUENT_MALE_WORDS_PATH, FREQUENT_FEMALE_WORDS_PATH)
    features.extract_features()

    iterations = 15
    score_log_reg = 0
    score_svm = 0
    score_random_forest = 0
    score_svr = 0
    score_linearSVC = 0
    score_linearSVR = 0

    #parameters = {'kernel':['rbf'], 'C':[2**(-5), 2**(-3), 2**(-1), 2**(1), 2**(3), 2**(5), 2**(7), 2**(9)], 'gamma':[2**(-9), 2**(-7), 2**(-5), 2**(-3), 2**(-1), 2**(1), 2**(3)] }
    for i in xrange(0, iterations):
        train_x, train_y, test_x, test_y = features.get_train_test_data()

        log_reg = LogReg()
        log_reg.fit(train_x, train_y)

        svm_clf = svm.SVC()
        svm_clf.fit(train_x, train_y)

        ranfor_clf = RandomForestClassifier()
        ranfor_clf.fit(train_x, train_y)

       # grid_search_SVC = grid_search.GridSearchCV(svm_clf, parameters)
       # grid_search_SVC.fit(train_x, train_y)

        predicted_y_log_reg = log_reg.predict(test_x)
        score_log_reg += accuracy_score(test_y, predicted_y_log_reg)

        predicted_y_svm = svm_clf.predict(test_x)
        score_svm += accuracy_score(test_y, predicted_y_svm)

        predicted_y_random_forest = ranfor_clf.predict(test_x)
        score_random_forest += accuracy_score(test_y, predicted_y_random_forest)


    score_log_reg_avg = score_log_reg / iterations
    score_svm_avg = score_svm / iterations
    score_random_forest_avg = score_random_forest / iterations


    print " Score Log Reg : ", score_log_reg_avg
    print " SVM Score : ", score_svm_avg
    print " Random Forest Score : ", score_random_forest_avg

    #
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


    #initialize classifiers
    print ""
    print "Cross validation cv=10 on train data"
    train_x, train_y, test_x, test_y = features.get_train_test_data()
    log_reg = LogReg()
    svm_clf = svm.SVC()
    ranfor_clf = RandomForestClassifier()
    clfs = {'SVM SVC(rbf)' : svm_clf, 'random forest' : ranfor_clf, 'logistic regression' : log_reg}
    for clf in clfs:
        scores = cross_validation.cross_val_score(clfs[clf], train_x, train_y, cv=10)
        print clf + ' : ' + str(scores.mean())


# Starting point of program
main()