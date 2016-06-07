import os
import numpy as np
from sklearn.metrics import accuracy_score
from preprocessing import Preprocess
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from ageFeatureExtraction import AgeFeatureExtraction
from genderFeatureExtraction import GenderFeatureExtraction
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import time
import nltk
import re

PATH_TO_PROJECT_DIRECTORY='C:/Users/borna/Desktop/TAR/Minesweepers_AuthorProfiling/AuthorProfiling/'
STOP_WORDS_PATH=PATH_TO_PROJECT_DIRECTORY + 'stopwords.txt'
SWAG_WORDS_PATH=PATH_TO_PROJECT_DIRECTORY + 'swag_words.txt'
FREQUENT_MALE_WORDS_PATH=PATH_TO_PROJECT_DIRECTORY + 'frequent_male_words.txt'
FREQUENT_FEMALE_WORDS_PATH=PATH_TO_PROJECT_DIRECTORY + 'frequent_female_words.txt'

def main():

    start_time=time.time()
    path = os.getcwd()
    # print path
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')

    pre_process = Preprocess(path)
    pre_process.load_data()
    pre_process.truth_data()
    users, truth_users = pre_process.get_data()

    features = AgeFeatureExtraction(users, truth_users, STOP_WORDS_PATH, SWAG_WORDS_PATH)
    #features = GenderFeatureExtraction(users, truth_users, STOP_WORDS_PATH, FREQUENT_MALE_WORDS_PATH, FREQUENT_FEMALE_WORDS_PATH)
    features.extract_features()

    iterations = 100
    score_log_reg = 0
    score_svm = 0
    score_svm_linear = 0
    for i in xrange(0, iterations):
        train_x, train_y, test_x, test_y = features.get_train_test_data()

        log_reg = LogReg()
        log_reg.fit(train_x, train_y)

        svm_clf = svm.SVC()
        svm_clf.fit(train_x, train_y)

        svm_linear = svm.SVC(kernel='linear')
        svm_linear.fit(train_x, train_y)


        predicted_y_log_reg = log_reg.predict(test_x)
        score_log_reg += accuracy_score(test_y, predicted_y_log_reg)

        predicted_y_svm = svm_clf.predict(test_x)
        score_svm += accuracy_score(test_y, predicted_y_svm)

        predicted_y_svm_linear = svm_linear.predict(test_x)
        score_svm_linear += accuracy_score(test_y, predicted_y_svm_linear)

    score_log_reg_avg = score_log_reg / iterations
    score_svm_avg = score_svm / iterations
    score_svm_linear_avg = score_svm_linear / iterations

    print " Score Log Reg : ", score_log_reg_avg
    print " SVM with RBF kernel Score : ", score_svm_avg
    print " SVM with linear kernel Score : ", score_svm_linear_avg


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

    run_time=time.time() - start_time
    print("")
    print ("Run Time: " + str(run_time) + "s")

# Starting point of program
main()