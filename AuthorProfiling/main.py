import os
import numpy as np
from sklearn.metrics import accuracy_score
from preprocessing import Preprocess
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import time
import nltk
import re
from sklearn.cross_validation import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_absolute_error
from sklearn.linear_model import Ridge
import math

# Import classes
from ageFeatureExtraction import AgeFeatureExtraction
from genderFeatureExtraction import GenderFeatureExtraction
from extrovertedFeatureExtraction import ExtrovertedFeatureExtraction
from stableFeatureExtraction import StableFeatureExtraction
from agreeableFeatureExtraction import AgreeableFeatureExtraction
from conscientiousFeatureExtraction import ConscientiousFeatureExtraction
from openFeatureExtraction import OpenFeatureExtraction

PATH_TO_WORD_FILES= 'C:/Users/User/Desktop/APT/Minesweepers_AuthorProfiling/AuthorProfiling/word_files/'
STOP_WORDS_PATH= PATH_TO_WORD_FILES + 'stopwords.txt'
SWAG_WORDS_PATH= PATH_TO_WORD_FILES + 'swag_words.txt'
FREQUENT_MALE_WORDS_PATH= PATH_TO_WORD_FILES + 'frequent_male_words.txt'
FREQUENT_FEMALE_WORDS_PATH= PATH_TO_WORD_FILES + 'frequent_female_words.txt'

#for personality traits
POSITIVE_WORDS = PATH_TO_WORD_FILES + 'positive_words.txt'
NEGATIVE_WORDS = PATH_TO_WORD_FILES + 'negative_words.txt'
ANGER_WORDS = PATH_TO_WORD_FILES + 'anger_words.txt'
ANTICIPATION_WORDS = PATH_TO_WORD_FILES + 'anticipation_words.txt'
DISGUST_WORDS = PATH_TO_WORD_FILES + 'disgust_words.txt'
FEAR_WORDS = PATH_TO_WORD_FILES + 'fear_words.txt'
JOY_WORDS = PATH_TO_WORD_FILES + 'joy_words.txt'
SADNESS_WORDS = PATH_TO_WORD_FILES + 'sadness_words.txt'
SURPRISE_WORDS = PATH_TO_WORD_FILES + 'surprise_words.txt'
TRUST_WORDS = PATH_TO_WORD_FILES + 'trust_words.txt'


def main():
    start_time=time.time()
    path = os.getcwd()
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    pre_process = Preprocess(path)
    pre_process.load_data()
    pre_process.truth_data()
    users, truth_users = pre_process.get_data()

    # dictionary of file paths to .txt files containing words with specific emotion
    emotion_words_files = {'positive_words_file' : POSITIVE_WORDS, 'negative_words_file' : NEGATIVE_WORDS, 'anger_words_file' : ANGER_WORDS, 'anticipation_words_file' : ANTICIPATION_WORDS, 'disgust_words_file' : DISGUST_WORDS,
                   'fear_words_file' : FEAR_WORDS, 'joy_words_file' : JOY_WORDS, 'sadness_words_file' : SADNESS_WORDS, 'surprise_words_file' : SURPRISE_WORDS, 'trust_words_file' : TRUST_WORDS}


    #features = AgeFeatureExtraction(users, truth_users, STOP_WORDS_PATH, SWAG_WORDS_PATH)
    #features = GenderFeatureExtraction(users, truth_users, STOP_WORDS_PATH, FREQUENT_MALE_WORDS_PATH, FREQUENT_FEMALE_WORDS_PATH)
    #features = ExtrovertedFeatureExtraction(users, truth_users, STOP_WORDS_PATH, SWAG_WORDS_PATH, emotion_words_files)
    #features = StableFeatureExtraction(users, truth_users, STOP_WORDS_PATH, SWAG_WORDS_PATH, emotion_words_files)
    #features = AgreeableFeatureExtraction(users, truth_users, STOP_WORDS_PATH, SWAG_WORDS_PATH, emotion_words_files)
    features = ConscientiousFeatureExtraction(users, truth_users, STOP_WORDS_PATH, SWAG_WORDS_PATH, emotion_words_files)
    #features = OpenFeatureExtraction(users, truth_users, STOP_WORDS_PATH, SWAG_WORDS_PATH, emotion_words_files)
    features.extract_features()

    iterations = 100
    score_svr_mean = 0
    score_svr_default_mean=0
    score_ridge_mean = 0
    for itera in xrange(0, iterations):
        train_x, train_y, test_x, test_y = features.get_train_test_data()

        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        svr_clf = GridSearchCV(svm.SVR(), parameters)
        svr_clf.fit(train_x, train_y)

        predict_svr_optimized = svr_clf.predict(test_x)
        score_svr_optimized = mean_squared_error(test_y, predict_svr_optimized)

        svr_default_clf =svm.SVR()
        svr_default_clf.fit(train_x, train_y)

        predict_svr_default = svr_default_clf.predict(test_x)
        score_svr_default = mean_squared_error(test_y, predict_svr_default)

        ridge_clf = Ridge()
        ridge_clf.fit(train_x, train_y)

        predict_ridge = ridge_clf.predict(test_x)
        score_ridge = mean_squared_error(test_y, predict_ridge)


        score_svr_mean += math.sqrt(score_svr_optimized)
        score_svr_default_mean+=math.sqrt(score_svr_default)
        score_ridge_mean += math.sqrt(score_ridge)


    print "Error Optimized SVM                  ", score_svr_mean/iterations
    print "Error Default SVM                  ", score_svr_default_mean / iterations
    print "Error Ridge:                     ", score_ridge_mean/iterations


    # iterations = 100
    # score_log_reg = 0
    # score_svm = 0
    # score_svm_linear = 0
    # for i in xrange(0, iterations):
    #     train_x, train_y, test_x, test_y = features.get_train_test_data()
    #
    #     log_reg = LogReg()
    #     log_reg.fit(train_x, train_y)
    #
    #     svm_clf = svm.SVC()
    #     svm_clf.fit(train_x, train_y)
    #
    #     svm_linear = svm.SVC(kernel='linear')
    #     svm_linear.fit(train_x, train_y)
    #
    #     predicted_y_log_reg = log_reg.predict(test_x)
    #     score_log_reg += accuracy_score(test_y, predicted_y_log_reg)
    #
    #     predicted_y_svm = svm_clf.predict(test_x)
    #     score_svm += accuracy_score(test_y, predicted_y_svm)
    #
    #     predicted_y_svm_linear = svm_linear.predict(test_x)
    #     score_svm_linear += accuracy_score(test_y, predicted_y_svm_linear)
    #
    # score_log_reg_avg = score_log_reg / iterations
    # score_svm_avg = score_svm / iterations
    # score_svm_linear_avg = score_svm_linear / iterations
    #
    # print " Score Log Reg : ", score_log_reg_avg
    # print " SVM with RBF kernel Score : ", score_svm_avg
    # print " SVM with linear kernel Score : ", score_svm_linear_avg

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

    print ("")
    run_time=time.time()- start_time
    print ("Run Time : " + str(run_time))


# Starting point of program
main()