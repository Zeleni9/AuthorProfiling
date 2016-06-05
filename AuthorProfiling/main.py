import os
import numpy as np
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import re
import nltk
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
from nltk.tokenize import TweetTokenizer
from nltk.tag import PerceptronTagger
from nltk.tag import StanfordPOSTagger
from nltk.tag.stanford import StanfordPOSTagger
import time

PATH_TO_PROJECT_DIRECTORY='C:/Users/borna/Desktop/TAR/Minesweepers_AuthorProfiling/AuthorProfiling/'
STOP_WORDS_PATH=PATH_TO_PROJECT_DIRECTORY + 'stopwords.txt'
SWAG_WORDS_PATH=PATH_TO_PROJECT_DIRECTORY + 'swag_words.txt'
FREQUENT_MALE_WORDS_PATH=PATH_TO_PROJECT_DIRECTORY + 'frequent_male_words.txt'
FREQUENT_FEMALE_WORDS_PATH=PATH_TO_PROJECT_DIRECTORY + 'frequent_female_words.txt'
STANFORD_POS_TAGGER_MODEL = PATH_TO_PROJECT_DIRECTORY + 'stanford-postagger/models/english-bidirectional-distsim.tagger'
STANFORD_POS_TAGGER_JAR = PATH_TO_PROJECT_DIRECTORY+ 'stanford-postagger/stanford-postagger.jar'

def main():

    start_time=time.time()
    path = os.getcwd()
    #print path
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')


    pre_process = Preprocess(path)
    pre_process.load_data()
    pre_process.truth_data()
    users, truth_users = pre_process.get_data()

    #features = AgeFeatureExtraction(users, truth_users, STOP_WORDS_PATH, SWAG_WORDS_PATH)
    features = GenderFeatureExtraction(users, truth_users, STOP_WORDS_PATH, FREQUENT_MALE_WORDS_PATH, FREQUENT_FEMALE_WORDS_PATH)
    features.extract_features()

    iterations = 100
    score_log_reg = 0
    score_svm = 0
    score_random_forest = 0
    for i in xrange(0, iterations):
        train_x, train_y, test_x, test_y = features.get_train_test_data()

        log_reg = LogReg()
        log_reg.fit(train_x, train_y)

        svm_clf = svm.SVC()
        svm_clf.fit(train_x, train_y)

        ranfor_clf = RandomForestClassifier()
        ranfor_clf.fit(train_x, train_y)



        svm_linearSVC = svm.LinearSVC()
        svm_linearSVC.fit(train_x, train_y)

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
    # print ""
    # print "Cross validation cv=10 on train data"
    # train_x, train_y, test_x, test_y = features.get_train_test_data()
    # log_reg = LogReg()
    # svm_clf = svm.SVC()
    # ranfor_clf = RandomForestClassifier()
    # clfs = {'Score Log Reg' : log_reg, 'SVM Score' : svm_clf, 'Random Forest Score' : ranfor_clf}
    # for clf in clfs.keys():
    #     scores = cross_validation.cross_val_score(clfs[clf], train_x, train_y, cv=10)
    #     print clf + ' : ' + str(scores.mean())

    run_time=time.time() - start_time
    print("")
    print ("Run Time: " + str(run_time) + "s")

# Starting point of program
main()