import os
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
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

STOP_WORDS_PATH='C:/Users/borna/Desktop/TAR/Minesweepers_AuthorProfiling/AuthorProfiling/stopwords.txt'

def main():

    path = os.getcwd()
    print path
    pre_process = Preprocess(path)
    pre_process.load_data()
    pre_process.truth_data()
    users, truth_users = pre_process.get_data()


    features = AgeFeatureExtraction(users, truth_users, STOP_WORDS_PATH)
    features.extract_features()
    train_x, train_y, test_x, test_y = features.get_train_test_data()


    log_reg = LogReg()
    log_reg.fit(train_x, train_y)

    svm_clf = svm.SVC()
    svm_clf.fit(train_x, train_y)

    ranfor_clf = RandomForestClassifier()
    ranfor_clf.fit(train_x, train_y)

    predicted_y_log_reg = log_reg.predict(test_x)
    score_log_reg = accuracy_score(test_y, predicted_y_log_reg)

    predicted_y_svm = svm_clf.predict(test_x)
    score_svm = accuracy_score(test_y, predicted_y_svm)

    predicted_y_random_forest = ranfor_clf.predict(test_x)
    score_random_forest = accuracy_score(test_y, predicted_y_random_forest)

    print " Score Log Reg : ", score_log_reg
    print " SVM Score : ", score_svm
    print " Random Forest Score : ", score_random_forest


    # #feature selection tool - doesn't work for SVM.svc
    # rfe = RFE(log_reg, 6)
    # rfe = rfe.fit(train_x, train_y)
    # # summarize the selection of the attributes
    # print(rfe.support_)
    # print(rfe.ranking_)

    # model = ExtraTreesClassifier()
    # model.fit(train_x, train_y)
    # # display the relative importance of each attribute
    # print(model.feature_importances_)


    #initialize classifiers
    #log_reg = LogReg()
    #svm_clf = svm.SVC()
    #gnb_clf = GaussianNB()
    #ranfor_clf = RandomForestClassifier()
    #clfs = {'logistic regression' : log_reg, 'linear SVM' : svm_clf, 'GaussianNB' : gnb_clf, 'random forest' : ranfor_clf}
    #for clf in clfs:
    #    scores = cross_validation.cross_val_score(clfs[clf], df[all_features], df['label'], cv=10)
    #    print clf + ' : ' + str(scores.mean())


# Starting point of program
main()