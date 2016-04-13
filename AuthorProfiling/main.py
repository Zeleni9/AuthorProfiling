import os
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation

from preprocessing import Preprocessing
from features import Features

def main():

    path = os.getcwd()
    preProcess = Preprocessing(path)
    preProcess.load_data()
    preProcess.clean_data()
    preProcess.truth_data()
    users, truth_users = preProcess.get_data()


    # Loadanje featur-a


    # Odabir model

    #initialize classifiers
    log_reg = LogReg()
    svm_clf = svm.SVC()
    gnb_clf = GaussianNB()
    ranfor_clf = RandomForestClassifier()

    #clfs = {'logistic regression' : log_reg, 'linear SVM' : svm_clf, 'GaussianNB' : gnb_clf, 'random forest' : ranfor_clf}
    #for clf in clfs:
    #    scores = cross_validation.cross_val_score(clfs[clf], df[all_features], df['label'], cv=10)
    #    print clf + ' : ' + str(scores.mean())


# Starting point of program
main()