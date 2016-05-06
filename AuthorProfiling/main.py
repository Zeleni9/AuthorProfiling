import os
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from nltk.tokenize import TweetTokenizer
import nltk
from nltk import word_tokenize
from nltk.util import ngrams

from sklearn.metrics import accuracy_score

from preprocessing import Preprocess
from ageFeatureExtraction import AgeFeatureExtraction
from genderFeatureExtraction import GenderFeatureExtraction

def main():

    path = os.getcwd()
    print path
    pre_process = Preprocess(path)
    pre_process.load_data()
    pre_process.truth_data()
    users, truth_users = pre_process.get_data()


    features = AgeFeatureExtraction(users, truth_users, '/Users/filipzelic/Documents/Minesweepers_AuthorProfiling/AuthorProfiling/stopwords.txt')
    features.extract_features()
    train_x, train_y, test_x, test_y = features.get_train_test_data()

    log_reg = LogReg()
    log_reg.fit(train_x, train_y)

    svm_clf = svm.SVC()
    svm_clf.fit(train_x, train_y)

    predicted_y = log_reg.predict(test_x)
    score = accuracy_score(test_y, predicted_y)

    predicted_y_svm = svm_clf.predict(test_x)
    score_2 = accuracy_score(test_y, predicted_y_svm)
    print " Score Log Reg : ", score
    print " SVM Score : ", score_2


    #tokenizer = TweetTokenizer()
    #rec = "Ivan doesnt like shcool :)"
    #tokens = tokenizer.tokenize(rec)
    #trigrams = ngrams(tokens, 3)


    #token_trigrams = [' '.join(tri) for tri in trigrams]
    #print token_trigrams
    # Loadanje featur-a

    # Odabir model

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