from __future__ import division
from collections import defaultdict
import re
import numpy as np

from nltk.util import ngrams
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words as nltk_corpus

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

from featureExtraction import FeatureExtraction

class BigFiveFeatureExtraction(FeatureExtraction):

    TWEET_LEN_MAX = 140

    def __init__(self, users, truth_users, stopwords_file, swagwords_file, emotion_words_files):
        self.swag_words = self.txt_file_to_list(swagwords_file)
        self.positive_words = self.txt_file_to_list(emotion_words_files['positive_words_file'])
        self.negative_words = self.txt_file_to_list(emotion_words_files['negative_words_file'])
        self.anger_words = self.txt_file_to_list(emotion_words_files['anger_words_file'])
        self.anticipation_words = self.txt_file_to_list(emotion_words_files['anticipation_words_file'])
        self.disgust_words = self.txt_file_to_list(emotion_words_files['disgust_words_file'])
        self.fear_words = self.txt_file_to_list(emotion_words_files['fear_words_file'])
        self.joy_words = self.txt_file_to_list(emotion_words_files['joy_words_file'])
        self.sadness_words = self.txt_file_to_list(emotion_words_files['sadness_words_file'])
        self.surprise_words = self.txt_file_to_list(emotion_words_files['surprise_words_file'])
        self.trust_words = self.txt_file_to_list(emotion_words_files['trust_words_file'])

        super(BigFiveFeatureExtraction, self).__init__(users, truth_users, stopwords_file)