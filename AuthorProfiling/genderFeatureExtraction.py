from __future__ import division
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import collections

from featureExtraction import FeatureExtraction

class GenderFeatureExtraction(FeatureExtraction):

    def __init__(self, users, truth_users, stopwords_file):
        self.structural_features = defaultdict(list)
        self.type = 0
        self.data = defaultdict(list)
        super(GenderFeatureExtraction, self).__init__(users, truth_users, stopwords_file)

    def extract_features(self):
        docs=[]
        trigrams=[]
        trigram_count={}
        unigram_count={}
        for key, value in self.sorted_users.iteritems():
            word_count = self.word_count(''.join(value))

            text, url_count = self.process_links(value)
            #self.structural_features[key].append(url_count/word_count)

            text, mention_count = self.process_mentions(text)
            #self.structural_features[key].append(mention_count/word_count)


            text, hastag_count = self.process_hashtags(text)
            #self.structural_features[key].append(hastag_count/word_count)

            stopwords_count = self.count_stopwords(text)
            #self.structural_features[key].append(stopwords_count/word_count)

            for trigram in self.tokens_trigrams('||'.join(text)):
                trigram_count[trigram]=trigram_count.get(trigram,0) + 1
            for unigram in self.tokens_unigrams('||'.join(text)):
                unigram_count[unigram] = unigram_count.get(unigram, 0) + 1

            docs.append('||'.join(text))

            # print self.structural_features
            # punctaion count
            # character overload count
            # swag ratio
            # emoticon ratio (iz baze)
            # duzinu tweetova

        frequent_trigrams=0
        for trigram,count in trigram_count.iteritems():
            if (count>2):
                frequent_trigrams += 1

        frequent_unigrams= 0
        for unigram, count in unigram_count.iteritems():
            if (count>5):
                frequent_unigrams += 1

        X_trigrams = self.get_trigrams_tf_idf(docs,frequent_trigrams)
        X_unigrams= self.get_unigrams_tf_idf(docs,frequent_unigrams)

        row_idx = 0
        for key in self.sorted_users.keys():
            for value in X_trigrams[row_idx]:
                self.structural_features[key].append(value)
            row_idx += 1

        row_idx = 0
        for key in self.sorted_users.keys():
            for value in X_unigrams[row_idx]:
                self.structural_features[key].append(value)
            row_idx += 1

        self.data = self.join_users_truth(self.structural_features, self.transform_gender, self.type)
        self.feature_number = len(self.structural_features.values()[0])



    def get_train_test_data(self):
        return self.prepare_data(self.data, self.feature_number)

    def transform_gender(self, gender):
        if (gender == "M"): return 0
        elif (gender == "F"): return 1