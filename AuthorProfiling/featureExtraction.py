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

from sklearn.preprocessing import StandardScaler

class FeatureExtraction(object):

    TWEET_LEN_MAX = 140

    def __init__(self, users, truth_users, stopwords_file):
        self.users = users
        self.truth_users = truth_users
        self.stopwords =  []
        with open(stopwords_file) as f:
            data = f.readlines()
            for line in data:
                self.stopwords.append(line.strip())
        self.train_coeff = 0.7
        self.y_column = 1
        self.sorted_users = collections.OrderedDict(sorted(users.items()))


    def process_links(self, input):
        (result, count) = re.subn(r"http\S+", "", '\n'.join(input), flags=re.MULTILINE)
        return result.split('\n'), float(count)/len(input)


    def process_mentions(self, input):
        (result, count) = re.subn(r"@username", "", '\n'.join(input), flags=re.MULTILINE)
        return result.split('\n'), float(count)/len(input)


    def process_hashtags(self, input):
        (result, count) = re.subn(r"#", "", '\n'.join(input), flags=re.MULTILINE)
        return result.split('\n'), float(count)/len(input)


    def count_stopwords(self, input):
        count = 0
        for tweet in input:
            for word in tweet.split(' '):
                if word.strip() in self.stopwords:
                    count = count + 1
        return count/len(input)


    def word_count(self, input):
        count = 0
        for word in input.split(' '):
            count = count + 1
        return count


    def char_count(self, input):
        count = 0
        for char in input:
            count += 1
        return count


    def three_dot_count(self,input):
        count=0;
        for tweet in input:
            count+=tweet.count('...')
        return count


    def exclamation_overload_count(self, input):
        count = 0;
        for tweet in input:
            count += len(re.findall('!!+',tweet))
        return count


    def punctuation_count(self, input):
        count = 0;
        for tweet in input:
            count += len(re.findall('[?.!]', tweet))
        return count


    def emoticon_count(self,input):
        count = 0;
        for tweet in input:
            count += len(re.findall('[:;][/)\'P*D(]', tweet))
            count += len(re.findall('<3',tweet))
        return count


    def quotation_count(self, input):
        count = 0;
        for tweet in input:
            count += len(re.findall('"', tweet))
        return count


    # returns count of character appearance after they repeat three or more time in a row in some string
    def char_overload_count(self, input):
        count = 0
        for idx in xrange(2, len(input)):
            if input[idx] == input[idx - 1] and input[idx] == input[idx - 2]:
                count += 1
        return count


    # returns average tweet length from array like input
    def tweet_length_avg(self, input):
        lengths = [len(i) for i in input]
        return 0 if len(lengths) == 0 else (float(sum(lengths)) / len(lengths))


    # returns average word length in tweets
    def word_length_avg(self, input):
        word_lengths =  []
        for tweet in input:
            for word in tweet.split(' '):
                word_lengths.append(len(word))
        return float(sum(word_lengths)) / len(word_lengths)


    #returns tfidf matrix for trigrams in dataset
    def get_trigrams_tf_idf(self, input , feature_num):
        trigram_vectorizer = TfidfVectorizer(tokenizer=self.tokens_trigrams,ngram_range=(1, 1),stop_words=self.stopwords, max_features=feature_num)
        X = trigram_vectorizer.fit_transform(input)
        # features = trigram_vectorizer.get_feature_names()
        # print features
        return X.toarray()


    #input = all tweets of  one user
    #tweets are separated with '||'"
    #with this method TfidfVectorizer produces trigrams (trigram = string composed of 3 words) for which tfidf values are computed
    def tokens_trigrams(self, input):
        stemmer=PorterStemmer()
        trigrams=set()
        tweets=input.split('||')
        for tweet in tweets:
            tokenizer = RegexpTokenizer(r'[a-z]+')
            tokens = tokenizer.tokenize(tweet.lower())
            filtered_words = [stemmer.stem(word) for word in tokens if not word in self.stopwords]
            for trigram in ngrams(filtered_words, 3):
                trigrams.add(' '.join(trigram))
        return list(trigrams)


    # returns tfidf matrix for unigrams in dataset
    def get_unigrams_tf_idf(self, input, feature_num):
        trigram_vectorizer = TfidfVectorizer(tokenizer=self.tokens_unigrams, ngram_range=(1, 1),
                                             stop_words=self.stopwords, max_features=feature_num)
        X = trigram_vectorizer.fit_transform(input)
        # features = trigram_vectorizer.get_feature_names()
        # print features
        return X.toarray()


    def tokens_unigrams(self, input):
        stemmer = PorterStemmer()
        unigrams = set()
        all_tweets = input.split('||')
        for tweet in all_tweets:
            tokenizer = RegexpTokenizer(r'[a-z]+')
            tokens = tokenizer.tokenize(tweet.lower())
            filtered_words = [stemmer.stem(word) for word in tokens if not word in self.stopwords]
            for unigram in filtered_words:
                unigrams.add(unigram)
        return list(unigrams)


    # Method joins features dictionary with truth dictionary by user
    def join_users_truth(self, structural_features, transform, type):
        data = defaultdict(list)
        for key in self.users.keys():
            y_label = transform(self.truth_users[key][type])
            features = structural_features[key]
            data[key] = [features, y_label]      # Appends list of features and y value for each user
        return data


    # Method splitting vector [[features], [label]] into train_x and train_y
    # Values are normalized with StandardScaler
    def prepare_data(self, data, feature_number):
        len_data = len(data.keys())
        train_num = int( len_data * self.train_coeff)
        data_x = np.zeros(shape=(len_data, feature_number))
        data_y = np.zeros(shape=(len_data, self.y_column))
        for i, value in enumerate(data.itervalues()):
            data_x[i] = value[0]
            data_y[i] = value[1]
        scaler = StandardScaler().fit(data_x)
        data_x_std = scaler.transform(data_x)
        #np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
        #print data_x_std
        return data_x_std[0:train_num], data_y[0:train_num], data_x_std[train_num:], data_y[train_num:]


