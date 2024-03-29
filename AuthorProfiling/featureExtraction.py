from __future__ import division
from collections import defaultdict
import re
import numpy as np
from nltk.util import ngrams
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

class FeatureExtraction(object):

    TWEET_LEN_MAX = 140

    def __init__(self, users, truth_users, stopwords_file):
        self.users = users
        self.truth_users = truth_users
        self.stopwords =  self.txt_file_to_list(stopwords_file)
        self.train_coeff = 0.7
        self.y_column = 1
        self.sorted_users = collections.OrderedDict(sorted(users.items()))

    # opens .txt file that contains words in each row and returns all words as list
    def txt_file_to_list(self, txt_file):
        list = []
        with open(txt_file) as f:
            data = f.readlines()
            for line in data:
                list.append(line.strip())
        return list


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
                if word.lower().strip() in self.stopwords:
                    count = count + 1
        return count/len(input)

    # returns count of words in input tweets that apeear in word_list
    # can be used for multiple features, such as: count swag, count frequent male words...
    def count_feature_from_file(self, tweets, word_list):
        count = 0
        total_word_count=0
        for tweet in tweets:
            for word in tweet.split():
                total_word_count+=1
                word = re.sub(r"[^A-Za-z'-/]", "", word).lower().strip()
                for word in re.split('-|/|,',word):
                    word=re.sub('[.()]', '', word)
                    if str(word) in word_list: # some problems with comparing unicode to string, so I added this str conversion
                        count += 1
        return count/total_word_count # normalize with total_word_count or with len(tweets)

    # return count of uppercase words in all tweets
    def uppercase_words_count(self, input):
        return len(re.findall(r'\b[A-Z]{2,}\b', ' '.join(input)))/len(input)

    def word_count(self, input):
        count = 0
        for word in input.split():
            count = count + 1
        return count


    def char_count(self, input):
        count = 0
        for char in input:
            count += 1
        return count


    def three_dot_count(self,input):
        return len(re.findall('\.\.\.+',' '.join(input)))


    def exclamation_overload_count(self, input):
        return len(re.findall('!!+',' '.join(input)))

    def question_mark_overload_count(self,input):
        return len(re.findall('\?\?+', ' '.join(input)))

    def punctuation_count(self, input):
        return len(re.findall('[?.!]',' '.join(input)))


    def emoticon_count(self,input):
        return len(re.findall('[:;<][3/)\'P*D(]', ' '.join(input)))


    def quotation_count(self, input):
        return len(re.findall('"', ' '.join(input)))


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


    def count_words_longer_than_6_letters(self, input):
        return len(re.findall(r'\b\w{6,}\b', ' '.join(input))) / len(input)



    # append features for every user from ngram TF-IDF matrix
    def append_ngram_tfidf_features(self, ngram_tfidf_matrix, structural_features):
        row_idx = 0
        for key in self.sorted_users.keys():
            for value in ngram_tfidf_matrix[row_idx]:
                structural_features[key].append(value)
            row_idx += 1
        return structural_features


    #returns tfidf matrix for trigrams in dataset
    def get_trigrams_tf_idf(self, input , feature_num):
        trigram_vectorizer = TfidfVectorizer(tokenizer=self.tokens_trigrams,ngram_range=(1, 1),stop_words=self.stopwords, max_features=feature_num)
        X = trigram_vectorizer.fit_transform(input)
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

        shuffled_data = defaultdict(list)
        keys = shuffle(list(data.keys()))
        for key in keys:
            shuffled_data[key] = data[key]

        len_data = len(data.keys())
        train_num = int( len_data * self.train_coeff)
        data_x = np.zeros(shape=(len_data, feature_number))
        data_y = np.zeros(shape=len_data)
        for i, value in enumerate(shuffled_data.itervalues()):
            data_x[i] = value[0]
            data_y[i] = value[1]

        scaler = StandardScaler().fit(data_x)
        data_x_scaled = scaler.transform(data_x)
        return data_x_scaled[0:train_num], data_y[0:train_num], data_x_scaled[train_num:], data_y[train_num:]


