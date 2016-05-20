from __future__ import division
from collections import defaultdict

from featureExtraction import FeatureExtraction
from sklearn.preprocessing import StandardScaler

class AgeFeatureExtraction(FeatureExtraction):

    def __init__(self, users, truth_users, stopwords_file):
        self.structural_features = defaultdict(list)
        self.type = 1
        self.data = defaultdict(list)
        super(AgeFeatureExtraction, self).__init__(users, truth_users, stopwords_file)


    def extract_features(self):

        for key, value in self.users.iteritems():

            word_count = self.word_count(''.join(value))
            char_count = self.char_count(''.join(value))

            text, url_count = self.process_links(value)
            self.structural_features[key].append(url_count/word_count)

            text, mention_count = self.process_mentions(text)
            #self.structural_features[key].append(mention_count/word_count)


            text, hastag_count = self.process_hashtags(text)
            #self.structural_features[key].append(hastag_count/word_count)

            stopwords_count = self.count_stopwords(text)
            #self.structural_features[key].append(stopwords_count/word_count)

            # character overload count
            char_overload_count = self.char_overload_count(''.join(value))
            self.structural_features[key].append(char_overload_count/char_count)

            # tweet length ratio
            tweet_length_avg = self.tweet_length_avg(value)
            self.structural_features[key].append(tweet_length_avg / self.TWEET_LEN_MAX)

            # word length ratio
            word_length_avg = self.word_length_avg(value)
            self.structural_features[key].append(word_length_avg)

        self.data = self.join_users_truth(self.structural_features, self.transform_age, self.type)
        self.feature_number = len(self.structural_features.values()[0])


    def get_train_test_data(self):
        return self.prepare_data(self.data, self.feature_number)


    def transform_age(self, age):
        if (age == "18-24"): return 0
        elif (age == "25-34"): return 1
        elif (age == "35-49"): return 2
        elif (age == "50-XX"): return 3
