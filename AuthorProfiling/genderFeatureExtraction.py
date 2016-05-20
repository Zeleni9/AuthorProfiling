from __future__ import division
from collections import defaultdict
import collections

from featureExtraction import FeatureExtraction

class GenderFeatureExtraction(FeatureExtraction):

    def __init__(self, users, truth_users, stopwords_file):
        self.structural_features = defaultdict(list)
        self.type = 0
        self.data = defaultdict(list)
        super(GenderFeatureExtraction, self).__init__(users, truth_users, stopwords_file)


    def extract_features(self):

        for key, value in self.sorted_users.iteritems():
            word_count = self.word_count(''.join(value))

            text, url_count = self.process_links(value)
            self.structural_features[key].append(url_count/word_count)

            text, mention_count = self.process_mentions(text)
            self.structural_features[key].append(mention_count/word_count)

            text, hastag_count = self.process_hashtags(text)
            self.structural_features[key].append(hastag_count/word_count)

            stopwords_count = self.count_stopwords(text)
            self.structural_features[key].append(stopwords_count/word_count)

        self.data = self.join_users_truth(self.structural_features, self.transform_gender, self.type)
        self.feature_number = len(self.structural_features.values()[0])


    def get_train_test_data(self):
        return self.prepare_data(self.data, self.feature_number)


    def transform_gender(self, gender):
        if (gender == "M"): return 0
        elif (gender == "F"): return 1