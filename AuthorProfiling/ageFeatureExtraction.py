from __future__ import division
from collections import defaultdict
from featureExtraction import FeatureExtraction

class AgeFeatureExtraction(FeatureExtraction):

    def __init__(self, users, truth_users, stopwords_file, swagwords_file):
        self.structural_features = defaultdict(list)
        self.type = 1
        self.data = defaultdict(list)
        self.swag_words = self.txt_file_to_list(swagwords_file)

        super(AgeFeatureExtraction, self).__init__(users, truth_users, stopwords_file)


    def extract_features(self):
        docs=[]
        trigram_count = {}

        for key, value in self.sorted_users.iteritems():

            text, url_count = self.process_links(value)
            self.structural_features[key].append(url_count)

            # uppercase words count
            uppercase_words_count = self.uppercase_words_count(value)
            self.structural_features[key].append(uppercase_words_count)

            # character overload count
            char_count = self.char_count(''.join(value))
            char_overload_count = self.char_overload_count(''.join(value))
            self.structural_features[key].append(char_overload_count/char_count)

            # tweet length ratio
            tweet_length_avg = self.tweet_length_avg(value)
            self.structural_features[key].append(tweet_length_avg)

            # word length ratio
            word_length_avg = self.word_length_avg(value)
            self.structural_features[key].append(word_length_avg)

            # ...+ count
            three_dot_count=self.three_dot_count(text)
            self.structural_features[key].append(three_dot_count)

            # " count
            quotation_count = self.quotation_count(value)
            self.structural_features[key].append(quotation_count)

            # emoticon count
            emoticon_count = self.emoticon_count(value)
            self.structural_features[key].append(emoticon_count)

            for trigram in self.tokens_trigrams('||'.join(text)):
                trigram_count[trigram] = trigram_count.get(trigram, 0) + 1

            docs.append('||'.join(text))


        frequent_trigrams = 0
        for trigram, count in trigram_count.iteritems():
            if (count > 2):
                frequent_trigrams += 1

        self.structural_features = self.append_ngram_tfidf_features(self.get_trigrams_tf_idf(docs, frequent_trigrams), self.structural_features)

        self.data = self.join_users_truth(self.structural_features, self.transform_age, self.type)
        self.feature_number = len(self.structural_features.values()[0])


    def get_train_test_data(self):
        return self.prepare_data(self.data, self.feature_number)


    def transform_age(self, age):
        if (age == "18-24"): return 0
        elif (age == "25-34"): return 1
        elif (age == "35-49"): return 2
        elif (age == "50-XX"): return 3
