from __future__ import division
from collections import defaultdict

from featureExtraction import FeatureExtraction

class ExtrovertedFeatureExtraction(FeatureExtraction):

    def __init__(self, users, truth_users, stopwords_file):
        self.structural_features = defaultdict(list)
        self.type = 2
        self.data = defaultdict(list)
        super(ExtrovertedFeatureExtraction, self).__init__(users, truth_users, stopwords_file)


    def extract_features(self):
        docs=[]
        trigram_count = {}

        for key, value in self.sorted_users.iteritems():

            # uppercase words count
            uppercase_words_count = self.uppercase_words_count(value)
            self.structural_features[key].append(uppercase_words_count)

            text, url_count = self.process_links(value)
            self.structural_features[key].append(url_count)

            text, mention_count = self.process_mentions(text)
            #self.structural_features[key].append(mention_count)

            text, hastag_count = self.process_hashtags(text)
            #self.structural_features[key].append(hastag_count)

            stopwords_count = self.count_stopwords(text)
            #self.structural_features[key].append(stopwords_count)

            # character overload count
            char_count = self.char_count(''.join(value))
            char_overload_count = self.char_overload_count(''.join(value))
            #self.structural_features[key].append(char_overload_count/char_count)

            # tweet length ratio
            tweet_length_avg = self.tweet_length_avg(value)
            self.structural_features[key].append(tweet_length_avg)

            # word length ratio
            word_length_avg = self.word_length_avg(value)
            self.structural_features[key].append(word_length_avg)

            # ... count
            three_dot_count=self.three_dot_count(value)
            #self.structural_features[key].append(three_dot_count)

            # !!+ count
            exclamation_count = self.exclamation_overload_count(value)
            #self.structural_features[key].append(exclamation_count)

            # " count
            quotation_count = self.quotation_count(value)
            #self.structural_features[key].append(quotation_count)

            punctuation_count = self.punctuation_count(text)
            #self.structural_features[key].append(punctuation_count)

            # emoticon count
            emoticon_count = self.emoticon_count(value)
            #self.structural_features[key].append(emoticon_count)

            for trigram in self.tokens_trigrams('||'.join(text)):
                trigram_count[trigram] = trigram_count.get(trigram, 0) + 1

            docs.append('||'.join(text))


        frequent_trigrams = 0
        for trigram, count in trigram_count.iteritems():
            if (count > 2):
                frequent_trigrams += 1
        #print frequent_trigrams

        self.structural_features = self.append_ngram_tfidf_features(self.get_trigrams_tf_idf(docs, frequent_trigrams), self.structural_features)
        #self.structural_features = self.append_ngram_tfidf_features(self.get_unigrams_tf_idf(docs, 1000), self.structural_features)

        self.data = self.join_users_truth(self.structural_features, self.do_nothing, self.type)
        self.feature_number = len(self.structural_features.values()[0])


    def get_train_test_data(self):
        return self.prepare_data(self.data, self.feature_number)


    def do_nothing(self, args):
        return float(args)*1.0