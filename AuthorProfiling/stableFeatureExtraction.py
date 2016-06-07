from __future__ import division
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from nltk.tag import PerceptronTagger
from collections import Counter
from nltk.data import find
import re

from featureExtraction import FeatureExtraction

PICKLE = "averaged_perceptron_tagger.pickle"
AP_MODEL_LOC = 'file:'+str(find('taggers/averaged_perceptron_tagger/'+PICKLE))

class StableFeatureExtraction(FeatureExtraction):

    def __init__(self, users, truth_users, stopwords_file, swagwords_file, positive_words_file, negative_words_file):
        self.structural_features = defaultdict(list)
        self.type = 3
        self.data = defaultdict(list)
        self.swag_words = []
        self.positive_words = []
        self.negative_words = []
        self.perceptron_tagger = PerceptronTagger(load=False)
        self.perceptron_tagger.load(AP_MODEL_LOC)

        with open(swagwords_file) as f:
            data = f.readlines()
            for line in data:
                self.swag_words.append(line.strip())

        with open(positive_words_file) as f:
            data = f.readlines()
            for line in data:
                self.positive_words.append(line.strip())

        with open(negative_words_file) as f:
            data = f.readlines()
            for line in data:
                self.negative_words.append(line.strip())

        super(StableFeatureExtraction, self).__init__(users, truth_users, stopwords_file)


    def extract_features(self):
        docs=[]
        trigram_count = {}

        for key, value in self.sorted_users.iteritems():

            text, url_count = self.process_links(value)
            #self.structural_features[key].append(url_count)

            text, mention_count = self.process_mentions(text)
            #self.structural_features[key].append(mention_count)

            text, hashtag_count = self.process_hashtags(text)
            self.structural_features[key].append(hashtag_count)

            # uppercase words count
            uppercase_words_count = self.uppercase_words_count(text)
            #self.structural_features[key].append(uppercase_words_count)

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
            #self.structural_features[key].append(word_length_avg)

            # positive words count
            positive_words_count = self.count_feature_from_file(text, self.positive_words)
            #self.structural_features[key].append(positive_words_count)

            # negative words count
            negative_words_count = self.count_feature_from_file(text, self.negative_words)
            #self.structural_features[key].append(negative_words_count)

            # swag count
            swag_count = self.count_feature_from_file(text, self.swag_words)
            #self.structural_features[key].append(swag_count)

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
            emoticon_count = self.emoticon_count(text)
            #self.structural_features[key].append(emoticon_count)

            pos_tags = self.get_pos_tags(text)
            F_score = self.calculate_F_Score(pos_tags)
            self.structural_features[key].append(F_score)

            # for trigram in self.tokens_trigrams('||'.join(text)):
            #     trigram_count[trigram] = trigram_count.get(trigram, 0) + 1

            docs.append('||'.join(text))


        # frequent_trigrams = 0
        # for trigram, count in trigram_count.iteritems():
        #     if (count > 2):
        #         frequent_trigrams += 1

        #self.structural_features = self.append_ngram_tfidf_features(self.get_trigrams_tf_idf(docs,500), self.structural_features)
        #self.structural_features = self.append_ngram_tfidf_features(self.get_unigrams_tf_idf(docs, 1000), self.structural_features)

        self.data = self.join_users_truth(self.structural_features, self.do_nothing, self.type)
        self.feature_number = len(self.structural_features.values()[0])


    def get_train_test_data(self):
        return self.prepare_data(self.data, self.feature_number)

        # returns pos tags of all tweets for one user in nested list format - [[pos_tags_first_tweet],[pos_tags_second_tweet], ... ,[pos_tags_last_tweet]]
    def get_pos_tags(self, input):
        tweet_tokenizer = TweetTokenizer()
        sentences = [tweet_tokenizer.tokenize(re.sub(r'["]', '', tweet)) for tweet in input]
        return self.perceptron_tagger.tag_sents(sentences)

    # calculates F_score based on pos tags for each user
    def calculate_F_Score(self, tagged):
        counts = Counter(tag[:2] for tagged_sentence in tagged for word, tag in tagged_sentence)
        F_score = 0.5 * ((counts['NN'] + counts['JJ'] + counts['IN'] + counts['DT']) -
                         (counts['PR'] + counts['WP'] + counts['WD'] + counts['VB'] +
                          counts['MD'] + counts['RB'] + counts['WR'] + counts['UH']) + 100)
        return F_score


    def do_nothing(self, args):
        return float(args)*1.0