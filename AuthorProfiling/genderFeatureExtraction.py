from __future__ import division
from collections import defaultdict
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from featureExtraction import FeatureExtraction
from nltk.tokenize import TweetTokenizer
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tag import PerceptronTagger
from collections import Counter
from itertools import chain
from nltk.data import find
import time
import re

PICKLE = "averaged_perceptron_tagger.pickle"
AP_MODEL_LOC = 'file:'+str(find('taggers/averaged_perceptron_tagger/'+PICKLE))

class GenderFeatureExtraction(FeatureExtraction):
    def __init__(self, users, truth_users, stopwords_file, frequent_male_words_file, frequent_female_words_file):
        self.structural_features = defaultdict(list)
        self.type = 0
        self.frequent_male_words = []
        self.frequent_female_words = []
        self.perceptron_tagger = PerceptronTagger(load=False)
        self.perceptron_tagger.load(AP_MODEL_LOC)
        self.data = defaultdict(list)

        with open(frequent_male_words_file) as f:
            data = f.readlines()
            for line in data:
                self.frequent_male_words.append(line.strip())

        with open(frequent_female_words_file) as f:
            data = f.readlines()
            for line in data:
                self.frequent_female_words.append(line.strip())

        super(GenderFeatureExtraction, self).__init__(users, truth_users, stopwords_file)

    def extract_features(self):
        docs = []
        trigrams = []
        trigram_count = {}
        unigram_count = {}
        for key, value in self.sorted_users.iteritems():

            text, url_count = self.process_links(value)
            self.structural_features[key].append(url_count)

            text, mention_count = self.process_mentions(text)
            self.structural_features[key].append(mention_count)

            text, hastag_count = self.process_hashtags(text)
            #self.structural_features[key].append(hastag_count)

            # counts most frequent male function words
            frequent_male_function_words_count = self.count_feature_from_file(value, ['the', 'this', 'that','as','one'])#'the', 'this', 'that','as','one'
            self.structural_features[key].append(frequent_male_function_words_count)

            # counts most frequent female function words
            frequent_female_function_words_count = self.count_feature_from_file(value, ['for', 'with', 'she','not','and','in'])#'for', 'with', 'she','not','and','in'
            self.structural_features[key].append(frequent_female_function_words_count)

            # counts words that are most likely to be used by men
            frequent_male_words_count = self.count_feature_from_file(value, self.frequent_male_words)
            #self.structural_features[key].append(frequent_male_words_count)

            # counts words that are most likely to be used by women
            frequent_female_words_count = self.count_feature_from_file(value, self.frequent_female_words)
            #self.structural_features[key].append(frequent_female_words_count)

            # character overload count
            char_count = self.char_count(''.join(value))
            char_overload_count = self.char_overload_count(''.join(value))
            #self.structural_features[key].append(char_overload_count / char_count)

            # !!+ count
            exclamation_count = self.exclamation_overload_count(value)
            #self.structural_features[key].append(exclamation_count / char_count)

            stopwords_count = self.count_stopwords(text)
            #self.structural_features[key].append(stopwords_count)

            pos_tags=self.get_pos_tags_for_user(text)
            F_score=self.calculate_F_Score(pos_tags)
            self.structural_features[key].append(F_score)

            # for trigram in self.tokens_trigrams('||'.join(text)):
            #     trigram_count[trigram] = trigram_count.get(trigram, 0) + 1
            # for unigram in self.tokens_unigrams('||'.join(text)):
            #     unigram_count[unigram] = unigram_count.get(unigram, 0) + 1

            docs.append('||'.join(text))



        # frequent_trigrams = 0
        # for trigram, count in trigram_count.iteritems():
        #     if (count > 0):
        #         frequent_trigrams += 1
        # print frequent_trigrams

        # frequent_unigrams = 0
        # for unigram, count in unigram_count.iteritems():
        #     if (count > 10):
        #         frequent_unigrams += 1
        # print frequent_unigrams;

        #self.structural_features = self.append_ngram_tfidf_features(self.get_trigrams_tf_idf(docs, 450), self.structural_features)
        #self.structural_features = self.append_ngram_tfidf_features(self.get_unigrams_tf_idf(docs, 1000), self.structural_features)

        self.data = self.join_users_truth(self.structural_features, self.transform_gender, self.type)
        self.feature_number = len(self.structural_features.values()[0])

    def get_train_test_data(self):
        return self.prepare_data(self.data, self.feature_number)


    #returns pos tags of all tweets for one user in nested list format - [[pos_tags_first_tweet],[pos_tags_second_tweet], ... ,[pos_tags_last_tweet]]
    def get_pos_tags_for_user(self,input):
        tweet_tokenizer = TweetTokenizer()
        sentences = [tweet_tokenizer.tokenize(re.sub(r'["]', '', tweet)) for tweet in input]
        return self.perceptron_tagger.tag_sents(sentences)


    #calculates F_score based on pos tags for each user
    def calculate_F_Score(self, tagged):
        counts = Counter(tag[:2] for tagged_sentence in tagged for word, tag in tagged_sentence)
        F_score = 0.5 * ((counts['NN'] + counts['JJ'] + counts['IN'] + counts['DT']) -
                         (counts['PR'] + counts['WP'] + counts['WD'] + counts['VB'] +
                          counts['MD'] + counts['RB'] + counts['WR'] + counts['UH']) + 100)
        return F_score

    #def get_gender_preferential_features_for_user

    def transform_gender(self, gender):
        if (gender == "M"):
            return 0
        elif (gender == "F"):
            return 1
