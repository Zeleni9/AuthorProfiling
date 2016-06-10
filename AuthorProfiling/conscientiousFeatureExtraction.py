from __future__ import division
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from nltk.tag import PerceptronTagger
from collections import Counter
from nltk.data import find
import re

from bigFiveFeatureExtraction import BigFiveFeatureExtraction

PICKLE = "averaged_perceptron_tagger.pickle"
AP_MODEL_LOC = 'file:'+str(find('taggers/averaged_perceptron_tagger/'+PICKLE))

class ConscientiousFeatureExtraction(BigFiveFeatureExtraction):

    def __init__(self, users, truth_users, stopwords_file, swagwords_file, emotion_words_files):
        self.structural_features = defaultdict(list)
        self.type = 5
        self.data = defaultdict(list)
        self.perceptron_tagger = PerceptronTagger(load=False)
        self.perceptron_tagger.load(AP_MODEL_LOC)

        super(ConscientiousFeatureExtraction, self).__init__(users, truth_users, stopwords_file, swagwords_file, emotion_words_files)


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
            self.structural_features[key].append(uppercase_words_count)

            stopwords_count = self.count_stopwords(text)
            self.structural_features[key].append(stopwords_count)

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
            self.structural_features[key].append(positive_words_count)

            # fear words count
            fear_words_count = self.count_feature_from_file(text, self.fear_words)
            # self.structural_features[key].append(fear_words_count)

            # joy words count
            joy_words_count = self.count_feature_from_file(text, self.joy_words)
            self.structural_features[key].append(joy_words_count)

            # sadness words count
            sadness_words_count = self.count_feature_from_file(text, self.sadness_words)
            self.structural_features[key].append(sadness_words_count)

            # surprise words count
            surprise_words_count = self.count_feature_from_file(text, self.surprise_words)
            self.structural_features[key].append(surprise_words_count)

            # trust words count
            trust_words_count = self.count_feature_from_file(text, self.trust_words)
            self.structural_features[key].append(trust_words_count)

            punctuation_count = self.punctuation_count(text)
            #self.structural_features[key].append(punctuation_count)

            # emoticon count
            emoticon_count = self.emoticon_count(text)
            #self.structural_features[key].append(emoticon_count)

            pos_tags = self.get_pos_tags(text)
            F_score = self.calculate_F_Score(pos_tags)
            self.structural_features[key].append(F_score)

            docs.append('||'.join(text))

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