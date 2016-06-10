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

class StableFeatureExtraction(BigFiveFeatureExtraction):
    def __init__(self, users, truth_users, stopwords_file, swagwords_file, emotion_words_files):
        self.structural_features = defaultdict(list)
        self.type = 3
        self.data = defaultdict(list)
        self.perceptron_tagger = PerceptronTagger(load=False)
        self.perceptron_tagger.load(AP_MODEL_LOC)

        super(StableFeatureExtraction, self).__init__(users, truth_users, stopwords_file, swagwords_file, emotion_words_files)


    def extract_features(self):
        docs=[]
        trigram_count = {}

        for key, value in self.sorted_users.iteritems():

            text, url_count = self.process_links(value)
            #self.structural_features[key].append(url_count)

            text, mention_count = self.process_mentions(text)
            #self.structural_features[key].append(mention_count)

            text, hashtag_count = self.process_hashtags(text)
            # self.structural_features[key].append(hashtag_count)

            # uppercase words count
            uppercase_words_count = self.uppercase_words_count(text)
            self.structural_features[key].append(uppercase_words_count)

            # tweet length ratio
            tweet_length_avg = self.tweet_length_avg(value)
            self.structural_features[key].append(tweet_length_avg)

            # anticipation words count
            anticipation_words_count = self.count_feature_from_file(text, self.anticipation_words)
            self.structural_features[key].append(anticipation_words_count)

            # ... count
            three_dot_count=self.three_dot_count(value)
            self.structural_features[key].append(three_dot_count)

            # !!+ count
            exclamation_count = self.exclamation_overload_count(value)
            self.structural_features[key].append(exclamation_count)

            pos_tags = self.get_pos_tags(text)
            F_score = self.calculate_F_Score(pos_tags)
            self.structural_features[key].append(F_score)

            word_count=self.word_count(' '.join(text))
            words_with_possesive_ending_count=self.get_pos_tag_count(pos_tags, "PO") #ie. friend's
            self.structural_features[key].append(words_with_possesive_ending_count/word_count)

            negations_count = self.count_feature_from_file(text, ['no','not','neither','nowhere','none','nor','nope'])
            self.structural_features[key].append(negations_count)

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

    def get_pos_tag_count(self, tagged, pos_tag):
        counts = Counter(tag[:2] for tagged_sentence in tagged for word, tag in tagged_sentence)
        return counts[pos_tag]

    # calculates F_score based on pos tags for each user
    def calculate_F_Score(self, tagged):
        counts = Counter(tag[:2] for tagged_sentence in tagged for word, tag in tagged_sentence)
        F_score = 0.5 * ((counts['NN'] + counts['JJ'] + counts['IN'] + counts['DT']) -
                         (counts['PR'] + counts['WP'] + counts['WD'] + counts['VB'] +
                          counts['MD'] + counts['RB'] + counts['WR'] + counts['UH']) + 100)
        return F_score


    def do_nothing(self, args):
        return float(args)*1.0