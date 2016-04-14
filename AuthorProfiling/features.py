from __future__ import division
import nltk
import re
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from collections import defaultdict
class Features:

    def __init__(self, users, truth_users, stopwords_file):
        self.users = users
        self.truth_users = truth_users
        self.structural_features = defaultdict(list)
        self.stopwords =  []
        with open(stopwords_file) as f:
            data = f.readlines()
            for line in data:
                self.stopwords.append(line.strip())

    def clean_data(self):

        for key, value in self.users.iteritems():
            clean_lines = []



            word_count = self.word_count(''.join(value))

            text, url_count = self.process_links(value)
            self.structural_features[key].append(url_count/word_count)

            text, mention_count = self.process_mentions(text)
            self.structural_features[key].append(mention_count/word_count)


            text, hastag_count = self.process_hashtags(text)
            self.structural_features[key].append(hastag_count/word_count)

            stopwords_count = self.count_stopwords(text)
            self.structural_features[key].append(stopwords_count/word_count)

            # punctaion count
            # character overload count
            # swag ratio
            # emoticon ratio (iz baze)
            # duzinu tweetova




            #for line in value:




                # removing -> This is a tweet with a url: http://t.co/0DlGChTBIx

                # other cleaning

             #   clean_lines.append(result)
            #self.users[key] = clean_lines



    def process_links(self, input):
        (result, count) = re.subn(r"http\S+", "", '\n'.join(input), flags=re.MULTILINE)
        return result.split('\n'), count

    def process_mentions(self, input):
        (result, count) = re.subn(r"@username\s*", "", '\n'.join(input), flags=re.MULTILINE)
        return result.split('\n'), count

    def process_hashtags(self, input):
        (result, count) = re.subn(r"#", "", '\n'.join(input), flags=re.MULTILINE)
        return result.split('\n'), count

    def count_stopwords(self, input):
        count = 0
        for tweet in input:
            for word in tweet.split(' '):
                if word.strip() in self.stopwords:
                    count = count + 1
        return count


    def word_count(self, input):
        count = 0
        for word in input.split(' '):
            count = count + 1
        return count

    def prepare_data(self):
        train_num = int( len(self.structural_features.keys()) * 0.7)
        train_x = np.zeros(shape=(len(self.structural_features.keys()), 4))
        train_y = np.zeros(shape=(len(self.structural_features.keys()), 1))
        counter = 0
        for key, value in self.structural_features.iteritems():

            if counter == 0:
                print key
            train_x[counter] = value
            counter += 1

        counter = 0
        for key, value in self.truth_users.iteritems():

            if counter == 0:
                print key
            if value[0] == 'M':
                val = 1
            else :
                val = 0
            train_y[counter] = val # gender
            counter += 1

        return train_x[0:train_num], train_y[0:train_num], train_x[train_num:], train_y[train_num:]