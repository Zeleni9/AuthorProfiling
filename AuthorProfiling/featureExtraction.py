from collections import defaultdict
import re
import numpy as np

class FeatureExtraction(object):

    TWEET_LEN_MAX = 140

    def __init__(self, users, truth_users, stopwords_file):
        """
        :param users:
        :param truth_users:
        :param stopwords_file:
        :return:
        """
        self.users = users
        self.truth_users = truth_users
        self.stopwords =  []
        with open(stopwords_file) as f:
            data = f.readlines()
            for line in data:
                self.stopwords.append(line.strip())
        self.train_coeff = 0.7
        self.y_column = 1

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

    def char_count(self, input):
        count = 0
        for char in input:
            count += 1
        return count

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



    # Method joins features dictionary with truth dictionary by user
    def join_users_truth(self, structural_features, transform, type):
        data = defaultdict(list)
        for key, value in self.users.iteritems():
            y_label = transform(self.truth_users[key][type])
            features = structural_features[key]
            data[key] = [features, y_label]      # Appends list of features and y value for each user
        return data




    # Method splitting vector [[features], [label]] into train_x and train_y
    def prepare_data(self, data, feature_number):
        len_data = len(data.keys())
        train_num = int( len_data * self.train_coeff)
        data_x = np.zeros(shape=(len_data, feature_number))
        data_y = np.zeros(shape=(len_data, self.y_column))
        for i, value in enumerate(data.itervalues()):
            data_x[i] = value[0]
            data_y[i] = value[1]

        return data_x[0:train_num], data_y[0:train_num], data_x[train_num:], data_y[train_num:]


