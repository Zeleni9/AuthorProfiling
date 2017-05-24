import os
from preprocessing import Preprocess
import time

# Import classes
from ageFeatureExtraction import AgeFeatureExtraction
from genderFeatureExtraction import GenderFeatureExtraction
from extrovertedFeatureExtraction import ExtrovertedFeatureExtraction
from stableFeatureExtraction import StableFeatureExtraction
from agreeableFeatureExtraction import AgreeableFeatureExtraction
from conscientiousFeatureExtraction import ConscientiousFeatureExtraction
from openFeatureExtraction import OpenFeatureExtraction
from regressionModel import RegressionModel
from classificationModel import ClassificationModel

PATH_TO_WORD_FILES= os.getcwd() +  '/word_files/'
STOP_WORDS_PATH= PATH_TO_WORD_FILES + 'stopwords.txt'
SWAG_WORDS_PATH= PATH_TO_WORD_FILES + 'swag_words.txt'
FREQUENT_MALE_WORDS_PATH= PATH_TO_WORD_FILES + 'frequent_male_words.txt'
FREQUENT_FEMALE_WORDS_PATH= PATH_TO_WORD_FILES + 'frequent_female_words.txt'

# for personality traits
POSITIVE_WORDS = PATH_TO_WORD_FILES + 'positive_words.txt'
NEGATIVE_WORDS = PATH_TO_WORD_FILES + 'negative_words.txt'
ANGER_WORDS = PATH_TO_WORD_FILES + 'anger_words.txt'
ANTICIPATION_WORDS = PATH_TO_WORD_FILES + 'anticipation_words.txt'
DISGUST_WORDS = PATH_TO_WORD_FILES + 'disgust_words.txt'
FEAR_WORDS = PATH_TO_WORD_FILES + 'fear_words.txt'
JOY_WORDS = PATH_TO_WORD_FILES + 'joy_words.txt'
SADNESS_WORDS = PATH_TO_WORD_FILES + 'sadness_words.txt'
SURPRISE_WORDS = PATH_TO_WORD_FILES + 'surprise_words.txt'
TRUST_WORDS = PATH_TO_WORD_FILES + 'trust_words.txt'


def main():
    start_time=time.time()
    path = os.getcwd()

    #nltk.download('averaged_perceptron_tagger')
    #nltk.download('punkt')
    pre_process = Preprocess(path)
    pre_process.load_data()
    pre_process.truth_data()
    users, truth_users = pre_process.get_data()

    # dictionary of file paths to .txt files containing words with specific emotion
    emotion_words_files = {'positive_words_file' : POSITIVE_WORDS, 'negative_words_file' : NEGATIVE_WORDS, 'anger_words_file' : ANGER_WORDS, 'anticipation_words_file' : ANTICIPATION_WORDS, 'disgust_words_file' : DISGUST_WORDS,
                   'fear_words_file' : FEAR_WORDS, 'joy_words_file' : JOY_WORDS, 'sadness_words_file' : SADNESS_WORDS, 'surprise_words_file' : SURPRISE_WORDS, 'trust_words_file' : TRUST_WORDS}

    features_list = []
    num_models = 7
    features = AgeFeatureExtraction(users, truth_users, STOP_WORDS_PATH, SWAG_WORDS_PATH)
    features.extract_features()
    features_list.append([features, "c", "Age classification: "])

    features = GenderFeatureExtraction(users, truth_users, STOP_WORDS_PATH, FREQUENT_MALE_WORDS_PATH, FREQUENT_FEMALE_WORDS_PATH)
    features.extract_features()
    features_list.append([features, "c", "Gender classification: "])

    features = ExtrovertedFeatureExtraction(users, truth_users, STOP_WORDS_PATH, SWAG_WORDS_PATH, emotion_words_files)
    features.extract_features()
    features_list.append([features, "r", "Extroverted regression: "])

    features = StableFeatureExtraction(users, truth_users, STOP_WORDS_PATH, SWAG_WORDS_PATH, emotion_words_files)
    features.extract_features()
    features_list.append([features, "r", "Stable regression: "])

    features = AgreeableFeatureExtraction(users, truth_users, STOP_WORDS_PATH, SWAG_WORDS_PATH, emotion_words_files)
    features.extract_features()
    features_list.append([features, "r", "Agreeable regression: "])

    features = ConscientiousFeatureExtraction(users, truth_users, STOP_WORDS_PATH, SWAG_WORDS_PATH, emotion_words_files)
    features.extract_features()
    features_list.append([features, "r", " Conscientious regression:"])

    features = OpenFeatureExtraction(users, truth_users, STOP_WORDS_PATH, SWAG_WORDS_PATH, emotion_words_files)
    features.extract_features()
    features_list.append([features, "r", "Open regression:"])

    iterations = 100
    for i in range(0, num_models):
        model = ""
        if (features_list[i][1] == "c"):
            model = ClassificationModel(features_list[i][0], iterations)
        elif (features_list[i][1] == "r"):
            model = RegressionModel(features_list[i][0], iterations)

        print features_list[i][2]
        model.evaluate_models()
        print


    print ("")
    run_time=time.time()- start_time
    print ("Run Time : " + str(run_time))


# Starting point of program
main()
