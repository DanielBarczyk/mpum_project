import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import re

personalities = ["enfj", "enfp", "entj", "entp", "esfj", "esfp", "estj", "estp", "infj", "infp", "intj", "intp", "isfj", "isfp", "istj", "istp"]

class Data():
    """ Use get_data() to get features and labels for Twitter MBTI data """

    def __init__(self, distribution="even") -> None:
        file = open("twitter_MBTI.csv", "r")
        self.raw_data = np.array(list(csv.reader(file, delimiter=",")))
        file.close()

        self.raw_data = self.raw_data[1:] # Remove CSV header
        if distribution == "even":
            self.raw_data = self.__get_even_distribution()
        elif distribution == "full":
            pass
        else:
            self.raw_data = self.raw_data[np.random.choice(self.raw_data.shape[0], 2000, replace=False)]
        
        self.data = self.raw_data[:, 1]
        self.data = np.array([re.sub(r"\S*https?:\S*", "", data) for data in self.data]) # Remove urls from text
        self.data = np.array([re.sub(r"@[^\s]+", "", data) for data in self.data]) # Remove usernames from text
        
        self.labels = self.raw_data[:, 2]
        self.all_labels = np.unique(self.labels)
        self.labels = np.array([np.where(self.all_labels == label) for label in self.labels]).flatten()

        self.__vectorize()

    def __get_even_distribution(self):
        result = np.empty(shape=(0,3))
        base_data = np.random.shuffle(self.raw_data)
        for personality in personalities:
            temp = [data for data in base_data if data[2] == personality]
            result = np.vstack([result, temp[:81]])
        return result

    def __vectorize(self):
        self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')
        self.features = self.tfidf.fit_transform(self.data.tolist()).toarray()
    
    def info(self):
        """ Displays bar chart with number of elements per label """
        count = [np.count_nonzero(self.labels == label) for label in range(0, 16)]
        plt.bar(self.all_labels, count)
        plt.xlabel("MBTI Personality")
        plt.ylabel("Count")
        plt.show()

    def get_most_popular_features(self, n=20):
        """ Prints n most popular features for each label """
        for category in self.all_labels:
            features_chi2 = chi2(self.features, self.labels == category)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(self.tfidf.get_feature_names_out())[indices]
            print(category)
            print(feature_names[-n:])

    def index_to_label(self, idx):
        return self.all_labels[idx]

    def get_data(self):
        """ Returns features and labels for Twitter MBTI data """
        return self.features, self.labels
