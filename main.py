import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB


class LanguageDetector():

    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(ngram_range=(1,2), max_features=1000, preprocessor=self._remove_noise)

    def _remove_noise(self, document):
        noise_pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+"]))
        clean_text = re.sub(noise_pattern, "", document)
        return clean_text

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)

import requests

data_url = 'https://raw.githubusercontent.com/nsorros/pyLanguage/master/data.csv'
datafile = requests.get(data_url).text
dataset = [(line[:-3], line[-2:]) for line in datafile.split('\n')]

X, y = zip(*dataset)
XTrain, XTest, yTrain, yTest = train_test_split(X, y, random_state=1)

language_detector = LanguageDetector()
language_detector.fit(XTrain, yTrain)
print(language_detector.predict('This is not right'))
print(language_detector.score(XTest, yTest))3
