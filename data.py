import os
import nltk
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


class DataWorker:
    stopwords = nltk.corpus.stopwords.words('english')

    def __init__(self, filename):
        self.data = pd.read_csv(filename, index_col=0)
        self.data.head()
        print('=> ', self.data.shape[0], 'rows and', self.data.shape[1], 'columns loaded\n')
        print('=> Counts:', self.data['product'].value_counts())
        self.data.dropna(axis=0, inplace=True)
        self.tfidf = TfidfVectorizer(analyzer=self.text_clean)
        self.x_tfidf = self.tfidf.fit_transform(self.data['narrative'])
        print('=> Matrix shape:', self.x_tfidf.shape)

    def get_words_bag(self):
        feature_names = self.tfidf.get_feature_names()
        return pd.DataFrame(self.x_tfidf.toarray(), columns=feature_names)

    # Returning input text as list of fords in lowercase without digits, words with length < 3, stopwords
    def text_clean(self, text):
        clean_words = []

        word_list = text.split()
        for word in word_list:
            word_l = word.lower().strip()
            if word_l.isalpha():
                if len(word_l) > 3:
                    if word_l not in self.stopwords:
                        clean_words.append(word_l)
                    else:
                        continue
        return clean_words
