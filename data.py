import nltk
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


class DataWorker:
    stopwords = nltk.corpus.stopwords.words('english')

    def __init__(self, filename=""):
        self.count_vect = CountVectorizer(analyzer=self.text_clean)
        self.classifier = MultinomialNB()
        self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',
                                     ngram_range=(1, 2), stop_words='english', analyzer=self.text_clean)
        self.tfidf_transformer = TfidfTransformer()
        if filename != "":
            self.data = pd.read_csv(filename, index_col=0)
            self.data.head()
            self.data = self.data[['product', 'narrative']].sample(n=70000)
            print('=> ', self.data.shape[0], 'rows and', self.data.shape[1], 'columns loaded\n')
            print('=> Counts:\n', self.data['product'].value_counts())
            self.data.dropna(axis=0, inplace=True)
            self.features = self.tfidf.fit_transform(self.data['narrative'])
            print('=> Matrix shape:', self.features.shape)

    def get_words_bag(self):
        feature_names = self.tfidf.get_feature_names()
        return pd.DataFrame(self.features.toarray(), columns=feature_names)

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

    def create_predictor(self):
        X_train, X_test, y_train, y_test = train_test_split(self.data['narrative'], self.data['product'],
                                                            random_state=0)
        X_train_counts = self.count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        x_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        self.classifier = MultinomialNB().fit(x_train_tfidf, y_train)

    def save_predictor(self):
        with open('count_vect.pickle', 'wb') as picklefile:
            pickle.dump(self.count_vect, picklefile)

        with open('text_classifier.pickle', 'wb') as picklefile:
            pickle.dump(self.classifier, picklefile)

    def load_predictor(self):
        with open('count_vect.pickle', 'rb') as vectorizer:
            self.count_vect = pickle.load(vectorizer)

        with open('text_classifier.pickle', 'rb') as training_model:
            self.classifier = pickle.load(training_model)

    def predict(self, data):
        prediction = self.classifier.predict(data)

        print(prediction)
        return prediction

    def predict_string(self, text):
        prediction = self.classifier.predict(self.count_vect.transform([text]))
        return prediction
