import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import nltk
nltk.download('stopwords')

news_dataset = pd.read_csv('news.csv')

news_dataset.head()

news_dataset = news_dataset.drop(columns=['Unnamed: 0'])
encoder = LabelEncoder()
news_dataset['label'] = encoder.fit_transform(news_dataset['label'])

news_dataset.head()

news_dataset['text'] = news_dataset['title']+ ' '+news_dataset['text']

X = news_dataset.drop(columns=['label','title'], axis=1)
Y = news_dataset['label']

port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
news_dataset['text'] = news_dataset['text'].apply(stemming)
news_dataset['title'] = news_dataset['title'].apply(stemming)

X = news_dataset['text'].values
Y = news_dataset['label'].values

vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

X_new = X_test[3]
prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')

import pickle as pk
pk.dump(model, open('model.pkl', 'wb'))
pk.dump(vectorizer, open('scaler.pkl', 'wb'))