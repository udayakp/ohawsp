import pickle
import numpy as np
from model import NLPModel
import pandas as pd
from sklearn.model_selection import train_test_split

model = NLPModel()


def build_model():
    with open('./data/train.tsv') as f:
        data = pd.read_csv(f, sep='\t')
    pos_neg = data[(data['Sentiment'] == 0) | (data['Sentiment'] == 4)]

    pos_neg['Binary'] = pos_neg.apply(
        lambda x: 0 if x['Sentiment'] == 0 else 1, axis=1)

    model.vectorizer_fit(pos_neg.loc[:, 'Phrase'])
    print('Vectorizer fit complete')

    X = model.vectorizer_transform(pos_neg.loc[:, 'Phrase'])
    print('Vectorizer transform complete')
    y = pos_neg.loc[:, 'Binary']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.train(X_train, y_train)
    print('Model training complete')

    model.pickle_clf()
    model.pickle_vectorizer()


clf_path = './data/SentimentClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

vec_path = './data/TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

def predict(msg):
    user_query = msg
    uq_vectorized = model.vectorizer_transform(np.array([user_query]))
    prediction = model.predict(uq_vectorized)
    pred_proba = model.predict_proba(uq_vectorized)
    if prediction == 0:
            pred_text = '&#128542'
    else:
            pred_text = '&#128526'
    confidence = round(pred_proba[0], 3)
    output = {'prediction': pred_text, 'confidence': confidence}
    if(confidence>0.3):
        return pred_text
    else:
        pred_text = '&#128528'
        return pred_text
