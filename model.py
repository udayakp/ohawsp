from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class NLPModel(object):

    def __init__(self):
        self.clf = MultinomialNB()
        self.vectorizer = TfidfVectorizer()
    def vectorizer_fit(self, X):
        """Fits a TFIDF vectorizer to the text"""
        self.vectorizer.fit(X)

    def vectorizer_transform(self, X):
        """Transform the text data to a sparse TFIDF matrix
        """
        X_transformed = self.vectorizer.transform(X)
        return X_transformed

    def train(self, X, y):
        """Trains the classifier to associate the label with the sparse matrix
        """
        self.clf.fit(X, y)

    def predict_proba(self, X):
        """Returns probability for the binary class '1' in a numpy array
        """
        y_proba = self.clf.predict_proba(X)
        return y_proba[:, 1]

    def predict(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.clf.predict(X)
        return y_pred

    def pickle_vectorizer(self, path='./data/TFIDFVectorizer.pkl'):
        """Saves the trained vectorizer for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            print("Pickled vectorizer at {}".format(path))

    def pickle_clf(self, path='./data/SentimentClassifier.pkl'):
        """Saves the trained classifier for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled classifier at {}".format(path))

    def plot_roc(self, X, y, size_x=12, size_y=12):
        """Plot the ROC curve for X_test and y_test.
        """
        y_pred = model.predict_proba(x_columns)

        fpr, tpr, threshold = roc_curve(y_true, y_pred[:, 1])
        area_under_curve = auc(fpr, tpr)

        # method I: plt
        fig, ax = plt.subplots(figsize=(size_x, size_y))
        model_name = str(type(model)).split('.')[-1].strip(">\'")
        plt.title(f'{model_name} ROC')
        ax.plot(fpr, tpr, 'k', label='AUC = %0.3f' % area_under_curve)

        ax.legend(loc='lower right')
        ax.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

