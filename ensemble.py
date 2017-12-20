import numpy as np
from sklearn.externals import joblib

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.base_estimator = weak_classifier
        self.n_estimators = n_weakers_limit
        self.quality_estimators = 0
        self.estimators = {}
        self.alpha = {}

    def is_good_enough(self, t):
        ''''' 
            the 1 to t weak classifer come together 
        '''
        self.prob_sum += self.estimators[t].predict_proba(self.X)[:, 1].flatten() * self.alpha[t]
        # print self.prob_sum
        pre_y = np.sign(self.prob_sum - np.ones(self.y.shape))
        t = (pre_y != self.y).sum()
        is_good_enough = True if t == 0 else False
        return is_good_enough

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        self.X = np.array(X)
        self.y = np.array(y)
        self.prob_sum = np.zeros(self.y.shape)
        self.sample_weight = np.ones((self.X.shape[0], 1)).flatten() / self.X.shape[0]
        for i in range(self.n_estimators):
            self.estimators.setdefault(i)
            self.alpha.setdefault(i)
        for i in range(self.n_estimators):

            self.estimators[i] = self.base_estimator.fit(self.X, self.y)
            epsilon = 1.0 - self.estimators[i].score(self.X, self.y, self.sample_weight)
            self.alpha[i] = 0.5 * np.log((1.0 - epsilon) / max(epsilon, 1e-16))
            sg = self.estimators[i].predict_proba(self.X)[:, 1]
            Z = self.sample_weight * np.exp(- self.alpha[i] * self.y * sg.transpose())
            self.sample_weight = (Z / Z.sum()).flatten()
            self.quality_estimators = i
            if self.is_good_enough(i):
                print("%d weak classifier is enough to  make the error to 0" % (i + 1))
                break

    def predict_scores(self, X_test):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        prob_sum = - np.ones(len(X_test))
        for i in range(self.quality_estimators + 1):
            prob_sum += self.estimators[i].predict_proba(X_test)[:, 1].flatten() * self.alpha[i]
        return prob_sum


    def predict(self, X_test, threshold = 0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        prob_sum = - np.ones(len(X_test))
        for i in range(self.quality_estimators + 1):
            prob_sum += self.estimators[i].predict_proba(X_test)[:, 1].flatten() * self.alpha[i]
        pre_y = np.sign(prob_sum)
        return pre_y

    @staticmethod
    def save(model, filename):
        joblib.dump(model, filename)
        print("Classifier saved to {}".format(filename))

    @staticmethod
    def load(filename):
        return joblib.load(filename)
