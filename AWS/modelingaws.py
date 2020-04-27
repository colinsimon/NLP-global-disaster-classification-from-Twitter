# Basics
import pandas as pd
import numpy as np
# import time

# Modeling toolbox
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# NLP specific
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC

train = pd.read_csv('train_clean.csv')
test = pd.read_csv('test_clean.csv')

features1 = [
    'keywords_stemmed'
]
features2 = [
    'text_nourl',
    'keywords_stemmed'
]

X = train[features2]
y = train['target']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size = 0.25,
                                                   random_state=42,
                                                   stratify = y)

cvec = CountVectorizer()
xtrain_cv = cvec.fit_transform(X_train['keywords_stemmed'])
xtest_cv = cvec.transform(X_test['keywords_stemmed'])

tfidf = TfidfVectorizer()
xtrain_tf = tfidf.fit_transform(X_train['keywords_stemmed'])
xtest_tf = tfidf.transform(X_test['keywords_stemmed'])

mnb_cv = MultinomialNB()
mnb_tf = MultinomialNB()

mnb_cv.fit(xtrain_cv, y_train)
mnb_tf.fit(xtrain_tf, y_train)

print('MNB train accuracy:' + str(mnb_cv.score(xtrain_cv, y_train)))
print('MNB train accuracy:' + str(mnb_cv.score(xtest_cv, y_test)))
print('MNB train accuracy:' + str(mnb_tf.score(xtrain_tf, y_train)))
print('MNB train accuracy:' + str(mnb_tf.score(xtest_tf, y_test)))

from sklearn.metrics import accuracy_score
svm_cv = SVC()
svm_tf = SVC()
svm_cv.fit(xtrain_cv, y_train)
svm_tf.fit(xtrain_tf, y_train)
y_pred_svmcv = svm_cv.predict(xtest_cv)
y_pred_svmtf = svm_tf.predict(xtest_tf)
print('SVM train accuracy:' + str(accuracy_score(y_test, y_pred_svmcv)))
print('SVM test accuracy:' + str(accuracy_score(y_test, y_pred_svmtf)))

# from sklearn.neural_network import MLPClassifier

# mlp_cv = MLPClassifier()
# mlp_tf = MLPClassifier()
# mlp_cv.fit(xtrain_cv, y_train)
# mlp_tf.fit(xtrain_tf, y_train)
# print('mlp train score:' + str(mlp_cv.score(xtrain_cv, y_train)))
# print('mlp test score:' + str(mlp_tf.score(xtest_cv, y_test)))

scores = []
scores.append('MNB CVEC train accuracy:' + str(mnb_cv.score(xtrain_cv, y_train)))
scores.append('MNB CVEC test accuracy:' + str(mnb_cv.score(xtest_cv, y_train)))
scores.append('MNB TFIDF train accuracy:' + str(mnb_tf.score(xtrain_tf, y_train)))
scores.append('MNB TFIDF test accuracy:' + str(mnb_tf.score(xtest_tf, y_train)))
scores.print('SVM CVEC test accuracy:' + str(accuracy_score(y_test, y_pred_svmcv)))
print('SVM TFIDF test accuracy:' + str(accuracy_score(y_test, y_pred_svmcv)))

AWS_scores = pd.DataFrame(scores)
AWS_scores.to_csv('AWS_scores.csv', index=False)
