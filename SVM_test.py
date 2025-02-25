import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn import metrics
from sklearn import svm
import pandas as pd


def get_trained_SVM(training_data, training_labels):
    classifier = svm.SVC(kernel='rbf', random_state=42)

    # trainmodel using the training data
    classifier.fit(training_data, training_labels)

    return classifier


def get_prediction_results(svm, data):
    y_pred = svm.predict(data)

    return y_pred


def get_scores(prediction_results, actual_results):
    acc = metrics.accuracy_score(actual_results, prediction_results)
    pre = metrics.precision_score(actual_results, prediction_results, average='macro')
    rec = metrics.recall_score(actual_results, prediction_results, average='macro')

    return acc, pre, rec


def train_and_test_new_SVM(training_data, test_data, training_labels, test_labels):
    SVM = get_trained_SVM(training_data, training_labels)
    y_pred = SVM.predict(test_data)
    scores = get_scores(y_pred, test_labels)

    return scores


cali = fetch_california_housing()

scaler = StandardScaler()
normalizer = Normalizer()
cali_data = scaler.fit_transform(cali.data)
cali_data = normalizer.fit_transform(cali_data)

labels=["bad", "common", "good"]
cali_labels = pd.cut(cali.target, labels=labels, bins=3)
cali_data_train, cali_data_test, cali_label_train, cali_label_test = train_test_split(cali_data, cali_labels, test_size=0.20, random_state=42)

scores = train_and_test_new_SVM(cali_data_train, cali_data_test, cali_label_train, cali_label_test)
print(scores)

