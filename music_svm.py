import random
import time
from pyexpat import features
from sklearn import metrics
from sklearn import svm
import pandas as pd
from our_classes import Track
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from collections import defaultdict

def get_trained_SVM(training_tracks: list):
    classifier = svm.SVC(kernel='rbf', random_state=42, probability=True)

    training_data = [track.features for track in training_tracks]
    training_labels = [track.label for track in training_tracks]

    # train model using the training data
    ova_classifier = OneVsRestClassifier(classifier)
    ova_classifier.fit(training_data, training_labels)

    return ova_classifier

def classify_and_get_conf_scores(svm, track):
    y_pred_ova = svm.predict([track.features])
    y_confidence = svm.predict_proba([track.features])

    print(y_pred_ova, y_confidence)
    return y_pred_ova, y_confidence


def get_scores(prediction_results, actual_results):
    print("pred",prediction_results)
    print("act", actual_results)

    acc = metrics.accuracy_score(actual_results, prediction_results)
    pre = metrics.precision_score(actual_results, prediction_results, average='macro')
    rec = metrics.recall_score(actual_results, prediction_results, average='macro')

    return acc, pre, rec

def train_and_test_new_SVM(training_tracks: list, test_tracks: list):
    SVM = get_trained_SVM(training_tracks)

    predictions_and_confidences = [classify_and_get_conf_scores(SVM, track) for track in test_tracks]
    y_pred = [prediction_results[0] for prediction_results in predictions_and_confidences]
    y_actual = [track.label for track in test_tracks]

    scores = get_scores(y_pred, y_actual)
    return scores

number_of_features = 88
feature_range = 1000
track_to_create = 1000

rand = random.Random()
rand.seed(100)

temp_labels = ["angry", "sad", "happy", "romantic"]
temp_features = [([rand.randrange(0, feature_range) for i in range(number_of_features)]) for i in range(track_to_create)]
print(temp_features[:10])
temp_tracks = [Track(None, rand.choice(temp_labels)) for i in range(track_to_create)]
for i in range(len(temp_tracks)):
    temp_tracks[i].features = temp_features[i]

print(train_and_test_new_SVM(temp_tracks[round(track_to_create*0.8):], temp_tracks[round(track_to_create*0.8):]))


training_tracks, test_tracks = temp_tracks[round(track_to_create*0.8):], temp_tracks[-round(track_to_create*0.8):]


svm = get_trained_SVM(training_tracks)

predictions = []
correct_results = []
for test_track in test_tracks:
    one_prediction = classify_and_get_conf_scores(svm, test_track)[0]
    predictions.append(one_prediction)

    correct_results.append(test_track.label)

scores = get_scores(predictions, correct_results)
print(scores)

def predict_by_label(svm:OneVsRestClassifier, track:list):
    co_results = []



    return co_results


grouped_tracks = defaultdict(list)
for track in test_tracks:
    grouped_tracks[track.label].append(track)

confidence_results = []
for label, track_list in grouped_tracks.items():
        print(label)
        confidence_scores = []
        for track in track_list:
            print(label, track)
            confidences = classify_and_get_conf_scores(svm, track)[1]
            confidence_scores.append(confidences)

        confidence_array = np.array(confidence_scores)
        conf_sum = np.sum(confidence_array, axis=0)
        confidence_results.append((label, conf_sum))

for results in confidence_results:
    print(results)
